# Mine Expenditure Extractor

A task-focused RAG pipeline that parses mining technical reports
(e.g. NI 43-101, JORC) and extracts structured capital & operating cost
breakdowns for the primary mine described in each document.

Built to handle **20+ PDFs** end-to-end: parse → chunk → embed →
retrieve → extract → save JSON + markdown per file.

---

## Why this project

Technical reports for mining operations are long (often 100-300 pages),
cost figures live in ruleless text-aligned tables that defeat most
off-the-shelf table extractors, and the report frequently mentions
multiple sister mines. For each PDF we need to:

1. Identify which mine the report is actually **about**.
2. Pull the capital and operating cost breakdown for that mine.
3. Do all of the above **fast** (Marker-based pipelines take 30-60s per
   PDF just for parsing — unworkable at 20 docs).

---

## Architecture

```
PDF  ─►  PyMuPDF layout extraction  ─►  LLM table cleanup (only on
                                         pages flagged by heuristic)
                                         │
                                         ▼
                                    Markdown
                                         │
            markdown-aware chunker  ◄────┘   (keeps tables intact)
                    │
                    ▼
     Sentence-Transformers embeddings  ─►  ChromaDB (persistent)
                                              │
                                              ▼
                          Cross-encoder reranker  ◄─  Query
                                              │
                                              ▼
                                       Top-k snippets
                                              │
                                              ▼
                     LLM mine detector  ─►  primary mine
                                              │
                                              ▼
                     LLM cost extractor  ─►  validated Pydantic JSON
                                              │
                                              ▼
                                   outputs/<pdf>.json + .md
```

### Design choices

| Concern | Choice | Why |
|---|---|---|
| PDF parsing | **PyMuPDF** `get_text("blocks")` | 20-50× faster than Marker, no GPU needed |
| Table handling | Heuristic → LLM enhancer only when needed | Most pages never hit an LLM — cost & latency stay low |
| Chunking | Markdown-aware, never splits tables | Ensures cost totals + breakdown travel together |
| Embeddings | `all-MiniLM-L6-v2` | Fast, compact, 384-dim, cosine similarity |
| Vector store | Chroma persistent client | Zero-config, file-backed |
| Reranking | `ms-marco-MiniLM-L-6-v2` cross-encoder | Sharper top-k after bi-encoder recall |
| LLM | Groq `llama-3.3-70b-versatile` | Free tier, fast, JSON mode |
| Outputs | Pydantic models → JSON + markdown | Machine + human readable, validated |
| Caching | Parsed pages cached by content hash | Re-runs are essentially free |

---

## Project layout

```
mine_expenditure_extractor/
├── run.py                       # CLI entry point
├── settings.py                  # Config (env-driven)
├── requirements.txt
├── .env.example
├── README.md
├── pdfs/                        # ← Drop your 20 PDFs here
├── outputs/                     # Per-PDF JSON + markdown reports
├── cache/                       # Parsed-page cache (auto-generated)
├── vector_index/                # Chroma persistence (auto-generated)
├── logs/                        # Rotating log files (auto-generated)
└── mine_extractor/
    ├── logging_config.py
    ├── llm_client.py            # Groq client with retry/backoff
    ├── parsing/
    │   ├── page_classifier.py   # Heuristic: does this page look tabular?
    │   ├── table_enhancer.py    # LLM rewrites aligned text → markdown tables
    │   └── pdf_to_markdown.py   # PyMuPDF parser + disk cache
    ├── indexing/
    │   ├── chunker.py           # Markdown-aware, table-preserving splitter
    │   ├── embedder.py          # sentence-transformers wrapper
    │   └── vector_index.py      # Chroma wrapper
    ├── retrieval/
    │   ├── reranker.py          # Cross-encoder reranker
    │   └── searcher.py          # Two-stage retrieval
    ├── extraction/
    │   ├── schemas.py           # Pydantic output models
    │   ├── prompts.py           # System + user templates
    │   ├── mine_detector.py     # Identify + choose primary mine
    │   └── cost_extractor.py    # Extract capex / opex breakdown
    └── pipeline/
        ├── ingest_pipeline.py   # PDF → chunks → vectors
        └── extract_pipeline.py  # Retrieve → LLM → save JSON + md
```

---

## Setup

### 1. Clone / unzip and create a virtual environment

```bash
cd mine_expenditure_extractor
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The first run of the embedder / reranker will download the HuggingFace
models (~100 MB total).

### 3. Configure the LLM key

```bash
cp .env.example .env
# then edit .env and paste your Groq API key
```

Get a free Groq key at https://console.groq.com/keys. If you prefer a
different OpenAI-compatible endpoint (OpenAI, Together.ai, OpenRouter,
…) you only need to change `GROQ_BASE_URL` and `LLM_MODEL` in `.env`.

---

## Usage

### Drop PDFs in

```
pdfs/
├── TR_Greenhills-V15.pdf
├── another-mine-report.pdf
└── ...up to however many you have
```

### One-shot (ingest + extract)

```bash
python run.py process
```

That's it. For each PDF you'll get:

```
outputs/
├── TR_Greenhills-V15.json    # Validated Pydantic dump
└── TR_Greenhills-V15.md      # Human-readable report
```

### Fine-grained commands

```bash
# Index only (no LLM extraction yet)
python run.py ingest

# Index specific files / directories
python run.py ingest --files pdfs/TR_Greenhills-V15.pdf pdfs/other.pdf
python run.py ingest --files /path/to/pdf_dir

# Extract everything already indexed
python run.py extract

# Extract for one file only
python run.py extract --source TR_Greenhills-V15.pdf

# Inspect the index
python run.py list

# Wipe the vector store and start over
python run.py clear
```

---

## What the output looks like

After running `process` on the bundled sample PDF you'll see (truncated):

`outputs/TR_Greenhills-V15.md`

```markdown
# Expenditure Extraction — Greenhills Coal Operation

- Source document: `TR_Greenhills-V15.pdf`
- Location: British Columbia, Canada
- Operator: Teck Resources Limited
- Report type: NI 43-101 Technical Report
- Currency: CAD
- Confidence: high

## Capital Costs

**Total:** 3,431,571 thousand CAD (LOM 2020-2065)

| Category               | Amount     | Units          | Period         |
|------------------------|-----------:|----------------|----------------|
| Mining equipment       | 1,922,582  | thousand CAD   | LOM 2020-2065  |
| Plant & infrastructure | 221,735    | thousand CAD   | LOM 2020-2065  |
| Infrastructure         | 306,084    | thousand CAD   | LOM 2020-2065  |
| Pit development        | 370,798    | thousand CAD   | LOM 2020-2065  |
| Sustainability         | 610,372    | thousand CAD   | LOM 2020-2065  |

## Operating Costs

**Total:** 128.32 CAD/t clean coal

| Category              | Amount | Units              |
|-----------------------|-------:|--------------------|
| Mining and processing | 87.78  | CAD/t clean coal   |
| Transportation        | 34.43  | CAD/t clean coal   |
| Other                 | 6.11   | CAD/t clean coal   |
```

And the companion `TR_Greenhills-V15.json` is the same data as a
machine-readable structure (see `mine_extractor/extraction/schemas.py`
for the full schema), including every retrieval snippet the LLM saw —
which is useful for audit and debugging.

---

## Performance notes

Measured on the sample PDF (132 pages, 2.9 MB):

| Stage | Time |
|---|---|
| Parse PDF (no LLM) | **~0.5 s** |
| Parse PDF (LLM enhancement on ~6 table pages) | ~3-6 s |
| Chunk + embed (first run) | ~3 s (+ model load) |
| Chunk + embed (warm) | ~1 s |
| Extraction (2 Groq calls) | ~2-4 s |

A full `process` run on 20 PDFs typically completes in under **3
minutes** once models are warm, compared to 15-30+ minutes for
Marker-based pipelines.

---

## Tuning

Almost everything lives in `settings.py` or `.env`:

- `CHUNK_SIZE`, `CHUNK_OVERLAP` — chunk granularity
- `LLM_TABLE_ENHANCE` — set `false` to skip LLM table cleanup entirely
- `LLM_MODEL`, `LLM_TEMPERATURE` — swap models or make outputs more creative
- `EMBEDDING_DEVICE` — set to `cuda` if you have a GPU

---

## Troubleshooting

**`GROQ_API_KEY is not set`** — copy `.env.example` to `.env` and fill in the key.

**Extraction returns low confidence with empty breakdowns** — try
raising `search_top_k` / `final_top_k` in `settings.py`, or inspect the
`retrieval_snippets` array in the output JSON to see what the LLM
actually saw. If the right pages aren't there, the report may use
non-standard terminology — adjust `_COST_QUERIES` in
`mine_extractor/extraction/cost_extractor.py`.

**Table enhancement is slow** — set `LLM_TABLE_ENHANCE=false` in
`.env`. Text extraction alone is usually sufficient for cost tables in
NI 43-101 style reports because the LLM extractor at the end of the
pipeline can read layout-aligned text just fine.

**Re-ingestion looks like it did nothing** — parsed pages are cached in
`cache/`. Delete the cache directory to force a fresh parse, or change
the PDF's content and the cache key will invalidate automatically.

---

## Contributing / dev setup

Install the dev extras and the pre-commit hooks once:

```bash
pip install -e ".[dev]"
pre-commit install
```

That wires up the following on every `git commit`:

- `pre-commit-hooks` — trailing whitespace, EOF newline, YAML/TOML/JSON
  validity, merge-conflict markers, >2 MB file guard, private-key guard
- `ruff` — lint + auto-fix (replaces flake8 + isort + pyupgrade)
- `ruff-format` — opinionated formatter (replaces black)
- `gitleaks` — scans staged changes for accidentally committed secrets

Run against the whole repo any time:

```bash
pre-commit run --all-files
```

Ruff is also configured in `pyproject.toml`, so editor integrations
(VS Code "Ruff" extension, PyCharm Ruff plugin, etc.) will pick up the
same rules automatically.

---

## License

MIT — do as you please with this code.
