# MINEA Evaluation Package

MINEA (Multiple Infused Needle Extraction Accuracy) evaluation for triple extraction quality.

## Structure

```
minea/
├── __init__.py              # Package initialization
├── needles.py               # Needle generation and injection logic
├── extraction.py            # Triple extraction runner
├── evaluation.py            # Needle matching criteria (exact, keyword, semantic, judge)
├── content_filter.py        # Content quality filtering
├── data_loader.py           # Document loading and preprocessing
├── docs/                    # Documentation
│   ├── MINEA_EXPLANATION.md
│   ├── MINEA_FINAL_STATUS.md
│   ├── MINEA_POLISH_SUMMARY.md
│   ├── MINEA_README.md
│   ├── MINEA_SUMMARY.md
│   └── ...
└── tests/                   # Test files
    ├── test_content_filter.py
    ├── test_needle_fraction_enforcement.py
    └── ...
```

## Usage

Run the main evaluation:

```bash
cd /home/ubuntu/X-RAG/src
PYTHONPATH=. python -m experiments.minea_triple_eval
```

Or use the convenience scripts:

```bash
./run_minea_corrected.sh
./run_minea_strict_niah.sh
```

## Module Overview

### `needles.py`
- `generate_needles()`: Generate synthetic needle triples using LLM
- `inject_needles()`: Inject needles into text with NIAH compliance (10-30% fraction)
- `_GenerateNeedleTriples`: DSPy signature for needle generation

### `extraction.py`
- `run_extraction_on_enriched_chunk()`: Run triple extraction on needle-enriched chunks
- Handles retries, error handling, and response parsing

### `evaluation.py`
- `exact_match()`: String-based exact matching
- `keyword_match()`: Keyword overlap matching
- `semantic_match()`: Embedding-based semantic similarity
- `llm_judge_match()`: LLM-as-judge contextual matching
- `evaluate_needle_capture()`: Orchestrates all evaluation criteria
- `_JudgeNeedleMatch`: DSPy signature for LLM judge

### `content_filter.py`
- `is_valid_content_for_extraction()`: Filters out non-extractable content
- Rejects: ToC, reference lists, author lists, figure captions

### `data_loader.py`
- `load_sample_documents()`: Load and filter chunks from preprocessing cache
- Uses SaT segmentation for consistency with other experiments

## Configuration

All configuration is in `minea_triple_eval.py`:

```python
CANDIDATE_MODELS = {...}           # Models to evaluate
NEEDLE_GEN_MODEL = "..."           # Model for needle generation
JUDGE_MODEL = "..."                # Model for LLM judge
SAMPLE_SIZE = 10                   # Number of documents
NEEDLES_PER_DOC = 5                # Needles per document
KEYWORD_MATCH_THRESHOLD = 0.5      # Keyword overlap threshold
SEMANTIC_SIM_THRESHOLD = 0.75      # Cosine similarity threshold
```

## Refactoring Notes

**Previous structure:**
- Single 1183-line file (`minea_triple_eval.py`)
- All logic mixed together

**Current structure:**
- Main file: 300 lines (orchestration only)
- 5 focused modules (100-300 lines each)
- Cleaner separation of concerns
- Easier to test and maintain

**Benefits:**
- Each module has a single responsibility
- Easier to understand and modify individual components
- Better code reuse potential
- Documentation and tests organized separately
