# MINEA Refactoring Summary

**Date:** 2026-05-04  
**Branch:** leanrag-faithful-eval

## Overview

Refactored the MINEA triple extraction evaluation experiment to improve maintainability and organization.

## Changes

### Code Reduction
- **Before:** 1183 lines in single file (`minea_triple_eval.py`)
- **After:** 399 lines in main file + 963 lines across 5 focused modules
- **Net reduction:** Main orchestration file is **66% smaller**

### New Structure

```
experiments/
├── minea/                           # New subpackage
│   ├── __init__.py                 # Package initialization (6 lines)
│   ├── needles.py                  # Needle generation & injection (271 lines)
│   ├── extraction.py               # Triple extraction runner (113 lines)
│   ├── evaluation.py               # Matching criteria (334 lines)
│   ├── content_filter.py           # Content quality filter (87 lines)
│   ├── data_loader.py              # Document loading (152 lines)
│   ├── README.md                   # Package documentation
│   ├── docs/                       # Documentation files (10 markdown files)
│   │   ├── MINEA_EXPLANATION.md
│   │   ├── MINEA_FINAL_STATUS.md
│   │   ├── MINEA_POLISH_SUMMARY.md
│   │   └── ...
│   └── tests/                      # Test files (7 test scripts)
│       ├── test_content_filter.py
│       ├── test_needle_fraction_enforcement.py
│       └── ...
├── minea_triple_eval.py            # Main experiment (399 lines, refactored)
├── minea_triple_eval.py.backup     # Original file (preserved)
└── ...
```

### Module Responsibilities

**`needles.py`**
- Needle triple generation via DSPy
- Needle injection with NIAH compliance (10-30% fraction)
- Random placement within document chunks
- Contains `_GenerateNeedleTriples` DSPy signature

**`extraction.py`**
- Triple extraction on enriched chunks
- Retry logic and error handling
- Response parsing and validation
- Uses `_ExtractTriples` from xrag.dataset_processing

**`evaluation.py`**
- Four matching criteria: exact, keyword, semantic, judge
- Embedding-based semantic similarity
- LLM-as-judge contextual matching
- Aggregate MINEA score computation
- Contains `_JudgeNeedleMatch` DSPy signature

**`content_filter.py`**
- Content quality validation
- Filters structural sections (ToC, references, captions)
- Ensures extractable narrative content

**`data_loader.py`**
- Document sampling and loading
- SaT segmentation via preprocessing cache
- Content filtering integration
- Document selection logic

**`minea_triple_eval.py` (main)**
- Configuration constants
- Experiment orchestration
- Results aggregation and output
- Summary table printing

## Benefits

### Maintainability
- Each module has a single, clear responsibility
- Easier to understand and modify individual components
- Reduced cognitive load when working on specific features

### Testability
- Individual modules can be tested in isolation
- Test files organized in dedicated directory
- Easier to add new test cases

### Organization
- Documentation consolidated in `docs/` subdirectory
- Test files moved to `tests/` subdirectory
- Experiments folder decluttered (4 Python files vs. 11 previously)

### Reusability
- Modules can be imported independently
- Functions take explicit parameters (no global dependencies)
- Clear interfaces between components

## Backward Compatibility

✅ **Fully compatible** - no changes to usage:

```bash
# Still works exactly as before
cd /home/ubuntu/X-RAG/src
PYTHONPATH=. python -m experiments.minea_triple_eval

# Shell scripts unchanged
./run_minea_corrected.sh
./run_minea_strict_niah.sh
```

Output format and results file location remain unchanged.

## Files Moved

### Documentation → `minea/docs/`
- MINEA_CODE_REVIEW.md
- MINEA_EXPLANATION.md
- MINEA_FAITHFULNESS_ASSESSMENT.md
- MINEA_FINAL_STATUS.md
- MINEA_POLISH_SUMMARY.md
- MINEA_QUICKREF.md
- MINEA_README.md
- MINEA_SUMMARY.md
- NIAH_ENFORCEMENT_UPDATE.md
- REMAINING_ISSUES.md

### Tests → `minea/tests/`
- test_content_filter.py
- test_critical_fixes.py
- test_minea_fixes.py
- test_minea_fixes_standalone.py
- test_needle_fraction_enforcement.py
- test_needle_generation.py
- test_random_distribution.py

## Future Improvements

Potential enhancements enabled by this refactoring:

1. **Configuration management:** Move constants to YAML/JSON config file
2. **CLI interface:** Add argparse for runtime configuration
3. **Progress tracking:** Add progress bars for long-running operations
4. **Parallel evaluation:** Run multiple models concurrently
5. **Result caching:** Cache needle generation and embeddings
6. **Metrics export:** Add Prometheus/Grafana metrics export

## Notes

- Original file preserved as `minea_triple_eval.py.backup`
- All imports tested and verified
- No functional changes to experiment logic
- Code style and formatting preserved
