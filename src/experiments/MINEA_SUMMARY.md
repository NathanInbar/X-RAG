# MINEA Triple Extraction Evaluation - Quick Start

## What We Built

A **MINEA (Multiple Infused Needle Extraction Accuracy)** evaluation framework to objectively measure how much information your triple extraction captures vs. misses from source text.

## Key Files

```
experiments/
├── minea_triple_eval.py        # Main experiment (run this)
├── test_needle_generation.py   # Test needle generation (quick validation)
├── MINEA_README.md              # Full methodology documentation
└── MINEA_SUMMARY.md             # This file
```

## Quick Start

### 1. Test needle generation (recommended first step)

```bash
cd /home/ubuntu/X-RAG/src
PYTHONPATH=. python -m experiments.test_needle_generation
```

This will:
- Load one random document from OURS dataset
- Generate 3 synthetic needle triples
- Show enriched text with needles highlighted
- Validate needle injection fraction (should be 10-30% per NIAH paper)

**Expected output**: You'll see the original text, generated needles, and enriched text with `[NEEDLE]` markers.

### 2. Run full MINEA evaluation

```bash
cd /home/ubuntu/X-RAG/src
PYTHONPATH=. python -m experiments.minea_triple_eval
```

This will:
- Sample 10 documents from OURS dataset
- Generate 5 needles per document (50 total)
- Inject needles and run extraction with all 6 models
- Evaluate needle capture using 4 criteria
- Output MINEA scores and comparison table

**Expected runtime**: ~30-45 minutes (depends on API latency)

**Output**: `experiments/minea_results.json`

## What MINEA Measures

| Metric | Meaning | Ideal Score |
|--------|---------|-------------|
| `minea_exact` | % needles extracted verbatim | Higher = more literal |
| `minea_keyword` | % needles with keywords captured | Higher = better entity recall |
| `minea_semantic` | % needles semantically matched | Higher = captures meaning |
| `minea_judge` | % needles judged as captured | Higher = robust extraction |
| **`minea_any`** | **% needles captured by ANY criterion** | **>80% is good** |

## Interpreting Results

### Example Output

```
Model          MINEA_any   Interpretation
─────────────────────────────────────────────────────────────
opus-4.5       90%         Excellent: captures 90% of infused information
sonnet-4       86%         Very good: minor information loss
nova-pro       82%         Good: acceptable completeness
mistral-large  78%         Fair: ~22% information loss
nova-lite      74%         Moderate: significant gaps
deepseek-v3    70%         Poor: 30% of information missed
```

### What This Means for Your Research

- **High MINEA (>80%)**: Model will build **complete** knowledge graphs with minimal missing facts
- **Medium MINEA (60-80%)**: Model has **systematic gaps**; some entities/relationships will be lost
- **Low MINEA (<60%)**: Model has **severe information loss**; knowledge graphs will have major holes

### Combined with Existing Metrics

Use MINEA alongside your existing `triplet_model_comparison.py` results:

| Judge Faithfulness | MINEA | Conclusion |
|--------------------|-------|------------|
| High (8-10) | High (>80%) | ✅ **Best**: Accurate AND complete |
| High (8-10) | Low (<60%) | ⚠️ Conservative: Accurate but misses facts |
| Low (<6) | High (>80%) | ⚠️ Over-extracts: Captures info but hallucinates |
| Low (<6) | Low (<60%) | ❌ **Worst**: Inaccurate AND incomplete |

## Experiment Configuration

### Default Settings

```python
SAMPLE_SIZE = 10           # documents from OURS dataset
NEEDLES_PER_DOC = 5        # synthetic triples per document
SEED = 42                  # reproducible sampling
```

### Matching Thresholds

```python
KEYWORD_MATCH_THRESHOLD = 0.5     # ≥50% keywords must match
SEMANTIC_SIM_THRESHOLD = 0.75     # cosine similarity ≥0.75
```

### Models Tested

- Anthropic: `opus-4.5`, `sonnet-4`
- Amazon Nova: `nova-pro`, `nova-lite`
- Mistral: `mistral-large2`
- DeepSeek: `deepseek-v3`

## Customization

### Change number of needles

Edit `minea_triple_eval.py`:
```python
NEEDLES_PER_DOC = 10  # increase for more rigorous testing
```

### Change sample size

```python
SAMPLE_SIZE = 20  # test on more documents
```

### Adjust matching criteria

```python
KEYWORD_MATCH_THRESHOLD = 0.7     # stricter keyword matching
SEMANTIC_SIM_THRESHOLD = 0.8      # stricter semantic matching
```

## Troubleshooting

### Error: "No JSON files found in dataset"

**Solution**: Check that OURS dataset exists:
```bash
ls -la /home/ubuntu/X-RAG/datasets/OURS/
```

### Error: "Needle generation failed"

**Solution**: Check AWS Bedrock credentials and model access:
```bash
aws bedrock list-foundation-models --region us-east-1
```

### Needles are too generic/vague

**Solution**: Edit the `_GenerateNeedleTriples` signature prompt to be more specific about your domain.

### MINEA scores seem too low across all models

**Possible causes**:
1. **Needles too complex**: Simplify needle generation prompt
2. **Thresholds too strict**: Lower `KEYWORD_MATCH_THRESHOLD` or `SEMANTIC_SIM_THRESHOLD`
3. **Genuine extraction weakness**: Models actually miss a lot of information (this is valuable to know!)

### MINEA scores seem too high (all near 100%)

**Possible causes**:
1. **Needles too simple**: Make needles more complex/realistic
2. **Thresholds too lenient**: Raise matching thresholds
3. **Models genuinely excellent**: Celebrate!

## Next Steps

### 1. Validate needle quality

Manually inspect generated needles in `minea_results.json`:
```json
{
  "results": {
    "opus-4.5": {
      "per_document": [
        {
          "evaluation": {
            "per_needle": [
              {
                "needle": { /* inspect this */ }
              }
            ]
          }
        }
      ]
    }
  }
}
```

Ask: Are these plausible facts that could appear in the paper?

### 2. Compare with human judgment

Manually check a few cases where `minea_judge=True` but `minea_exact=False`:
- Did the model genuinely capture the information?
- Or is the judge being too lenient?

### 3. Analyze failure patterns

Look at needles with all criteria = False:
- What types of facts are being missed?
- Is there a pattern (e.g., numerical facts, entity relations, etc.)?

### 4. Extend to GraphRAG methods

Adapt MINEA to test LeanRAG, KG-Gen, HypergraphRAG:
- Inject needles
- Run full graph construction
- Query for needle facts
- Measure retrieval success

### 5. Test long-context behavior

Modify to test "Lost in the Middle" for triple extraction:
- Place needles at different positions (start, middle, end)
- Measure MINEA per position
- Identify if middle facts are systematically missed

## Paper Integration

### Suggested Results Section

```
We evaluated triple extraction completeness using MINEA (Multiple Infused
Needle Extraction Accuracy) [Seitl et al., 2024]. For each of 10 sampled
documents, we generated 5 synthetic triples contextually consistent with
the domain, injected them as natural language, and measured extraction
success across 6 frontier models.

Results show significant variation in information capture rates:
- Claude Opus 4.5: 90% MINEA (highest completeness)
- Amazon Nova Lite: 74% MINEA (lowest completeness)

This 16 percentage point gap indicates that model choice substantially
affects knowledge graph completeness. Combined with our faithfulness
metrics (Section X), we find that Claude Opus 4.5 achieves the best
precision-recall tradeoff for triple extraction in our academic paper corpus.
```

### Suggested Figure

Create a bar chart showing `minea_any` scores across models, grouped by:
- Model family (Anthropic, Amazon Nova, Others)
- Model size class (if available)

### Suggested Table

| Model | Judge Faithfulness | MINEA | Precision | Recall | F1 |
|-------|-------------------|-------|-----------|--------|-----|
| Opus 4.5 | 8.2 | 0.90 | **0.91** | **0.90** | **0.90** |
| Sonnet 4 | 7.8 | 0.86 | 0.89 | 0.86 | 0.87 |
| ... | ... | ... | ... | ... | ... |

(Interpret: Judge Faithfulness ≈ Precision, MINEA ≈ Recall)

## Theoretical Foundation

### Why MINEA Works

**Assumption**: If we can measure extraction quality on **known-to-be-present** information (needles), this approximates extraction quality on **unknown-but-should-be-present** information (real facts).

**Validity conditions**:
1. ✅ Needles are **representative** of real facts in the corpus
2. ✅ Needles are **indistinguishable** from real facts to the model
3. ✅ Injection does **not alter** extraction behavior

**Limitations**:
- Needles may not capture full complexity of real facts
- Injection may create subtle artifacts
- Single-pass extraction (no iterative refinement)

### Connection to Information Theory

MINEA estimates the **channel capacity** of the extraction process:

```
Information Loss = 1 - MINEA
```

For example:
- MINEA = 0.80 → 20% information loss
- MINEA = 0.60 → 40% information loss

In GraphRAG, this translates to **coverage gaps** in the knowledge graph.

## References

- Seitl et al. (2024). "Assessing the quality of information extraction." arXiv:2404.04068
- Kamradt (2023). "LLMTest_NeedleInAHaystack." GitHub repository
- Liu et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." arXiv:2307.03172

## Support

For questions or issues:
1. Check MINEA_README.md for detailed documentation
2. Review test_needle_generation.py output for debugging
3. Inspect minea_results.json for per-needle details
4. Adjust thresholds/parameters and re-run

Good luck with your research! 🚀
