# MINEA Quick Reference Card

## 🎯 Purpose
Measure **how much information** your triple extraction captures vs. misses from source text.

## 🚀 Quick Commands

```bash
# 1. Test needle generation (5 minutes)
cd /home/ubuntu/X-RAG/src
PYTHONPATH=. python -m experiments.test_needle_generation

# 2. Run full MINEA evaluation (~30-45 minutes)
PYTHONPATH=. python -m experiments.minea_triple_eval

# 3. Compare with existing triplet evaluation
PYTHONPATH=. python -m experiments.compare_evaluations
```

## 📊 Key Metrics

| Metric | What it measures | Good value |
|--------|------------------|------------|
| `minea_exact` | Verbatim extraction | 40-60% |
| `minea_keyword` | Entity recall | 60-80% |
| `minea_semantic` | Meaning capture | 70-85% |
| `minea_judge` | Contextual match | 75-90% |
| **`minea_any`** | **Overall recall** | **>80%** ✅ |

## 🎓 Interpretation

```
MINEA_any > 80%   →  Excellent (minimal information loss)
MINEA_any 60-80%  →  Good (acceptable gaps)
MINEA_any < 60%   →  Poor (significant information missing)
```

## 🔬 How It Works

1. **Generate** synthetic triples ("needles") relevant to your corpus
2. **Inject** needles as natural sentences into source text
3. **Extract** triples using model under test
4. **Measure** how many needles were successfully extracted
5. **Score** = (captured needles) / (total needles)

## 🆚 vs Existing Metrics

| Your existing metrics | MINEA |
|----------------------|-------|
| Measures **precision** | Measures **recall** |
| "Are triples correct?" | "Are we missing information?" |
| Judge faithfulness | Objective ground truth |

## ⚙️ Configuration

```python
# experiments/minea_triple_eval.py

SAMPLE_SIZE = 10           # Increase for more documents
NEEDLES_PER_DOC = 5        # Increase for more rigorous test
KEYWORD_MATCH_THRESHOLD = 0.5    # Lower = more lenient
SEMANTIC_SIM_THRESHOLD = 0.75    # Lower = more lenient
```

## 📂 Output Files

- `experiments/minea_results.json` - Full results with per-needle details
- Console output - Summary table ranking models

## 🔍 Debugging

```bash
# Check if needles are being generated correctly
PYTHONPATH=. python -m experiments.test_needle_generation

# If needle generation fails, check AWS credentials
aws bedrock list-foundation-models --region us-east-1

# If extraction fails, check logs for specific error
# (script shows detailed attempt-by-attempt logging)
```

## 📖 Full Documentation

- **MINEA_README.md** - Complete methodology and theory
- **MINEA_SUMMARY.md** - Setup guide and troubleshooting
- **compare_evaluations.py** - Side-by-side precision vs recall analysis

## 📝 For Your Paper

**Suggested text:**
> We evaluated information capture completeness using MINEA [Seitl et al., 2024],
> which measures extraction recall by injecting synthetic facts and measuring
> recovery rate. Results show Model X achieves 90% MINEA, indicating excellent
> information capture, while Model Y achieves only 70%, indicating significant
> information loss that would result in incomplete knowledge graphs.

## 🎯 Expected Results

Based on typical LLM performance:

```
Rank  Model          MINEA_any   Interpretation
────────────────────────────────────────────────
1     opus-4.5       85-92%      Best completeness
2     sonnet-4       80-88%      Excellent
3     nova-pro       75-85%      Very good
4     mistral-large  70-80%      Good
5     nova-lite      65-75%      Moderate gaps
6     deepseek-v3    60-70%      Significant loss
```

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| All MINEA scores near 0% | Check if extraction is working at all |
| All MINEA scores near 100% | Needles may be too simple |
| High variance across docs | Normal - some papers harder than others |
| Semantic match fails | Check Titan v2 embedding access |
| Judge match fails | Check Opus 4.5 API access |

## 🔗 Citation

```bibtex
@article{seitl2024minea,
  title={Assessing the quality of information extraction},
  author={Seitl, Filip and others},
  journal={arXiv preprint arXiv:2404.04068},
  year={2024}
}
```

## 💡 Pro Tips

1. **Start small**: Test with 3 documents before full 10
2. **Validate needles**: Manually check a few generated needles first
3. **Compare methods**: Run same evaluation on LeanRAG/KG-Gen/HypergraphRAG
4. **Track over time**: Re-run after model updates to track improvements
5. **Vary domains**: Test on different corpora to see domain generalization

---

**Questions?** See MINEA_README.md for full documentation.
