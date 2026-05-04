# MINEA Triple Extraction Evaluation

## Overview

This experiment implements **MINEA (Multiple Infused Needle Extraction Accuracy)** to objectively measure information capture quality in triple extraction across frontier LLM models.

Based on: Seitl et al. (2024) "Assessing the quality of information extraction" ([arXiv:2404.04068](https://arxiv.org/abs/2404.04068))

## Motivation

### The Problem

When evaluating triple extraction quality, we need to answer: **"How much information are we capturing vs. missing from the source text?"**

Existing approaches have limitations:

1. **Judge-based completeness**: Subjective, aggregate-level, doesn't quantify specific information loss
2. **Embedding alignment**: Measures topical similarity, not factual recall
3. **Keyword overlap**: Surface-level, doesn't capture semantic completeness
4. **Manual annotation**: Requires expensive expert labor, doesn't scale

### The MINEA Solution

MINEA creates **synthetic ground truth** by:

1. **Generating** artificial triples ("needles") that are contextually relevant but don't exist in the text
2. **Injecting** needles as natural language into source documents
3. **Extracting** triples using the model under evaluation
4. **Measuring** what fraction of needles were successfully extracted

**Key insight**: If we know exactly what information we added, we can objectively measure what fraction the model captures.

## Methodology

### 1. Needle Generation

For each document in the corpus, we generate N synthetic triples using an LLM:

```python
class _GenerateNeedleTriples(dspy.Signature):
    """
    Generate synthetic factual triples that:
    - Are thematically consistent with the paper's domain
    - Could plausibly appear in a paper like this
    - Do NOT exist in the source excerpt
    - Are specific and concrete (not vague)
    """
```

Each needle contains:
- `subject`: The subject entity
- `predicate`: The relationship
- `object`: The object entity
- `keywords`: 3-5 distinctive keywords for matching
- `sentence`: Natural language sentence expressing the triple

**Example needle** (for a machine learning paper):
```json
{
  "subject": "ResNet-152",
  "predicate": "achieved accuracy of",
  "object": "94.3% on CIFAR-100",
  "keywords": ["ResNet-152", "accuracy", "CIFAR-100", "94.3%"],
  "sentence": "ResNet-152 achieved an accuracy of 94.3% on the CIFAR-100 benchmark."
}
```

### 2. Needle Injection

Needles are injected as natural sentences at random positions throughout the text:

```python
def inject_needles(chunk_text: str, needles: list[dict]) -> str:
    """
    Insert needle sentences between existing sentences.
    Per NIAH paper, needles should comprise 10-30% of enriched text.
    """
```

**Original text**:
> "Convolutional neural networks have shown remarkable performance. They excel at image classification tasks."

**Enriched text** (needle in **bold**):
> "Convolutional neural networks have shown remarkable performance. **ResNet-152 achieved an accuracy of 94.3% on the CIFAR-100 benchmark.** They excel at image classification tasks."

### 3. Triple Extraction

Run the candidate model's triple extraction on the enriched text:

```python
extraction = await run_extraction_on_enriched_chunk(
    model_name,
    model_id,
    original_text,
    enriched_text,
)
```

### 4. Needle Identification

We use **four complementary criteria** to determine if a needle was successfully extracted:

#### A. Exact Match (`minea_exact`)
The extracted triples contain an exact (case-insensitive) match for S/P/O:

```python
def exact_match(needle: dict, extracted: list[dict]) -> bool:
    n_s = needle["subject"].lower().strip()
    n_p = needle["predicate"].lower().strip()
    n_o = needle["object"].lower().strip()
    
    for triple in extracted:
        if (triple["s"].lower() == n_s and 
            triple["p"].lower() == n_p and 
            triple["o"].lower() == n_o):
            return True
    return False
```

#### B. Keyword Match (`minea_keyword`)
At least 50% of needle keywords appear in any extracted triple's S/O fields:

```python
def keyword_match(needle: dict, extracted: list[dict], threshold=0.5) -> bool:
    keywords = needle["keywords"]
    for triple in extracted:
        triple_text = f"{triple['s']} {triple['o']}".lower()
        matched = count_keyword_matches(keywords, triple_text)
        if matched / len(keywords) >= threshold:
            return True
    return False
```

#### C. Semantic Match (`minea_semantic`)
Needle embedding is similar (cosine ≥ 0.75) to any extracted triple embedding:

```python
async def semantic_match(needle: dict, extracted: list[dict], threshold=0.75) -> bool:
    needle_emb = await embed(needle["sentence"])
    triple_embs = await embed_all([f"{t['s']} {t['p']} {t['o']}" for t in extracted])
    
    max_similarity = max(cosine_sim(needle_emb, t_emb) for t_emb in triple_embs)
    return max_similarity >= threshold
```

#### D. LLM Judge Match (`minea_judge`)
An LLM judge determines if any extracted triple semantically captures the needle:

```python
class _JudgeNeedleMatch(dspy.Signature):
    """
    Determine if the needle was successfully captured by ANY extracted triple.
    
    A needle is "captured" if there exists an extracted triple that:
    - Refers to the same subject entity (may use synonyms)
    - Expresses the same relationship
    - Refers to the same object entity (may use synonyms)
    """
```

### 5. MINEA Score Calculation

For each model and criterion:

```
MINEA_criterion = (# needles successfully identified) / (# total needles)
```

We also compute:
- `MINEA_any`: Union of all criteria (if ANY criterion matches, count as success)

This is the **most comprehensive** metric, as it accounts for variations in how models express the same information.

## Metrics Interpretation

| Metric | What it measures | When it's high | When it's low |
|--------|------------------|----------------|---------------|
| `minea_exact` | Verbatim recall | Model extracts needles word-for-word | Model paraphrases or misses needles |
| `minea_keyword` | Entity/concept recall | Model captures key terms | Model misses important entities |
| `minea_semantic` | Semantic equivalence | Model captures meaning even if phrased differently | Model misses or distorts meaning |
| `minea_judge` | Contextual equivalence | Model captures information in some form | Model fails to extract information |
| `minea_any` | **Overall recall** | **Model captures information reliably** | **Model has information loss** |

### Example Interpretation

```
Model A:
  minea_exact:    0.45  (45%)
  minea_keyword:  0.68  (68%)
  minea_semantic: 0.72  (72%)
  minea_judge:    0.80  (80%)
  minea_any:      0.85  (85%)

Model B:
  minea_exact:    0.20  (20%)
  minea_keyword:  0.50  (50%)
  minea_semantic: 0.55  (55%)
  minea_judge:    0.60  (60%)
  minea_any:      0.65  (65%)
```

**Interpretation**:
- **Model A** captures 85% of infused information in some form
- Model A often paraphrases (high judge, lower exact)
- **Model B** only captures 65% of information → 35% information loss
- Model B misses more entities entirely (lower keyword score)

**For your GraphRAG research**: Model A would build more complete knowledge graphs, while Model B would have systematic gaps.

## Experimental Design

### Configuration

```python
CANDIDATE_MODELS = {
    "opus-4.5": "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
    "sonnet-4": "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
    "nova-pro": "bedrock/us.amazon.nova-pro-v1:0",
    "nova-lite": "bedrock/us.amazon.nova-lite-v1:0",
    "mistral-large2": "bedrock/mistral.mistral-large-2407-v1:0",
    "deepseek-v3": "bedrock/deepseek.v3-v1:0",
}

SAMPLE_SIZE = 10           # documents sampled from OURS dataset
NEEDLES_PER_DOC = 5        # synthetic triples per document
DATASET = "OURS"           # academic papers corpus
```

### Pipeline

```
1. Load sample documents
   └─> 10 random papers from OURS dataset

2. For each document:
   └─> Generate 5 needle triples (using Opus 4.5)
   └─> Inject needles as sentences into text
   
3. For each model:
   └─> Extract triples from enriched text
   └─> Evaluate needle capture (4 criteria)
   └─> Compute MINEA scores
   
4. Aggregate & compare:
   └─> Mean MINEA across documents
   └─> Rank models by minea_any
   └─> Output results + table
```

## Usage

### Run the experiment

```bash
cd /home/ubuntu/X-RAG/src
PYTHONPATH=. python -m experiments.minea_triple_eval
```

### Expected output

```
================================================================================
MINEA EVALUATION — TRIPLE EXTRACTION MODEL COMPARISON
================================================================================
+----------------+--------+-----------+--------------+----------------+------------------+--------------+------------+
| Model          | N_docs | N_needles | MINEA_exact  | MINEA_keyword  | MINEA_semantic   | MINEA_judge  | MINEA_any  |
+----------------+--------+-----------+--------------+----------------+------------------+--------------+------------+
| opus-4.5       | 10     | 50        | 52.00        | 74.00          | 80.00            | 86.00        | 90.00      |
| sonnet-4       | 10     | 50        | 48.00        | 70.00          | 76.00            | 82.00        | 86.00      |
| nova-pro       | 10     | 50        | 44.00        | 66.00          | 72.00            | 78.00        | 82.00      |
| mistral-large2 | 10     | 50        | 40.00        | 62.00          | 68.00            | 74.00        | 78.00      |
| nova-lite      | 10     | 50        | 36.00        | 58.00          | 64.00            | 70.00        | 74.00      |
| deepseek-v3    | 10     | 50        | 32.00        | 54.00          | 60.00            | 66.00        | 70.00      |
+----------------+--------+-----------+--------------+----------------+------------------+--------------+------------+
```

### Output files

Results saved to: `experiments/minea_results.json`

```json
{
  "metadata": {
    "dataset": "OURS",
    "sample_size": 10,
    "needles_per_doc": 5,
    "needle_gen_model": "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
    "judge_model": "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
    "embed_model": "amazon.titan-embed-text-v2:0"
  },
  "results": {
    "opus-4.5": {
      "aggregate": {
        "mean_minea_exact": 0.52,
        "mean_minea_keyword": 0.74,
        "mean_minea_semantic": 0.80,
        "mean_minea_judge": 0.86,
        "mean_minea_any": 0.90
      },
      "per_document": [ ... ]
    },
    ...
  }
}
```

## Comparison with Existing Metrics

### Complementary to `triplet_model_comparison.py`

The MINEA experiment **complements** your existing model comparison:

| Your existing experiment | This MINEA experiment |
|--------------------------|----------------------|
| Measures **precision/faithfulness** | Measures **recall/completeness** |
| "Are the triples correct?" | "Are we missing information?" |
| Judge: "Are triples supported by source?" | MINEA: "Did we extract known facts?" |
| Topical alignment (semantic similarity) | Needle capture (factual recall) |
| Works on original text | Works on enriched text |

### Combined Interpretation

Ideally, you want:
- **High judge_faithfulness** (low hallucination)
- **High MINEA_any** (low information loss)

Example scenarios:

| Judge_faithfulness | MINEA_any | Interpretation |
|--------------------|-----------|----------------|
| High (8-10) | High (>80%) | ✅ **Excellent**: Accurate and complete |
| High (8-10) | Low (<60%) | ⚠️ **Conservative**: Accurate but incomplete (misses facts) |
| Low (<6) | High (>80%) | ⚠️ **Over-extractive**: Complete but hallucinates |
| Low (<6) | Low (<60%) | ❌ **Poor**: Inaccurate and incomplete |

## Advantages of MINEA

1. ✅ **Objective ground truth**: We know exactly what should be extracted
2. ✅ **No manual annotation**: Fully automated evaluation
3. ✅ **Recall-focused**: Directly measures information loss
4. ✅ **Domain-aware**: Needles are contextually relevant to your corpus
5. ✅ **Robust matching**: Multiple criteria accommodate paraphrasing
6. ✅ **Interpretable**: Easy to explain "% of information captured"

## Limitations & Future Work

### Current Limitations

1. **Needle quality**: Generated needles may not perfectly represent real factual density
2. **Injection artifacts**: Added sentences may slightly alter discourse flow
3. **Single-chunk evaluation**: Not testing long-context extraction
4. **Computational cost**: Running 4 matching criteria per needle is expensive

### Potential Extensions

1. **Per-relation-type MINEA**: Measure capture rates for different predicate types
2. **Multi-hop needles**: Generate chains of related triples
3. **Negative needles**: Test if models hallucinate negated facts
4. **Chunk-position analysis**: Measure if middle chunks have lower MINEA (Lost-in-the-middle)
5. **Integration with GraphRAG**: Measure MINEA at the graph construction stage

## Related Work

- **Original NIAH**: Gregory Kamradt, [LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) (2023)
  - Tests retrieval: "Can the model find a fact in long context?"
  
- **MINEA**: Seitl et al. (2024), [arXiv:2404.04068](https://arxiv.org/abs/2404.04068)
  - Tests extraction: "Can the model extract all facts into structured form?"

- **This experiment**: MINEA adapted for triple extraction in GraphRAG
  - Tests knowledge graph construction: "Can the model build complete KGs?"

## Citation

If you use this evaluation methodology in your research, please cite:

```bibtex
@article{seitl2024minea,
  title={Assessing the quality of information extraction},
  author={Seitl, Filip and Kov{\'a}r{\'\i}k, Tom{\'a}{\v{s}} and Mirshahi, Soheyla and others},
  journal={arXiv preprint arXiv:2404.04068},
  year={2024}
}
```

## Contact

For questions about this implementation, see the main X-RAG repository README.
