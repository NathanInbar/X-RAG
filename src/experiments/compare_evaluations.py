"""
Compare results from triplet_model_comparison.py and minea_triple_eval.py

This script loads both result files and displays a unified comparison table
showing both precision (judge metrics) and recall (MINEA metrics) for each model.

Usage:
    cd /home/ubuntu/X-RAG/src
    PYTHONPATH=. python -m experiments.compare_evaluations
"""

import json
from pathlib import Path
from prettytable import PrettyTable

from xrag.paths import SRC

TRIPLET_RESULTS = SRC / "experiments" / "triplet_comparison_results.json"
MINEA_RESULTS = SRC / "experiments" / "minea_results.json"


def load_results():
    """Load both result files."""
    triplet_data = None
    minea_data = None

    if TRIPLET_RESULTS.exists():
        with open(TRIPLET_RESULTS) as f:
            triplet_data = json.load(f)
    else:
        print(f"⚠️  Triplet results not found: {TRIPLET_RESULTS}")

    if MINEA_RESULTS.exists():
        with open(MINEA_RESULTS) as f:
            minea_data = json.load(f)
    else:
        print(f"⚠️  MINEA results not found: {MINEA_RESULTS}")

    return triplet_data, minea_data


def main():
    triplet_data, minea_data = load_results()

    if not triplet_data and not minea_data:
        print("❌ No results found. Run the experiments first:")
        print("  1. python -m experiments.triplet_model_comparison")
        print("  2. python -m experiments.minea_triple_eval")
        return

    # extract model results
    models = set()
    if triplet_data:
        models.update(k for k in triplet_data.keys() if k != "metadata")
    if minea_data:
        models.update(k for k in minea_data.get("results", {}).keys())

    models = sorted(models)

    # build comparison table
    table = PrettyTable()
    table.field_names = [
        "Model",
        "J_Faith",  # Judge Faithfulness (precision proxy)
        "J_Compl",  # Judge Completeness (model's self-assessment)
        "MINEA_any",  # MINEA (objective recall measure)
        "Precision-Recall",  # Interpretation
    ]
    table.float_format = ".2"

    rows = []
    for model in models:
        # get triplet metrics
        j_faith = None
        j_compl = None
        if triplet_data and model in triplet_data:
            agg = triplet_data[model].get("aggregate", {})
            j_faith = agg.get("mean_judge_faithfulness")
            j_compl = agg.get("mean_judge_completeness")

        # get MINEA metrics
        minea_any = None
        if minea_data and model in minea_data.get("results", {}):
            agg = minea_data["results"][model].get("aggregate", {})
            minea_any = agg.get("mean_minea_any")

        # compute approximate precision-recall classification
        classification = "N/A"
        if j_faith is not None and minea_any is not None:
            faith_pct = j_faith * 10  # scale 1-10 to 10-100
            minea_pct = minea_any * 100

            if faith_pct >= 75 and minea_pct >= 75:
                classification = "✅ Excellent"
            elif faith_pct >= 75 and minea_pct < 75:
                classification = "⚠️ Conservative"
            elif faith_pct < 75 and minea_pct >= 75:
                classification = "⚠️ Over-extracts"
            else:
                classification = "❌ Poor"

        rows.append(
            (
                model,
                f"{j_faith:.1f}" if j_faith is not None else "—",
                f"{j_compl:.1f}" if j_compl is not None else "—",
                f"{minea_any * 100:.1f}%" if minea_any is not None else "—",
                classification,
            )
        )

    # sort by combined quality (average of normalized faith and minea)
    def sort_key(row):
        try:
            faith = float(row[1]) / 10.0 if row[1] != "—" else 0
            minea = float(row[3].rstrip("%")) / 100.0 if row[3] != "—" else 0
            return (faith + minea) / 2
        except:
            return 0

    rows.sort(key=sort_key, reverse=True)

    for row in rows:
        table.add_row(row)

    print("\n" + "=" * 100)
    print("COMBINED EVALUATION: PRECISION vs RECALL")
    print("=" * 100)
    print(table)
    print("\nMetric Interpretation:")
    print("  J_Faith (Judge Faithfulness):  Precision proxy (1-10) — Are triples accurate?")
    print("  J_Compl (Judge Completeness):  Model's self-assessment (1-10) — Does model think it's complete?")
    print("  MINEA_any:                     Objective recall (%) — What % of known facts captured?")
    print("\nClassification:")
    print("  ✅ Excellent:      High precision (≥75%) + High recall (≥75%)")
    print("  ⚠️  Conservative:  High precision (≥75%) + Low recall (<75%) — Misses facts")
    print("  ⚠️  Over-extracts: Low precision (<75%) + High recall (≥75%) — Hallucinates")
    print("  ❌ Poor:           Low precision (<75%) + Low recall (<75%)")
    print()

    # detailed breakdown
    print("=" * 100)
    print("DETAILED BREAKDOWN")
    print("=" * 100)

    for model in models:
        print(f"\n{model}")
        print("-" * 50)

        if triplet_data and model in triplet_data:
            agg = triplet_data[model].get("aggregate", {})
            print("  Judge Metrics (from triplet_model_comparison):")
            print(f"    Faithfulness:     {agg.get('mean_judge_faithfulness', 'N/A'):.2f}/10")
            print(f"    Completeness:     {agg.get('mean_judge_completeness', 'N/A'):.2f}/10")
            print(f"    Granularity:      {agg.get('mean_judge_granularity', 'N/A'):.2f}/10")
            print(f"    Well-formedness:  {agg.get('mean_judge_well_formedness', 'N/A'):.2f}/10")
            print(f"    Topical Coverage: {agg.get('mean_topical_coverage', 'N/A'):.2%}")
            print(f"    Keyword Overlap:  {agg.get('mean_keyword_overlap', 'N/A'):.2%}")
        else:
            print("  Judge Metrics: Not available")

        if minea_data and model in minea_data.get("results", {}):
            agg = minea_data["results"][model].get("aggregate", {})
            print("  MINEA Metrics (from minea_triple_eval):")
            print(f"    Exact match:      {agg.get('mean_minea_exact', 'N/A'):.2%}")
            print(f"    Keyword match:    {agg.get('mean_minea_keyword', 'N/A'):.2%}")
            print(f"    Semantic match:   {agg.get('mean_minea_semantic', 'N/A'):.2%}")
            print(f"    Judge match:      {agg.get('mean_minea_judge', 'N/A'):.2%}")
            print(f"    ANY (overall):    {agg.get('mean_minea_any', 'N/A'):.2%}")
        else:
            print("  MINEA Metrics: Not available")

    print("\n" + "=" * 100)
    print("RECOMMENDATIONS FOR YOUR PAPER")
    print("=" * 100)

    # find best model by combined metric
    best_model = None
    best_score = 0
    for model in models:
        faith = None
        minea = None

        if triplet_data and model in triplet_data:
            faith = triplet_data[model].get("aggregate", {}).get("mean_judge_faithfulness")
        if minea_data and model in minea_data.get("results", {}):
            minea = minea_data["results"][model].get("aggregate", {}).get("mean_minea_any")

        if faith is not None and minea is not None:
            combined = (faith / 10.0 + minea) / 2
            if combined > best_score:
                best_score = combined
                best_model = model

    if best_model:
        print(f"\n🏆 Best Overall Model: {best_model}")
        print(f"   Combined Score: {best_score:.2%}")
        print("\n   Recommended for GraphRAG triple extraction in academic paper corpus.")
    else:
        print("\n⚠️  Unable to determine best model (incomplete data)")

    print("\n✅ Analysis complete!")
    print("\nNext steps:")
    print("  1. Review detailed breakdown above")
    print("  2. Check per-document results in JSON files for failure patterns")
    print("  3. Consider running with larger SAMPLE_SIZE for statistical significance")
    print()


if __name__ == "__main__":
    main()
