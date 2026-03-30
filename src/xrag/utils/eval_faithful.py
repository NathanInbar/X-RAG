from xrag.paths import DATASETS_DIR, RESULTS_DIR, CACHE_DIR
import json
import time
import dspy
from aiolimiter import AsyncLimiter
from prettytable import PrettyTable
from xrag.utils import mg_driver, Tokenizer
from xrag.config import config
from xrag.utils.eval import MINER
from xrag.utils.signatures import generate_augmented_response
from xrag.dataset_processing.preprocess import process_dataset_file
import traceback
import numpy as np

EVAL_JUDGE_LM = dspy.LM(config.models["eval_judge"], temperature=0.0)


def clamp(val, lo, hi):
    try:
        return max(lo, min(hi, int(val)))
    except (ValueError, TypeError):
        return lo


class _FaithfulEvalSignature(dspy.Signature):
    """
    You are an expert tasked with evaluating an answer to a question based on four criteria: Comprehensiveness, Diversity, Empowerment and Overall Quality.

    Your task is to evaluate the following answer based on four criteria. For each criterion, assign a score from 1 to 10, following the detailed scoring rubric.

    When explaining your score, you must refer directly to specific parts of the answer to justify your reasoning. Avoid general statements — your explanation must be grounded in the content provided.

    - Comprehensiveness:
    How much detail does the answer provide to cover all aspects and details of the question?

    - Diversity:
    How varied and rich is the answer in providing different perspectives and insights on the question?

    - Empowerment:
    How well does the answer help the reader understand and make informed judgments about the topic?

    - Overall Quality:
    Provide an overall evaluation based on the combined performance across all four dimensions. Consider both content quality and answer usefulness to the question.

    Scoring Guidelines:
    "1-2": "Low score description: Clearly deficient in this aspect, with significant issues."
    "3-4": "Below average score description: Lacking in several important areas, with noticeable problems."
    "5-6": "Average score description: Adequate but not exemplary, meets basic expectations with some minor issues."
    "7-8": "Above average score description: Generally strong but with minor shortcomings."
    "9-10": "High score description: Outstanding in this aspect, with no noticeable issues."
    """

    query: str = dspy.InputField(desc="The question being asked.")
    answer: str = dspy.InputField(desc="The generated answer to evaluate.")

    comprehensiveness: int = dspy.OutputField(desc="Score from 1 to 10.")
    empowerment: int = dspy.OutputField(desc="Score from 1 to 10.")
    diversity: int = dspy.OutputField(desc="Score from 1 to 10.")
    overall_quality: int = dspy.OutputField(desc="Score from 1 to 10.")


pred_faithful_eval = dspy.Predict(_FaithfulEvalSignature)


def faithful_metrics(result):
    comp, div, emp, overall = [], [], [], []
    for part in result:
        for query in part["queries"]:
            comp.append(query["comprehensiveness"])
            div.append(query["diversity"])
            emp.append(query["empowerment"])
            overall.append(query["overall_quality"])

    if not comp:
        return 0.0, 0.0, 0.0, 0.0

    arr = [np.array(v, dtype=float) for v in (comp, div, emp, overall)]
    return tuple(float(a.mean()) for a in arr)


def context_token_stats(result):
    tokens = []
    for part in result:
        for query in part["queries"]:
            tokens.append(query["context_tokens"])
    if not tokens:
        return 0.0, 0.0
    tokens = np.array(tokens, dtype=float)
    return float(tokens.mean()), float(np.median(tokens))


def mean_median_query_time(result):
    times = []
    for part in result:
        for query in part["queries"]:
            times.append(query["duration"])
    if not times:
        return 0.0, 0.0
    mean = sum(times) / len(times)
    times.sort()
    median = times[len(times) // 2]
    return mean, median


async def miner_evaluate_faithful(name: str, miner: "MINER", dataset: str):
    dataset_dir = DATASETS_DIR / dataset
    paths = [p for p in sorted(dataset_dir.iterdir()) if p.is_file() and p.suffix == ".json"]
    result_file = RESULTS_DIR / f"{name}.json"
    tmp_file = result_file.with_suffix(result_file.suffix + ".tmp")

    print(f"Writing results file to '{result_file}'")

    # ---- load existing results (cache) ----
    results_obj = {"name": name, "result": []}
    if result_file.is_file():
        try:
            with open(result_file, "r") as fp:
                loaded = json.load(fp)
            if isinstance(loaded, dict) and isinstance(loaded.get("result"), list):
                results_obj = loaded
                results_obj["name"] = name
            elif isinstance(loaded, list):
                results_obj["result"] = loaded
        except Exception:
            pass

    done_files = {
        r.get("filename")
        for r in results_obj["result"]
        if isinstance(r, dict) and r.get("filename") and ("error" not in r)
    }

    def _atomic_save():
        with open(tmp_file, "w") as fp:
            json.dump(results_obj, fp, indent=2, ensure_ascii=False)
            fp.write("\n")
        tmp_file.replace(result_file)

    _atomic_save()

    # ---- evaluation loop ----
    for i, p in enumerate(paths):

        if p.name in done_files:
            print(f"SKIP (cached): {p.name} ({i+1}/{len(paths)})")
            continue

        try:
            print(f"START EVAL: {p.name} ({i+1}/{len(paths)})")

            with open(p, "r") as fp:
                mine_data = json.load(fp)

            # Preprocess (only if missing)
            preprocessed_chunks = CACHE_DIR / f"{p.stem}__chunks.json"
            preprocessed_descs = CACHE_DIR / f"{p.stem}__g0_descriptions.json"
            if (not preprocessed_chunks.is_file()) or (not preprocessed_descs.is_file()):
                await process_dataset_file(p)

            # Ingest
            print("Ingesting...")
            ingest_st = time.time()
            await miner.ingest(preprocessed_chunks, preprocessed_descs)
            await miner.pre_retrieve(p.name)
            ingest_en = time.time()

            # Query + evaluate
            print("Evaluating...")
            query_results = []

            queries = mine_data.get("queries", [])

            for j, query in enumerate(queries):
                print(f"\rQuery {j+1}/{len(queries)}", end="", flush=True)

                # Retrieve
                q_st = time.time()
                bei, aei, rpi, chunks = await miner.retrieve_parts(query, preprocessed_chunks)
                q_en = time.time()

                # Token count of retrieved context
                full_context = "\n".join([bei, aei, rpi, chunks])
                context_tokens = len(Tokenizer.encode(full_context))

                # Generate response
                generated_answer = await generate_augmented_response(
                    query=query,
                    base_entity_info=bei,
                    agg_entity_info=aei,
                    reasoning_path_info=rpi,
                    relevant_chunk_texts=chunks,
                )

                if generated_answer is None:
                    print(f"\nWARNING: generate_augmented_response returned None for query {j+1}, skipping")
                    continue

                # Score once with temperature=0 (matching LeanRAG repo evaluate_score.py)
                with dspy.context(lm=EVAL_JUDGE_LM):
                    judge_result = await pred_faithful_eval.acall(
                        query=query, answer=generated_answer
                    )

                query_results.append({
                    "query": query,
                    "generated_answer": generated_answer,
                    "context_tokens": context_tokens,
                    "duration": q_en - q_st,
                    "comprehensiveness": clamp(judge_result["comprehensiveness"], 1, 10),
                    "diversity": clamp(judge_result["diversity"], 1, 10),
                    "empowerment": clamp(judge_result["empowerment"], 1, 10),
                    "overall_quality": clamp(judge_result["overall_quality"], 1, 10),
                })
            print("")

            result = {
                "filename": p.name,
                "ingest_duration": ingest_en - ingest_st,
                "queries": query_results,
            }

            await miner.reset()

        except Exception as e:
            tb = e.__traceback__
            last = traceback.extract_tb(tb)[-1]
            result = {"filename": p.name, "error": f"{last.filename}:{last.lineno} | {type(e).__name__}: {e}"}
            print(f"ERROR: {str(e)}")
            try:
                await miner.reset()
            except Exception:
                pass

        # persist + update cache
        results_obj["result"].append(result)
        if "error" not in result:
            done_files.add(p.name)
        _atomic_save()
        print(f"wrote result to {result_file}")


async def evaluate(
    eval_itms: list[callable],
    concurrency: int = 3,
):
    await mg_driver.init()
    limiter = AsyncLimiter(concurrency)

    async def limited(f):
        async with limiter:
            return await f

    print(f"Running {len(eval_itms)} evaluations with concurrency {concurrency}")
    [await f for f in eval_itms]
    print("Done!")

    show_results()


def show_results():
    errors = []
    table = PrettyTable()
    table.field_names = [
        "Name",
        "Comprehensiveness",
        "Diversity",
        "Empowerment",
        "Overall",
        "Context Tokens (mean)",
        "Query Duration (mean)",
        "Query Duration (median)",
    ]

    for f in RESULTS_DIR.iterdir():
        if not f.is_file() or f.suffix.lower() != ".json":
            continue

        print(f"Reading {f} ...")
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as e:
            errors.append({"filename": f.name, "error": f"failed to load results json: {e}"})
            continue

        if isinstance(data, dict):
            name = data.get("name", f.stem)
            results = data.get("result", [])
        elif isinstance(data, list):
            name = f.stem
            results = data
        else:
            errors.append({"filename": f.name, "error": f"unexpected json root type: {type(data)}"})
            continue

        if not isinstance(results, list):
            errors.append({"filename": f.name, "error": "missing/invalid 'result' list"})
            continue

        # detect faithful-eval results by checking for comprehensiveness key
        r_no_err = []
        is_faithful = False
        for r in results:
            if not isinstance(r, dict):
                errors.append({"filename": f.name, "error": f"non-dict result entry: {type(r)}"})
                continue
            if "error" in r:
                errors.append(r)
            else:
                r_no_err.append(r)
                if r.get("queries") and len(r["queries"]) > 0 and "comprehensiveness" in r["queries"][0]:
                    is_faithful = True

        if not is_faithful:
            continue

        mean_comp, mean_div, mean_emp, mean_overall = faithful_metrics(r_no_err)
        mean_tokens, _ = context_token_stats(r_no_err)
        mean_time, median_time = mean_median_query_time(r_no_err)

        table.add_row([
            name,
            f"{mean_comp:.2f}",
            f"{mean_div:.2f}",
            f"{mean_emp:.2f}",
            f"{mean_overall:.2f}",
            f"{mean_tokens:.0f}",
            f"{mean_time:.2f}s",
            f"{median_time:.2f}s",
        ])

    print(table)
    print("\nERRORS:")
    print(errors)
