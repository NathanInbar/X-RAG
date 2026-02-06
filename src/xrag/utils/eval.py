from xrag.paths import DATASETS_DIR, RESULTS_DIR, CACHE_DIR

VERBOSE = 0
JUST_ONE = 1

import json
import time
import dspy
from aiolimiter import AsyncLimiter
from prettytable import PrettyTable
from xrag.utils import mg_driver
from xrag.config import config
from xrag.dataset_processing.preprocess import process_dataset_file

class MINER(object):
    async def ingest(self, preprocess_results_filename: str):
        """ Ingest knowledge from some text. """
        pass
    async def pre_retrieve(self):
        pass
    async def retrieve(self, text: str) -> str:
        """ Find information relevant to a text. """
        pass
    async def reset(self):
        """ Forget ingested knowledge. """
        pass

class _EvalSignature(dspy.Signature):
    """ Does the context contain the information stated in the statement?. """
    context: str = dspy.InputField()
    statement: str = dspy.InputField()
    context_contains_statement: bool = dspy.OutputField()

dspy_evaluate = dspy.Predict(_EvalSignature)

def score_count(result):
    """ Computes total score and count. """
    score = 0
    count = 0
    for part in result:
        for query in part["queries"]:
            count += 1
            score += int(query["contained"])
    return score, count

class TrimSignature(dspy.Signature):
    """Trim the context to only include information necessary to answer the query.
    Remove irrelevant sections while preserving all relevant content exactly as written."""
    
    actual_context: str = dspy.InputField(
        desc = "The full context that may contain both relevant and irrelevant information"
    )
    statement: str = dspy.InputField(
        desc = "The statement that needs to be derived from the context."
    )
    optimal_context: str = dspy.OutputField(
        desc = (
            "The trimmed context containing only relevant information needed to derive the statement."
            "Use the exact wording from the actual_context."
            "Do not paraphrase, summarize, or rewrite. Remove only the irrelevant sections."
            "Output plain text with no formatting (bold, italics, or markdown)."
        )
    )

trim = dspy.Predict(TrimSignature)

def trim_to_optimal(context, statement):
    with dspy.context(lm=config.models["eval_judge"]):
        result = trim(
            actual_context=context,
            statement=statement
        )

    return result.optimal_context

def conciseness(result):
    """ 
    For each "correct" response:
    - Ask LLM to trim to optimal context.
    - Compute len(optimal)/len(actual)
    """
    optimal_length = 0
    actual_length = 0
    print("")
    for doc in result:
        if VERBOSE: print(doc['filename'])
        for query in doc["queries"]:
            if query["contained"]:
                actual_context = query["context"]
                statement = query["query"]
                optimal_context = trim_to_optimal(actual_context, statement)
                actual_length += len(actual_context)
                optimal_length += len(optimal_context)
                if VERBOSE: print("-" * 100)
                if VERBOSE: print(f"statement: {statement}\n")
                if VERBOSE: print(f"actual context ({actual_length}): {actual_context}\n")
                if VERBOSE: print(f"optimal context ({optimal_length}): {optimal_context}\n")
        
        if VERBOSE: print(f"\rDone {doc["filename"]}")
    if VERBOSE: print(f"total optimal_length: {optimal_length}")
    if VERBOSE: print(f"total actual_length: {actual_length}")
    if (actual_length == 0):
        raise ZeroDivisionError()
    concision = optimal_length / actual_length
    if VERBOSE: print(f"MEAN CONCISION: {concision}")
    return concision

def mean_median_query_time(result):
    times = []
    for part in result:
        for query in part["queries"]:
            times.append(query["duration"])
    mean = sum(times) / len(times)
    times.sort()
    median = times[len(times)//2]
    return mean, median

async def miner_evaluate_individual_with_preprocess(name: str, miner: "MINER", dataset: str):
    dataset_dir = DATASETS_DIR / dataset
    paths = list(dataset_dir.iterdir())
    result_file = RESULTS_DIR / f"{name}.json"
    tmp_file = result_file.with_suffix(result_file.suffix + ".tmp")

    print(f"Writing results file to '{result_file}'")

    # ---- load existing results (cache) ----
    results_obj = {"name": name, "result": []}
    if result_file.is_file():
        try:
            with open(result_file, "r") as fp:
                loaded = json.load(fp)
            # accept either full object {"name":..., "result":[...]} or a bare list (older attempts)
            if isinstance(loaded, dict) and isinstance(loaded.get("result"), list):
                results_obj = loaded
                results_obj["name"] = name  # keep current run name
            elif isinstance(loaded, list):
                results_obj["result"] = loaded
        except Exception:
            pass

    # consider a file "done" if we already have an entry for it that is not an error
    done_files = {
        r.get("filename")
        for r in results_obj["result"]
        if isinstance(r, dict) and r.get("filename") and ("error" not in r)
    }

    def _atomic_save():
        # atomic-ish save (write tmp then replace)
        with open(tmp_file, "w") as fp:
            json.dump(results_obj, fp, indent=2, ensure_ascii=False)
            fp.write("\n")
        tmp_file.replace(result_file)

    _atomic_save()

    # ---- evaluation loop ----
    for i, p in enumerate(paths):

        # caching skip
        if p.name in done_files:
            print(f"SKIP (cached): {p.name} ({i+1}/{len(paths)})")
            continue

        try:
            print(f"START EVAL: {p.name} ({i+1}/{len(paths)})")

            # Load data
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
            queries = []
            with dspy.context(lm=config.models["eval_judge"]):
                answers = mine_data.get("answers", [])
                for j, a in enumerate(answers):
                    print(f"\rQuery {j+1}/{len(answers)}", end="")
                    q_st = time.time()
                    info = await miner.retrieve(a, preprocessed_chunks)
                    q_en = time.time()
                    contained = (await eval.acall(context=info, statement=a)).context_contains_statement
                    queries.append(
                        {
                            "query": a,
                            "context": info,
                            "contained": contained,
                            "duration": q_en - q_st,
                        }
                    )
                print("")

            result = {
                "filename": p.name,
                "ingest_duration": ingest_en - ingest_st,
                "queries": queries,
            }

            await miner.reset()

        except Exception as e:
            result = {"filename": p.name, "error": str(e)}
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
        "Score",
        "Necessary Context",
        "Score with Conciseness",
        # "Conciseness",
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

        # tolerate either {"name":..., "result":[...]} or a bare list (older files)
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

        r_no_err = []
        for r in results:
            if not isinstance(r, dict):
                errors.append({"filename": f.name, "error": f"non-dict result entry: {type(r)}"})
                continue
            if "error" in r:
                errors.append(r)
            else:
                r_no_err.append(r)
            if JUST_ONE: break

        score, count = score_count(r_no_err)
        r_conciseness = conciseness(r_no_err)
        mean, median = mean_median_query_time(r_no_err)

        # avoid division by zero
        pct = (score / count * 100.0) if count else 0.0
        concise = r_conciseness if (r_conciseness and r_conciseness > 0) else 0.0
        # efficiency = (pct / concise) if concise else 0.0
        alpha = 0.8
        beta = 1 - alpha
        overall = alpha * pct + beta * r_conciseness

        table.add_row([
            name,
            f"{pct:.2f}% ({score}/{count})" if count else "n/a (0/0)",
            f"{concise:.2f}%" if concise else "n/a",
            # f"{efficiency:.2f}" if efficiency else "n/a",
            f"{overall:.2f}%" if overall else "n/a",
            f"{mean:.2f}s" if mean is not None else "n/a",
            f"{median:.2f}s" if median is not None else "n/a",
        ])

    print(table)
    print("\nERRORS:")
    print(errors)
