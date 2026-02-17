from xrag.paths import DATASETS_DIR, RESULTS_DIR, CACHE_DIR
import json
import time
import dspy
from aiolimiter import AsyncLimiter
from prettytable import PrettyTable
from xrag.utils import mg_driver, batched
from xrag.config import config
from xrag.utils.eval import MINER
from xrag.utils import Tokenizer
from xrag.dataset_processing.preprocess import process_dataset_file
import litellm
import traceback
from statistics import mean
import numpy as np

EMBED_MODEL = config.models["embed"]
EVAL_JUDGE_LM = dspy.LM(config.models["eval_judge"])

SIM_TAU = 0.65

def score_count(result):
    """ Computes total score and count. """
    score = 0
    count = 0
    for part in result:
        for query in part["queries"]:
            count += 1
            score += int(query["is_constructible"])
    return score, count


def conciseness(result):
    """ 
    For each "correct" response, how long is it? 
    
    Note: Having only one correct response that is concise will give a "good" score for this. 
    """
    length = 0
    count = 0
    for part in result:
        for query in part["queries"]:
            if query["is_constructible"]:
                length += len(query["context"])
                count += 1
    return length / count

def eval_metrics(result):
    rs_vals = []
    ge_vals = []
    f1_vals = []

    for part in result:
        for query in part["queries"]:
            rs_vals.append(query["evidence_recall"])
            ge_vals.append(query["mean_gold_sim"])   # adjust if needed
            f1_vals.append(query["evidence_f1"])

    if len(f1_vals) == 0:
        return 0.0, 0.0, 0.0, 0.0, (0.0, 0.0)

    rs_vals = np.array(rs_vals, dtype=float)
    ge_vals = np.array(ge_vals, dtype=float)
    f1_vals = np.array(f1_vals, dtype=float)

    mean_rs = float(rs_vals.mean())
    mean_ge = float(ge_vals.mean())
    mean_f1 = float(f1_vals.mean())

    median_f1 = float(np.median(f1_vals))
    q1 = float(np.percentile(f1_vals, 25))
    q3 = float(np.percentile(f1_vals, 75))
    iqr_f1 = q3 - q1

    return mean_f1, mean_rs, mean_ge, median_f1, iqr_f1

def mean_median_query_time(result):
    times = []
    for part in result:
        for query in part["queries"]:
            times.append(query["duration"])
    mean = sum(times) / len(times)
    times.sort()
    median = times[len(times)//2]
    return mean, median

class _EvalSignature(dspy.Signature):
    """
    You are a strict judge evaluating whether the GOLD answer can be constructed from the retrieved context.

    Inputs:
    - query: the question being asked.
    - gold_answer: the gold answer text (free-form or extractive span text).
    - retrieved_context: the context produced by a retrieval method (may contain summaries and/or excerpts).

    Task:
    Return True if, using ONLY the retrieved_context, a careful reader could produce an answer that matches the
    gold_answer in meaning (or contains it, for extractive-style answers). Otherwise return False.

    Rules:
    1) Use ONLY retrieved_context. Do NOT use outside knowledge.
    2) If the needed facts are missing, unclear, or require guessing, return False.
    3) If retrieved_context contradicts gold_answer, return False.
    4) If gold_answer is a short span, it must be explicitly present in retrieved_context or unambiguously stated
       with equivalent wording.
    5) If gold_answer is longer free-form text, the essential claims must be supported by retrieved_context.

    Output:
    - is_constructible: boolean only.
    """

    query: str = dspy.InputField(desc="The question being asked.")
    gold_answer: str = dspy.InputField(desc="Gold answer text (free-form or extractive span).")
    retrieved_context: str = dspy.InputField(desc="Retrieved context; the only allowed evidence.")

    is_constructible: bool = dspy.OutputField(
        desc="True if the gold_answer is constructible from retrieved_context alone; otherwise False."
    )

###
pred_answer_eval = dspy.Predict(_EvalSignature)

async def miner_evaluate_with_gold_answers(name: str, miner: "MINER", dataset: str):
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
            query_results = []

            queries = mine_data.get("queries", [])
            gold_answers = mine_data.get("answers", [])

            with dspy.context(lm=EVAL_JUDGE_LM):
                for j, (query, gold_answer) in enumerate(zip(queries, gold_answers)):

                    # skip where answer is unanswerable or yes/no
                    if gold_answer["answer_type"] not in ["free_form", "extractive_spans"]: continue
                    # skip where author has low nlp background experience:
                    nlp_background = gold_answer.get("nlp_background")
                    if nlp_background == "zero":
                        continue

                    # graphrag retrieval
                    print(f"\rQuery {j+1}/{len(queries)}", end="")
                    q_st = time.time()
                    context:str = await miner.retrieve(query, preprocessed_chunks)
                    q_en = time.time()
                
                    ### INSERT QUERY EVALUATION ALGORITHM HERE:
                    ### (NOTE:) TO TOKENIZE: CALL Tokenizer.encode(my_string)
                    ### (NOTE:) TO EMBED: CALL resp = await litellm.aembedding(model=EMBED_MODEL, input = my_string or for batch [str1,str2,...]) THEN USE resp['data]

                    # 1. Evidence F1, R-S, G-E
                    gold_evidence:list[str] = gold_answer["evidence"]
                    if gold_evidence is None:
                        print("WARNING: found answer with no evidence. skipping")
                        continue
                
                    resp = await litellm.aembedding(model=EMBED_MODEL, input=gold_evidence)
                    G = np.array([row["embedding"] for row in resp["data"]], dtype=np.float32)

                    context_split:list[str] = context.splitlines()
                    ctx_vecs = []
                    for batch in batched(context_split, 25):
                        resp = await litellm.aembedding(model=EMBED_MODEL, input=batch)
                        ctx_vecs.extend([row["embedding"] for row in resp["data"]])
                    C = np.array(ctx_vecs, dtype=np.float32)

                    eps= 1e-12
                    # normalize & compute similarity matrix
                    G_norm = (G / np.linalg.norm(G, axis=1, keepdims=True) + eps)
                    C_norm = (C / np.linalg.norm(C, axis=1, keepdims=True) + eps)

                    sim_matrix = G_norm @ C_norm.T

                    # semantic matching: gold evidence <-> context splits
                    max_sim_per_gold = sim_matrix.max(axis=1)
                    tau = 0.8

                    recall = float((max_sim_per_gold >= tau).sum() / len(max_sim_per_gold))
                    max_sim_per_ctx = sim_matrix.max(axis=0)
                    precision = float((max_sim_per_ctx >= tau).sum() / len(max_sim_per_ctx))
                    f1 = 0.0 if (precision + recall) == 0 else float(2 * precision * recall / (precision + recall))


                    # 2. can the answer be constructed from the context?: yes/no
                    is_constructible = (await pred_answer_eval.acall(
                        query=query, gold_answer=gold_answer["text"], retrieved_context=context)).is_constructible

                    query_results.append(
                        {
                            "query": query,
                            "context": context,
                            "duration": q_en - q_st,
                            "evidence_recall": recall,
                            "evidence_precision": precision,
                            "evidence_f1": f1,
                            "mean_gold_sim": float(max_sim_per_gold.mean()),
                            "min_gold_sim": float(max_sim_per_gold.min()),
                            "is_constructible": is_constructible
                        }
                    )
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
        "Score",
        "Context Length",
        "Conciseness",
        "Query Duration (mean)",
        "Query Duration (median)",
        "Mean F1",
        "Mean RS",
        "Mean GE",
        "Median F1",
        "IQR F1"
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

        score, count = score_count(r_no_err)
        r_conciseness = conciseness(r_no_err)
        mean, median = mean_median_query_time(r_no_err)


        # avoid division by zero
        pct = (score / count * 100.0) if count else 0.0
        concise = r_conciseness if (r_conciseness and r_conciseness > 0) else 0.0
        efficiency = (pct / concise) if concise else 0.0

        # new metrics
        mean_f1, mean_rs, mean_ge, median_f1, iqr_f1 = eval_metrics(r_no_err)

        table.add_row([
            name,
            f"{pct:.2f}% ({score}/{count})" if count else "n/a (0/0)",
            f"{concise:.2f}" if concise else "n/a",
            f"{efficiency:.8f}" if efficiency else "n/a",
            f"{mean:.2f}s" if mean is not None else "n/a",
            f"{median:.2f}s" if median is not None else "n/a",
            f"{mean_f1}",
            f"{mean_rs}",
            f"{mean_ge}",
            f"{median_f1}",
            f"{iqr_f1}"
        ])

    print(table)
    print("\nERRORS:")
    print(errors)
