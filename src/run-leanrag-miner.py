from pathlib import Path
import sys
CWD = Path(__name__).resolve().parent
sys.path.append(CWD)

secrets = CWD / "secrets.env"
if not secrets.is_file():
    raise ValueError(f"secrets file at '{secrets}' does not exist")

from dotenv import load_dotenv
load_dotenv(secrets)

import utils.mg_driver as mg_driver
from upsert import upsert_from_preprocessed
from leanrag_build import build
import leanrag_retrieve
import json
import time
import dspy
from dataset_preprocess import process_dataset_file
import asyncio
from aiolimiter import AsyncLimiter
from prettytable import PrettyTable
import textwrap

DATASET_DIRECTORY = Path("../datasets/MINE")
JUDGE_MODEL = dspy.LM("bedrock/us.amazon.nova-pro-v1:0")
results_dir = Path("/tmp/miner")

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
      
class LeanragMINER(MINER):
    def __init__(self):
        # tracks how many ingested articles were too small to construct a 'proper' leanrag graph
        # i.e had to fall back since recursion stopped at layer 0
        self.small_articles = set()

    async def ingest(self, preprocess_chunks_filepath:Path, preprocess_descriptions_filepath:Path):
        # create base KG
        await upsert_from_preprocessed(preprocess_chunks_filepath, preprocess_descriptions_filepath)
        self.small_articles.clear()

    async def pre_retrieve(self, article_name):
        # build LeanRAG KG with hierarchical clustering
        last_layer = await build()
        if last_layer == 0:
            self.small_articles.add(article_name)
    
    async def retrieve(self, text, preprocess_chunks_filepath:Path):
        return await leanrag_retrieve.get_response_context_data(text, preprocess_chunks_filepath)
    
    async def reset(self):
        await mg_driver.clear()

async def asd():
    await mg_driver.init()


class EvalSignature(dspy.Signature):
    """ Does the context contain the information stated in the statement?. """
    context: str = dspy.InputField()
    statement: str = dspy.InputField()
    context_contains_statement: bool = dspy.OutputField()

eval = dspy.Predict(EvalSignature)

def score_count(result):
    """ Computes total score and count. """
    score = 0
    count = 0
    for part in result:
        for query in part["queries"]:
            count += 1
            score += int(query["contained"])
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
            if query["contained"]:
                length += len(query["context"])
                count += 1
    return length / count


def mean_median_query_time(result):
    times = []
    for part in result:
        for query in part["queries"]:
            times.append(query["duration"])
    mean = sum(times) / len(times)
    times.sort()
    median = times[len(times)//2]
    return mean, median


async def miner_evaluate_individual_with_preprocess(name: str, miner: "MINER"):
    paths = list(DATASET_DIRECTORY.iterdir())
    result_file = results_dir / f"{name}.json"
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
            preprocessed_chunks = CWD / f"{p.stem}__chunks.json"
            preprocessed_descs = CWD / f"{p.stem}__g0_descriptions.json"
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
            with dspy.context(lm=JUDGE_MODEL):
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
        "Context Length",
        "Conciseness",
        "Query Duration (mean)",
        "Query Duration (median)",
    ]

    results_dir.mkdir(exist_ok=True)

    for f in results_dir.iterdir():
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

        table.add_row([
            name,
            f"{pct:.2f}% ({score}/{count})" if count else "n/a (0/0)",
            f"{concise:.2f}" if concise else "n/a",
            f"{efficiency:.2f}" if efficiency else "n/a",
            f"{mean:.2f}s" if mean is not None else "n/a",
            f"{median:.2f}s" if median is not None else "n/a",
        ])

    print(table)
    print("\nERRORS:")
    print(errors)

    
if __name__ == "__main__":
    eval_routine = evaluate(
        [miner_evaluate_individual_with_preprocess("leanrag-default", LeanragMINER())], concurrency=1
    )
    
    asyncio.run(eval_routine)