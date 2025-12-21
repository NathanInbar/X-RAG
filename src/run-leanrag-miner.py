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

async def miner_evaluate_individual_with_preprocess(name:str, miner: MINER):
    paths = list(DATASET_DIRECTORY.iterdir())
    result_file = results_dir/f"{name}.json"
    print(f"Writing results file to '{result_file}'")
    with open(result_file, "w") as fp:
        fp.write('{"name": '+f'"{name}"'+ ', "result": [\n')
    for i, p in enumerate(paths): 
        if p.name != "A Day in the Life of an Astronaut.json": continue
        try:
            print(f"START EVAL: {p.name} ({i+1}/{len(paths)})")

            # Load data 
            with open(p, "r") as fp:
                mine_data = json.load(fp)

            # Creates pre-process data (chunks + g0 descriptions json files) - check if it exists already from prev runs
            preprocessed_chunks = CWD / f"{p.stem}__chunks.json"
            preprocessed_descs = CWD / f"{p.stem}__g0_descriptions.json"
            if (not preprocessed_chunks.is_file()) or (not preprocessed_descs.is_file()):
                await process_dataset_file(p)

            # Ingest text 
            print("Ingesting...")
            ingest_st = time.time()
            await miner.ingest(preprocessed_chunks, preprocessed_descs)
            await miner.pre_retrieve(p.name)
            ingest_en = time.time()
            # Query and evaluate
            print("Evaluating...")
            queries = []
            with dspy.context(lm=JUDGE_MODEL):
                for j, a in enumerate(mine_data["answers"]):
                    print(f"\rQuery {j+1}/{len(mine_data["answers"])}", end="")
                    q_st = time.time()
                    info = await miner.retrieve(a, preprocessed_chunks)
                    q_en = time.time()
                    contained = (await eval.acall(context=info, statement=a)).context_contains_statement
                    queries.append({
                        "query": a,
                        "context": info,
                        "contained": contained,
                        "duration": q_en - q_st,
                    })
                print("")
            result = {
                "filename": p.name,
                "ingest_duration": ingest_en - ingest_st,
                "queries": queries,
            }
            await miner.reset()
        except Exception as e:
            result = {"error": str(e)}
        finally:
            with open (result_file, "a") as fp:
                json.dump(result, fp)
                if i==0 :#i < len(paths)-1
                    fp.write(",")
                fp.write("\n")
            print(f"wrote result to {result_file}")
    with open (result_file, "a") as fp:
        fp.write("]}")

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
        print(f"Reading {f} ...")
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        
        results:list = data["result"]
        r_no_err = []
        for r in results:
            if 'error' in r:
                errors.append(r)
            else: r_no_err.append(r)

        score, count = score_count(r_no_err)
        r_conciseness = conciseness(r_no_err)
        mean, median = mean_median_query_time(r_no_err)
        table.add_row([
            data["name"], 
            f"{score/count*100:.2f}% ({score}/{count})", 
            f"{r_conciseness:.2f}",
            f"{score/count*100/r_conciseness:.2f}",
            f"{mean:.2f}s",
            f"{median:.2f}s",
        ])
    print(table)
    print("\nERORRS:")
    print(errors)
    
if __name__ == "__main__":
    eval_routine = evaluate(
        [miner_evaluate_individual_with_preprocess("leanrag-default", LeanragMINER())], concurrency=1
    )
    
    asyncio.run(eval_routine)