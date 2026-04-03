import re
import json
import dspy
import string
import asyncio
import logging
import argparse
import warnings
import numpy as np
from pathlib import Path
from typing import get_args, Callable
from collections import Counter
from prettytable import PrettyTable
from tqdm.asyncio import tqdm_asyncio

from xrag.hgr.tools import preprocess_hgr_context
from xrag.paths import DATASETS_DIR, CACHE_DIR, RESULTS_DIR
from xrag.hgr.models import Domain, Query, Chunk
from xrag.hgr.HGRminer import HypergraphMINER
from xrag.config import config
from xrag.dataset_processing.preprocess import embed_call_with_retry

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
logging.getLogger("xrag.dataset_processing.preprocess").propagate = False
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

MODEL = config.models["hgr"]
LM = dspy.LM(MODEL)
dspy.configure(lm=dspy.LM(MODEL))

# delete these later
CACHE_DIR = DATASETS_DIR / "HGR" / "preprocess_cache"
RESULTS_DIR = DATASETS_DIR / "HGR" / "results"

# SCORE DISPLAY ----------------------------------------------------------------

def show_score() -> None:
    def _get_row(file: Path) -> tuple[str]:
        with open(file, "r") as f:
            queries: list[Query] = [Query(**item) for item in json.load(f)]

        all_em = [ q.em for q in queries if q.em is not None ]
        all_f1 = [ q.f1 for q in queries if q.f1 is not None ]
        all_rsim = [ q.rsim for q in queries if q.rsim is not None ]

        total = len(queries)

        avg_em = sum(all_em) / total
        avg_f1 = sum(all_f1) / total
        avg_rsim = sum(all_rsim) / total

        n_failed = len([ 1 for q in queries if q.retrieval_error == True or q.generation_error == True ])
        n_retrieval_error = len([ 1 for q in queries if q.retrieval_error == True])
        n_generation_error = len([ 1 for q in queries if q.generation_error == True])

        
        return (file.stem,
                f"{avg_em*100:.2f}%",
                f"{avg_f1*100:.2f}%",
                f"{avg_rsim*100:.2f}%",
                f"{(n_failed/total)*100:.2f}%",
                f"{n_retrieval_error}/{total}",
                f"{n_generation_error}/{total}")

    table = PrettyTable()
    table.float_format = ".2"
    table.field_names = ["Domain", "EM", "F1", "R-S", "% Failed", "Ret Err", "Gen Err"]
    for file in RESULTS_DIR.iterdir():
        table.add_row(_get_row(file))

    print(table)




# EVALUATION -------------------------------------------------------------------

def normalize_answer(answer: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))

async def _evaluate_single_query(query: Query, aggregation_fn: Callable, sem: asyncio.Semaphore) -> Query:
    
    def calculate_em(gold_list: list[str], predicted: str) -> float:
        em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 for gold in gold_list]
        return aggregation_fn(em_scores)
    
    def calculate_f1(gold_list: list[str], predicted: str) -> float:
        def compute_f1(gold: str, predicted: str) -> float:
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = 1.0 * num_same / len(predicted_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            return 2 * (precision * recall) / (precision + recall)
        f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
        return aggregation_fn(f1_scores)

    async def calculate_rsim(gold_list: list[str], predicted: str) -> float:
        
        def cosine_sim(a: list[float], b: list[float]) -> float:
            a, b = np.array(a), np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        resp = await embed_call_with_retry(to_embed=predicted)
        pred_embed = resp.data[0].embedding

        rsim_scores = []
        for gold_ans in gold_list:
            resp = await embed_call_with_retry(to_embed=gold_ans)
            gold_embed = resp.data[0].embedding 
            score = cosine_sim(pred_embed, gold_embed)
            rsim_scores.append(score)

        return aggregation_fn(rsim_scores)
    
    if query.generation_error or query.response is None:
        return query
    else:
        query.em = calculate_em(gold_list=query.golden_answers, predicted=query.response)
        query.f1 = calculate_f1(gold_list=query.golden_answers, predicted=query.response)
        query.rsim = await calculate_rsim(gold_list=query.golden_answers, predicted=query.response)
        return query

async def evaluate_all_queries(cache_path: Path, results_path: Path, aggregation_fn: Callable = np.max) -> None:
    with open(cache_path, "r") as f:
        queries: list[Query] = [Query(**item) for item in json.load(f)]

    sem = asyncio.Semaphore(16)
    queries_with_eval: list[Query] = await tqdm_asyncio.gather(
        *[ _evaluate_single_query(query=q, aggregation_fn=aggregation_fn, sem=sem) for q in queries]
    )

    with open(results_path, "w") as f:
        json.dump([q.model_dump() for q in queries_with_eval], f)
       
# GENERATION -------------------------------------------------------------------
      
class GenerateSignature(dspy.Signature):
    """You are a helpful assistant responding to questions based on given knowledge.
    Answer the given question using only the information provided in the knowledge field.
    The answer should be minimal and concise, with no preamble. Output in plain text.
    """
    question: str = dspy.InputField(desc="The question being asked.")
    knowledge: str = dspy.InputField(desc="The knowledge you can use to answer the question.")
    response: str = dspy.OutputField(desc="The answer to the question.")

generate_response = dspy.ChainOfThought(GenerateSignature)
    
async def _generate_single_query(query: Query, sem: asyncio.Semaphore) -> Query:
    if query.generation_error == False or query.retrieval_error == True:
        # already done or no retrieved context
        return query
    async with sem:
        try:
            res = await generate_response.acall(
                question=query.question,
                knowledge=query.retrieved_context)
            query.response = res.response
            query.generation_error = False
        except Exception as e:
            logger.error(f"Failed query: {e}")
            query.generation_error = True
            query.response = str(e)
        return query
        
async def generate_all_queries(cache_path: Path) -> None:
    with open(cache_path, "r") as f:
        queries: list[Query] = [Query(**item) for item in json.load(f)]
    sem = asyncio.Semaphore(16)
    queries_with_response: list[Query] = await tqdm_asyncio.gather(
        *[ _generate_single_query(query=q, sem=sem) for q in queries]
    )
    with open(cache_path, "w") as f:
        json.dump([q.model_dump() for q in queries_with_response], f)

# RETRIEVAL --------------------------------------------------------------------

async def _retrieve_single_query(query: Query,
                                 miner: HypergraphMINER,
                                 sem: asyncio.Semaphore) -> Query:
    if query.retrieval_error == False:
        # already done
        return query
    async with sem:
        try:
            context = await miner.retrieve(query.question)
            assert context != "", f"Empty context returned"
            query.retrieved_context = context       
            query.retrieval_error = False
        except Exception as e:
            if not isinstance(e, AssertionError): # assertion error is expected
                logger.error(f"Failed query: {e}")
            query.retrieval_error = True
            query.retrieved_context = str(e)
        return query
    
async def retrieve_all_queries(query_path: Path,
                               cache_path: Path,
                               miner: HypergraphMINER) -> None:
    with open(query_path, "r") as f:
        queries: list[Query] = [Query(**item) for item in json.load(f)]
    
    sem = asyncio.Semaphore(16)
    queries_with_context: list[Query] = await tqdm_asyncio.gather(
        *[ _retrieve_single_query(query=q, miner=miner, sem=sem) for q in queries]
    )
    
    with open(cache_path, "w") as f:
        json.dump([q.model_dump() for q in queries_with_context], f)

async def eval(domain: Domain):

    context_path = DATASETS_DIR / "HGR" / "contexts" / f"{domain}.json"
    query_path = DATASETS_DIR / "HGR" / "datasets" / domain / "questions.json"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results_path = results_path = RESULTS_DIR / f"{domain}.json"
    if results_path.exists():
        logger.info(f"Already done eval for domain '{domain}', skipping")
        return 

    # chunk context      
    pp_context_path = CACHE_DIR / f"{domain}_context.json"
    if not pp_context_path.exists():
        logger.info(f"Chunking context for domain '{domain}'...")
        assert context_path.exists()
        preprocess_hgr_context(context_path, pp_context_path)
    else:
        logger.info(f"Context already chunked for domain '{domain}', using cache")

    # ingest context
    miner = HypergraphMINER()
    logger.info("Starting ingestion...")
    assert pp_context_path.exists()
    await miner.ingest(pp_context_path)
    
    # ensure ingestion was complete ... 
    with open(pp_context_path, "r") as f:
        chunks = [Chunk(**item) for item in json.load(f)]
    n_failed = len([ 1 for c in chunks if c.failed == True])
    if n_failed > 0:
        logger.error(f"Edge and entity extraction for domain '{domain}' is not complete, skipping for now")
        return
    
    # create graph
    logger.info("Starting pre-retrieval...")
    graph_cache_json = CACHE_DIR / f"{domain}_graph.json"
    await miner.pre_retrieve(graph_cache_json)
    
    # query graph
    logger.info("Starting retrieval...")
    query_cache_path = CACHE_DIR / f"{domain}_queries.json"
    if not query_cache_path.exists():
        logger.info("Querying graph...")
        assert query_path.exists()
        await retrieve_all_queries(query_path=query_path, cache_path=query_cache_path, miner=miner)
    else:  
        logger.info(f"Done all queries for domain '{domain}', using cache")

    # get generation
    logger.info("Starting generation...")
    assert query_cache_path.exists()
    await generate_all_queries(cache_path=query_cache_path)

    # get score
    logger.info("Evaluating...")
    assert query_cache_path.exists()
    await evaluate_all_queries(cache_path=query_cache_path, results_path=results_path)

    logger.info(f"Done eval for domain '{domain}', results in {results_path}")

async def run_all():
    for domain in get_args(Domain):
        logger.info("Running eval on all domains")
        await eval(domain)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="agriculture")
    args = parser.parse_args()
    domain = args.domain

    if domain == "all":
        asyncio.run(run_all())
    else:
        assert args.domain in get_args(Domain)
        logger.info(f"Running eval for domain '{domain}'")
        asyncio.run(eval(domain))
    
    # show score
    logger.info("Showing score...")
    show_score()