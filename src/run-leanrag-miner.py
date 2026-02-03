
import utils.mg_driver as mg_driver
from utils.upsert import upsert_from_preprocessed
from leanrag.leanrag_build import build
import leanrag.leanrag_retrieve as leanrag_retrieve
import json
import time
import dspy
from dataset.dataset_preprocess import process_dataset_file
import asyncio
from aiolimiter import AsyncLimiter
from prettytable import PrettyTable
import textwrap

if __name__ == "__main__":
    eval_routine = evaluate(
        [miner_evaluate_individual_with_preprocess("leanrag-ours-1", LeanragMINER())], concurrency=1
    )
    
    asyncio.run(eval_routine)