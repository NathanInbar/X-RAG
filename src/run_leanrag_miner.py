
import asyncio
from xrag.utils.eval import evaluate, miner_evaluate_individual_with_preprocess, miner_evaluate_bulk
from xrag.leanrag.miner import LeanragMINER
from xrag.config import config

DATASET = config.dataset

if __name__ == "__main__":
    #leanrag: individual
    eval_routine = evaluate(
        [miner_evaluate_individual_with_preprocess("leanrag-ours-test-1", LeanragMINER(), DATASET)], concurrency=1
    )
    
    asyncio.run(eval_routine)