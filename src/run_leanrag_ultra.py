import asyncio
from xrag.leanrag.miner import LeanragMINER
from xrag.utils.eval_faithful import evaluate, miner_evaluate_faithful

DATASET = "ULTRA"

if __name__ == "__main__":
    eval_routine = evaluate(
        [miner_evaluate_faithful("leanrag-ultra", LeanragMINER(), DATASET)],
        concurrency=1
    )
    asyncio.run(eval_routine)
