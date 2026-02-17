import asyncio
from xrag.leanrag.miner import LeanragMINER
# from xrag.utils.eval import evaluate
from xrag.utils.eval_gold import evaluate, miner_evaluate_with_gold_answers

DATASET = "QASPER_TEST" # only dataset currently supporting gold context + answers

if __name__ == "__main__":
    #leanrag
    eval_routine = evaluate(
        [miner_evaluate_with_gold_answers("leanrag-qasper-test-1", LeanragMINER(), DATASET)], concurrency=1
    )
    
    asyncio.run(eval_routine)