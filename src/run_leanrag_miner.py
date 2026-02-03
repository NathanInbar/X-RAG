
import asyncio
from xrag.utils.eval import evaluate, miner_evaluate_individual_with_preprocess
from xrag.leanrag.miner import LeanragMINER

if __name__ == "__main__":
    #leanrag
    eval_routine = evaluate(
        [miner_evaluate_individual_with_preprocess("leanrag-ours-1", LeanragMINER(), "OURS")], concurrency=1
    )
    
    asyncio.run(eval_routine)