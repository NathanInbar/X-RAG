
import asyncio
from xrag.utils.eval import evaluate, miner_evaluate_individual_with_preprocess
from xrag.leanrag.miner import LeanragMINER
from xrag.hgr.miner import HypergraphMINER as newHGRMINER
from xrag.kggen.miner import KGv2MINER
from xrag.parrot.miner import ParrotMINER
from xrag.vectorrag.miner import BasicVectorMINER
from xrag.config import config
from xrag.hgr_old.miner import HypergraphMINER as oldHGRMINER

DATASET = config.dataset
# DATASET = "OURS_TEST"

if __name__ == "__main__":
    #leanrag
    eval_routine = evaluate(
        [
            miner_evaluate_individual_with_preprocess("updated-hgr-ours", newHGRMINER(), DATASET),
            miner_evaluate_individual_with_preprocess("old-hgr-ours", oldHGRMINER(), DATASET)
        ], concurrency=1
    )
    
    asyncio.run(eval_routine)