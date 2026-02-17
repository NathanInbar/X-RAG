import asyncio
from xrag.leanrag.miner import LeanragMINER
from xrag.parrot.miner import ParrotMINER
from xrag.vectorrag.miner import BasicVectorMINER
from xrag.kggen.miner import KGv2MINER
from xrag.hgr.miner import HypergraphMINER
from xrag.utils.eval_gold import evaluate
from xrag.utils.eval_gold import miner_evaluate_with_gold_answers

DATASET = "QASPER_TEST" # only dataset currently supporting gold context + answers

if __name__ == "__main__":

    eval_routine = evaluate([
        #parrot
        # miner_evaluate_with_gold_answers("parrot-qasper-test-1", ParrotMINER(), DATASET),

        # #vector
        # miner_evaluate_with_gold_answers("vector-qasper-test-1", BasicVectorMINER(), DATASET),

        # #kggen
        # miner_evaluate_with_gold_answers("kggen-qasper-test-1", KGv2MINER(), DATASET),

        # #hgr
        # miner_evaluate_with_gold_answers("hgr-qasper-test-1", HypergraphMINER(), DATASET),

        #leanrag
        miner_evaluate_with_gold_answers("leanrag-qasper-test-1", LeanragMINER(), DATASET)], 
        
        concurrency=1
    )
    
    asyncio.run(eval_routine)