import asyncio
from xrag.leanrag.miner import LeanragMINER
from xrag.parrot.miner import ParrotMINER
from xrag.vectorrag.miner import BasicVectorMINER
from xrag.kggen.miner import KGv2MINER
from xrag.hgr.miner import HypergraphMINER
from xrag.utils.eval_gold import evaluate
from xrag.utils.eval_gold import miner_evaluate_with_gold_answers

DATASET = "QASPER" # only dataset currently supporting gold context + answers

if __name__ == "__main__":

    eval_routine = evaluate([
        #parrot
        miner_evaluate_with_gold_answers("parrot-qasper", ParrotMINER(), DATASET),

        # #vector
        miner_evaluate_with_gold_answers("vector-qasper", BasicVectorMINER(), DATASET),

        # #kggen
        miner_evaluate_with_gold_answers("kggen-qasper", KGv2MINER(), DATASET),

        # #hgr
        # BUG:
        # ERROR: axis 1 is out of bounds for array of dimension 1
        # look in hgr tools at the np linalg calls
        
        # miner_evaluate_with_gold_answers("hgr-qasper-test-1", HypergraphMINER(), DATASET),

        #leanrag
        miner_evaluate_with_gold_answers("leanrag-qasper", LeanragMINER(), DATASET)], 
        
        concurrency=1
    )
    
    asyncio.run(eval_routine)