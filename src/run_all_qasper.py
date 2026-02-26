import asyncio
from pathlib import Path
from xrag.leanrag.miner import LeanragMINER
from xrag.parrot.miner import ParrotMINER
from xrag.utils.profiling import SimpleMiningProfiler
from xrag.vectorrag.miner import BasicVectorMINER
from xrag.kggen.miner import KGv2MINER
from xrag.hgr.miner import HypergraphMINER
from xrag.utils.eval_gold import evaluate
from xrag.utils.eval_gold import miner_evaluate_with_gold_answers

DATASET = "QASPER" # only dataset currently supporting gold context + answers

if __name__ == "__main__":

    vector_profiling = SimpleMiningProfiler(Path("./vector_profiling.json"))
    eval_routine = evaluate([
        #parrot
        miner_evaluate_with_gold_answers("parrot-qasper", ParrotMINER(), DATASET),

        # #vector
        miner_evaluate_with_gold_answers("vector-qasper", BasicVectorMINER(profiling=vector_profiling), DATASET),

        # #kggen
        miner_evaluate_with_gold_answers("kggen-qasper", KGv2MINER(), DATASET),

        # #hgr
        miner_evaluate_with_gold_answers("hgr-qasper-test-1", HypergraphMINER(), DATASET),

        #leanrag
        miner_evaluate_with_gold_answers("leanrag-qasper", LeanragMINER(), DATASET)], 
        
        concurrency=1,
    )
    
    asyncio.run(eval_routine)
    vector_profiling.finish()

    print("Profiling data:")
    for cat, (mean, median, _) in vector_profiling.category_mean_median().items():
        print(f"{cat} mean {mean*1000:.4f}ms median {median*1000:.4f}ms")
