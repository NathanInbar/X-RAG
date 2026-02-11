
import asyncio
from xrag.utils.eval import evaluate, miner_evaluate_individual
from xrag.models.leanrag.miner import LeanragMINER
from xrag.models.parrot.miner import ParrotMINER
from xrag.models.vector.miner import BasicVectorMINER
from xrag.models.kggen.miner import KGv2MINER
from xrag.models.hgr.miner import HypergraphMINER
from xrag.models.karenrag.miner import KarenMINER
from xrag.config import config

DATASET = config.dataset

if __name__ == "__main__":
    eval_routine = evaluate(
        [
            # # Parrot
            # miner_evaluate_individual("parrot", ParrotMINER(), DATASET),
            # miner_evaluate_individual("karen-test", KarenMINER(), DATASET),
         
            # # # VectorRAG
            # miner_evaluate_individual("vector_200_20_95", BasicVectorMINER(), DATASET),
            # miner_evaluate_individual("vector_200_20_85", BasicVectorMINER(quantile=0.85), DATASET),
            # miner_evaluate_individual("vector_200_0_95", BasicVectorMINER(overlap=0), DATASET),
            # miner_evaluate_individual("vector_100_20_95", BasicVectorMINER(chunk_size=100), DATASET),
            # miner_evaluate_individual("vector_100_20_85", BasicVectorMINER(chunk_size=100, quantile=0.85), DATASET),
            # miner_evaluate_individual("vector_100_0_95", BasicVectorMINER(chunk_size=100, overlap=0), DATASET),

            # # HypergraphRAG
            # miner_evaluate_individual("hgr_60_50_60_5", HypergraphMINER(60, 50, 60, 5), DATASET),
            # miner_evaluate_individual("hgr_60_30_60_3", HypergraphMINER(60, 30, 60, 3), DATASET),

            # # KG-GEN
            # miner_evaluate_individual("kg-gen", KGv2MINER(), DATASET),

            # LeanRAG
            miner_evaluate_individual("leanrag-ours-test-1", LeanragMINER(), DATASET, True)

        ], concurrency=1
    )
    
    asyncio.run(eval_routine)