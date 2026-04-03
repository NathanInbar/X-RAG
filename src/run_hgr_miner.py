
import asyncio
import argparse

from xrag.paths import DATASETS_DIR
from xrag.hgr.miner import HypergraphMINER as newHGRMINER
from xrag.hgr_old.miner import HypergraphMINER as oldHGRMINER
from xrag.utils.eval import evaluate, miner_evaluate_individual_with_preprocess

# DATASET = config.dataset
# # DATASET = "OURS_TEST"

parser = argparse.ArgumentParser()
parser.add_argument("dataset")

if __name__ == "__main__":
    args = parser.parse_args()

    if not args.dataset:
        raise RuntimeError("Must pass dataset name")
    dataset = args.dataset
    if not (DATASETS_DIR / dataset).exists():
        raise FileNotFoundError(f"{DATASETS_DIR / dataset} does not exist.")
    
    eval_routine = evaluate(
        [
            # miner_evaluate_individual_with_preprocess("updated-hgr-ours", newHGRMINER(), dataset),
            miner_evaluate_individual_with_preprocess(f"old-hgr-{dataset}", oldHGRMINER(), dataset)
        ], concurrency=1
    )
    
    asyncio.run(eval_routine)