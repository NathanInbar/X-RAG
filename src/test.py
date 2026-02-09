import asyncio
import json
import logging
from pathlib import Path
from xrag.paths import DATASETS_DIR, CACHE_DIR
from xrag.utils import mg_driver
from xrag.dataset_processing.preprocess import process_dataset_file
from xrag.leanrag.miner import LeanragMINER

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='umap')

LIMIT_LOOPS = 0
MAX_LOOPS = 1

logging.basicConfig(
    level=logging.INFO,
    format=' %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger("main")

async def main():
    await mg_driver.init()
    miner =  LeanragMINER()
    await miner.reset()

    # create a clear error output file
    debug_file = "debug_errors.json"
    with open(debug_file, "w") as f:
        pass

    datadir = Path(DATASETS_DIR) / Path("OURS/JSON Mine Dataset")
    total_files = len(list(datadir.iterdir()))
    count = 0
    errors = []
    for f in datadir.iterdir():
        count += 1
        logger.info(f"Starting {f.stem} ({count}/{total_files})")

        try:
            # check if cached preprocess exists
            chunk_cache = Path(CACHE_DIR) / Path(f"{f.stem}__chunks.json")
            desc_cache = Path(CACHE_DIR) / Path(f"{f.stem}__g0_descriptions.json")
            
            if not chunk_cache.exists() or not desc_cache.exists():
                # preprocess
                await process_dataset_file(f)

            # ingest
            await miner.ingest(chunk_cache, desc_cache)

            # build
            await miner.pre_retrieve(f.stem)
            logger.info(f"Done {f.stem}")

            await miner.reset()

            if LIMIT_LOOPS and count == MAX_LOOPS: break
        except Exception as e:
            logger.error(f"Failed for file {f.name}: {e}")

            error = {"filename": f.name, "error": str(e)}
            errors.append(error)
            with open(debug_file, "a") as f:
                json.dump(error, f)

            await miner.reset()

if __name__ == "__main__":
    asyncio.run(main())