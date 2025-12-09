import asyncio
import sys

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
print(ROOT)
sys.path.insert(0, str(ROOT))

from utils.models import Chunk
from utils.db import ChunkDatabase

DATASET_DIR = "datasets"

async def ingest_dataset(dataset_folder_name:str) -> None:
    """
    Produce a base knowledge graph of Entity - Relation -> Entity triplets
    in memgraph with chunk data in a ChunkDatabase
    """
    base = Path(__file__).resolve()
    target = base.parents[3] / DATASET_DIR /dataset_folder_name

    ingest_tasks = []
    for p in target.iterdir():
        if p.is_file() and str(p).endswith(".json"):
            # create an ingestion task
            ingest_tasks.append(asyncio.create_task(ingest_task(p)))

    await asyncio.gather(*ingest_tasks)
    print("Ingestion of dataset complete.")

async def ingest_task(file_path):
    """
    Ingest a single json document into the graph
    """
    print(f"ingesting file '{file_path}'")

if __name__ == "__main__" :
    asyncio.run(ingest_dataset("MINE"))