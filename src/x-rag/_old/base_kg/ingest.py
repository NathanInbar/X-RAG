import asyncio
import sys
import json
import logging

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
print(ROOT)
sys.path.insert(0, str(ROOT))

from utils.models import subject, predicate, obj
from utils.models import Chunk
from utils.db import ChunkDatabase
from utils.segmenter import TextSegmenter
from base_kg.extract_dspy import TripletExtractor
from dotenv import load_dotenv

# config
TEXT_SEGMENTER_MODEL = "sat-3l-sm"
TRIPLET_EXTRACTOR_MODEL = "bedrock/us.amazon.nova-pro-v1:0"
EMBED_MODEL = "bedrock/amazon.titan-embed-text-v2:0"

DATASET = "MINE"
AWS_REGION = "ca-central-1"
SECRETS_ENV = "../../secrets.env"

# internal global
_DATASET_DIR = "datasets"

segmenter = None
triplet_extractor = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def ingestion_task(file_path:Path):
    """
    Ingest a single json document into the graph
    """
    print(f"ingesting file '{file_path}' ..")

    # expect json object 'essay' key:
    with open(file_path, "r") as file:
        data = json.load(file)
        if not "essay" in data:
            print(f"dataset file '{file_path}' is incorrectly formatted: missing key 'essay'")
            return
        
    dataset_text = data['essay']
    chunk_texts = segmenter.create_segments(dataset_text)
    logger.debug(f"CHUNKS: \n{chunk_texts}\n")

    for txt in chunk_texts:
        triplets: list[tuple[subject,predicate,obj]] = await triplet_extractor.extract(txt)
        logger.debug(f"\nTRIPLETS: \n{triplets}\n")
    # get embedding for chunk text

    # assemble chunk object & add to chunk database


async def ingest_dataset(dataset_folder_name:str) -> None:
    """
    Produce a base knowledge graph of Entity - Relation -> Entity triplets
    in memgraph with chunk data in a ChunkDatabase
    """
    base:Path = Path(__file__).resolve()
    target:Path = base.parents[3] / _DATASET_DIR /dataset_folder_name

    ingest_tasks:list[asyncio.Task] = []
    for p in target.iterdir():
        if p.is_file() and str(p).endswith(".json"):
            # create an ingestion task
            ingest_tasks.append(asyncio.create_task(ingestion_task(p)))
            break # TEMP : only ingest 1 document

    await asyncio.gather(*ingest_tasks)
    print("Ingestion of dataset complete.")


if __name__ == "__main__" :

    load_dotenv(SECRETS_ENV)

    segmenter = TextSegmenter()
    segmenter.configure(model= TEXT_SEGMENTER_MODEL)

    triplet_extractor = TripletExtractor(model= TRIPLET_EXTRACTOR_MODEL)

    asyncio.run(ingest_dataset(DATASET))