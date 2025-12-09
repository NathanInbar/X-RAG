import asyncio
import sys
import json
import dspy

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
print(ROOT)
sys.path.insert(0, str(ROOT))

from utils.models import Chunk
from utils.db import ChunkDatabase
from utils.segmenter import TextSegmenter

DATASET_DIR = "datasets"

segmenter = None

class ExtractTriples(dspy.Signature):
    """
    Extract subject predicate object triples from the source text.
    Be thorough, accurate, and faithful to the source text.
    Return a JSON array under 'triples_json' with objects:
      {"subject": "...", "predicate": "...", "object": "..."}
    """

    source_text = dspy.InputField()
    triples_json = dspy.OutputField(desc="JSON array of {subject,predicate,object}")

async def ingestion_task(file_path:Path):
    """
    Ingest a single json document into the graph
    """
    print(f"ingesting file '{file_path}'")

    # expect json object 'essay' key:
    with open(file_path, "r") as file:
        data = json.load(file)
        if not "essay" in data:
            print(f"dataset file '{file_path}' is incorrectly formatted: missing key 'essay'")
            return
        
    dataset_text = data['essay']
    chunk_texts = segmenter.create_segments(dataset_text)

    for txt in chunk_texts:
        ... #todo - call dspy extraction , upsert to memgraph , ..

    # get embedding for chunk text

    # assemble chunk object & add to chunk database


async def ingest_dataset(dataset_folder_name:str) -> None:
    """
    Produce a base knowledge graph of Entity - Relation -> Entity triplets
    in memgraph with chunk data in a ChunkDatabase
    """
    base:Path = Path(__file__).resolve()
    target:Path = base.parents[3] / DATASET_DIR /dataset_folder_name

    ingest_tasks:list[asyncio.Task] = []
    for p in target.iterdir():
        if p.is_file() and str(p).endswith(".json"):
            # create an ingestion task
            ingest_tasks.append(asyncio.create_task(ingestion_task(p)))

    await asyncio.gather(*ingest_tasks)
    print("Ingestion of dataset complete.")


if __name__ == "__main__" :
    segmenter = TextSegmenter()
    asyncio.run(ingest_dataset("MINE"))