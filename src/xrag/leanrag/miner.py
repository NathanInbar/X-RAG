from pathlib import Path
from xrag.utils import mg_driver
from xrag.utils.eval import MINER
from xrag.utils.upsert import upsert_from_preprocessed
from xrag.leanrag.build import build
from xrag.leanrag.retrieve import get_response_context_data

class LeanragMINER(MINER):
    def __init__(self):
        # tracks how many ingested articles were too small to construct a 'proper' leanrag graph
        # i.e had to fall back since recursion stopped at layer 0
        self.small_articles = set()

    async def ingest(self, preprocess_chunks_filepath:Path, preprocess_descriptions_filepath:Path):
        # create base KG
        await upsert_from_preprocessed(preprocess_chunks_filepath, preprocess_descriptions_filepath)
        self.small_articles.clear()

    async def pre_retrieve(self, article_name):
        # build LeanRAG KG with hierarchical clustering
        last_layer = await build()
        if last_layer == 0:
            self.small_articles.add(article_name)
    
    async def retrieve(self, query_text, preprocess_chunks_filepath:Path):
        return "\n".join(await get_response_context_data(query_text, preprocess_chunks_filepath))
    
    async def reset(self):
        await mg_driver.clear()