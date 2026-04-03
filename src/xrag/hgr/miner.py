import ijson
import dspy
from pathlib import Path

from xrag.config import config
from xrag.utils.eval import MINER
from xrag.hgr.tools import aggregate, extract_all_edges_entities, query
from xrag.paths import CACHE_DIR

MODEL = config.models["hgr"]

class HypergraphMINER(MINER):
	kb = []
	eb = []
	hg = None

	def __init__(
		self,
		# Settings from paper
		kv: int = 60,
		tv: int = 50,
		kh: int = 60,
		th: int = 5,
		# This model doesn't make an error! Many do... 
		model: str = MODEL,
	):
		self.kv = kv
		self.tv = tv
		self.kh = kh
		self.th = th
		self.model = model

	def _graph_cache_path(self, article_name):
		return CACHE_DIR / f"{article_name}_graph.json"
	
	async def ingest(self, preprocess_chunk_json:Path, _:Path):
		chunks = []
		with open(preprocess_chunk_json, "rb") as in_file:
				for itm in ijson.items(in_file, "item"):
					for chunk in itm['chunks']:
						text = chunk["raw_text"]
						chunks.append(text)

		k, e = await extract_all_edges_entities(chunks)
		assert k and e
		self.kb = k
		self.eb = e

	async def pre_retrieve(self, article_name: str) -> None:
		graph_cache_json = self._graph_cache_path(article_name)
		self.hg = await aggregate(self.kb, self.eb, cache_path=graph_cache_json)

	async def retrieve(self, query_text: str, _: Path) -> str:
		assert not (self.hg is None)
		return (await query(query_text, self.hg, kv=self.kv, tv=self.tv, kh=self.kh, th=self.th))

	async def reset(self) -> None:
		self.kb = []
		self.eb = []
		self.hg = None