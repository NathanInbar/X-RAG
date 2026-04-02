import json
import dspy
from pathlib import Path

from xrag.config import config
from xrag.utils.eval import MINER
from xrag.hgr.tools import aggregate, extract_all_edges_entities, query, preprocess_hgr_context
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
	
	async def preprocess(self, context_path: Path, output_path: Path) -> None:
		if not output_path.exists():
			preprocess_hgr_context(context_path=context_path, output_path=output_path)

	async def ingest(self, pp_context_path: Path) -> None:
		chunks = await extract_all_edges_entities(pp_context_path)
		self.kb = [k for c in chunks for k in c.kb]
		self.eb = [e for c in chunks for e in c.e]

	async def pre_retrieve(self, graph_cache_json: Path) -> None:
		self.hg = await aggregate(self.kb, self.eb, cache_path=graph_cache_json)

	async def retrieve(self, query_text: str) -> str:
		assert not (self.hg is None)
		return (await query(query_text, self.hg, kv=self.kv, tv=self.tv, kh=self.kh, th=self.th))

	async def reset(self) -> None:
		self.kb = []
		self.eb = []
		self.hg = None