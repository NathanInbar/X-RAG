from xrag.hgr_old.tools import aggregate, extract_edges_entities, query
import dspy
from xrag.utils.eval import MINER
from xrag.config import config
from pathlib import Path
import ijson

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
		self.embedding_model = config.models["embed"]
	
	async def ingest(self, preprocess_chunk_json:Path, _:Path):
		error_count = 0
		total = 0
		# print("\n")
		with dspy.context(lm=dspy.LM(self.model)):
			with open(preprocess_chunk_json, "rb") as in_file:
				for itm in ijson.items(in_file, "item"):
					for i,chunk in enumerate(itm['chunks']):
						total += 1
						text = chunk["raw_text"]
						k, e = await extract_edges_entities(text)
						if not (k and e):
							error_count += 1
							print(f"\r{error_count} chunks failed")
						self.kb += k
						self.eb += e
		assert error_count == 0, f"Failed ingestion on {error_count} chunks"

	async def pre_retrieve(self, article_name):
		with dspy.context(lm=dspy.LM(self.model)):
			self.hg = await aggregate(self.kb, self.eb, self.embedding_model)

	async def retrieve(self, query_text, preprocess_chunks_filepath:Path) -> str:
		assert not (self.hg is None)
		with dspy.context(lm=dspy.LM(self.model)):
			k = await query(query_text, self.hg, self.embedding_model, kv=self.kv, tv=self.tv, kh=self.kh, th=self.th)
		return k

	async def reset(self):
		self.kb = []
		self.eb = []
		self.hg = None