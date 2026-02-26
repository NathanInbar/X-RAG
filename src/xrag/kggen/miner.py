import json
from xrag.utils.eval import MINER
from xrag.config import config
from xrag.kggen.tools import extract, make_graph, resolve, query
import dspy
import ijson
from pathlib import Path

class KGv2MINER(MINER):
	"""
	An implementation of kg-gen retooled to use externally processed data. 
	"""

	kb: list[tuple[str, str, str]] = []
	eb: list[str] = []
	kg = None

	def __init__(self):
		self.model = config.models["description_gen"]
		self.embedding_model = config.models["embed"]
	
	async def ingest(self, chunks: Path, descriptions: Path):
		with open(chunks, "r") as fp:
			chunks_data = json.load(fp)
		for item in chunks_data:
			triplets = [(d["s"], d["p"], d["o"]) for chunk in item["chunks"] for d in chunk["triples"]]
			entities = list(set([e for a, _, b in triplets for e in [a, b]]))

			self.kb += triplets
			self.eb += entities

		# with dspy.context(lm=dspy.LM(self.model)):
		# 	with open(chunks, "rb") as in_file:
		# 		for item in ijson.items(in_file, "item"):
		# 			for chunk in item["chunks"]:
		# 				text = chunk["raw_text"]

		# 				e, k = await extract(text)

		# 				self.kb += k
		# 				self.eb += e
	
	async def pre_retrieve(self, article_name):
		with dspy.context(lm=dspy.LM(self.model)):
			self.eb, self.kb = await resolve(self.eb, self.kb)
			self.kg = await make_graph(self.eb, self.kb, self.embedding_model)

	async def retrieve(self, query_text, preprocess_chunks_filepath:Path) -> str:
		assert not (self.kg is None)
		with dspy.context(lm=dspy.LM(self.model)):
			k = await query(query_text, self.kg, self.embedding_model)
		return "\n".join([" ".join(t) for t in k])

	async def reset(self):
		self.kb = []
		self.eb = []
		self.kg = None
