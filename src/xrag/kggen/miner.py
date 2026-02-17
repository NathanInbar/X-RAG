from xrag.utils.eval import MINER
from xrag.config import config
from xrag.kggen.tools import extract, make_graph, resolve, query
import dspy

class KGv2MINER(MINER):
	kb: list[tuple[str, str, str]] = []
	eb: list[str] = []
	kg = None

	def __init__(self):
		self.model = config.models["description_gen"]
		self.embedding_model = config.models["embed"]
	
	async def ingest(self, text: str):
		with dspy.context(lm=dspy.LM(self.model)):
			e, k = await extract(text)
		self.kb += k
		self.eb += e
	
	async def pre_retrieve(self):
		with dspy.context(lm=dspy.LM(self.model)):
			self.eb, self.kb = await resolve(self.eb, self.kb)
			self.kg = await make_graph(self.eb, self.kb, self.embedding_model)

	async def retrieve(self, text: str) -> str:
		assert not (self.kg is None)
		with dspy.context(lm=dspy.LM(self.model)):
			k = await query(text, self.kg, self.embedding_model)
		return "\n".join([" ".join(t) for t in k])

	async def reset(self):
		self.kb = []
		self.eb = []
		self.kg = None