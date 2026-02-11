from xrag.utils.eval import MINER, trim_to_optimal

class KarenMINER(MINER):
	kb = ""

	def __init__(self):
		pass

	async def ingest(self, text: str):
		self.kb += text + "\n"
		print(f"Parrot ingested {len(self.kb)} characters.")
	
	async def pre_retrieve(self):
		pass
		
	async def retrieve(self, query: str) -> str:
		return trim_to_optimal(self.kb, query)

	async def reset(self):
		self.kb = ""