from xrag.utils.eval import MINER

class ParrotMINER(MINER):
	kb = ""

	def __init__(self):
		pass

	async def ingest(self, text: str):
		self.kb += text + "\n"
		print(f"Parrot ingested {len(self.kb)} characters.")
	
	async def pre_retrieve(self):
		pass

	async def retrieve(self, text: str) -> str:
		return self.kb

	async def reset(self):
		self.kb = ""