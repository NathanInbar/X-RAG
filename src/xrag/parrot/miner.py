from xrag.utils.eval import MINER
from pathlib import Path
import ijson

class ParrotMINER(MINER):
	kb = ""

	def __init__(self):
		pass

	async def ingest(self, chunks: Path, descriptions: Path):
		with open(chunks, "rb") as in_file:
			for item in ijson.items(in_file, "item"):
				for chunk in item["chunks"]:
					self.kb += chunk["raw_text"] + "\n"

		print(f"Parrot ingested {len(self.kb)} characters.")
	
	async def pre_retrieve(self, article_name):
		pass

	async def retrieve(self, query_text, preprocess_chunks_filepath:Path) -> str:
		return self.kb

	async def reset(self):
		self.kb = ""
