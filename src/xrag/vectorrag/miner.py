import litellm
import numpy as np
from xrag.utils.eval import MINER
from xrag.config import config
from xrag.dataset_processing.preprocess import embed_call_with_retry
from pathlib import Path
import ijson
from xrag.utils.profiling import SimpleMiningProfiler

class BasicVectorMINER(MINER):
	""" 
	A MINER implementation for a very simple vector RAG system. 
	
	This implementation relies on externally processed chunk data. 
	"""

	chunks: list[str] = []
	embeddings: list = []

	def __init__(
		self,
		# chunk_size: int = 200,
		# overlap: int = 20,
		quantile: float = 0.95,
		profiling: SimpleMiningProfiler | None = None
	):
		# self.chunk_size = chunk_size
		# self.overlap = overlap
		self.quantile = quantile 
		self.embedding_model = config.models["embed"]
		self.profiling = profiling

	async def ingest(self, chunks: Path, descriptions: Path):
		with open(chunks, "rb") as in_file:
			for item in ijson.items(in_file, "item"):
				for chunk in item["chunks"]:
					self.chunks.append(chunk["raw_text"])
					self.embeddings.append(np.array(chunk["embedding"]))

					# new_chunks = [text[i*self.chunk_size:(i+1)*self.chunk_size+self.overlap] for i in range(0, len(text)//self.chunk_size)]
					# new_embeddings = await embed_call_with_retry(new_chunks)
					# new_embeddings = [np.array(e.embedding) for e in new_embeddings.data]

					# self.chunks += new_chunks
					# self.embeddings += new_embeddings
	
	async def pre_retrieve(self, article_name):
		pass

	async def retrieve(self, query_text, preprocess_chunks_filepath: Path) -> str:
		# Make embedding 
		if self.profiling:
			retrieve_embedding_event = self.profiling.start("retrieve_embedding")
		text_embedding = (await litellm.aembedding(self.embedding_model, input=[query_text]))
		text_embedding = np.array(text_embedding.data[0].embedding)
		if self.profiling:
			# There must be a better way to do this but I do not know of it 
			retrieve_embedding_event.end()

		# Find similarities 
		if self.profiling:
			retrieve_similarity_event = self.profiling.start("retrieve_similarity")
		cosine = np.abs(np.dot(self.embeddings, text_embedding) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(text_embedding)))
		sims = np.argsort(cosine)[::-1]
		if self.profiling:
			retrieve_similarity_event.end()

		# Find relevant 
		cutoff = np.quantile(sims, self.quantile)
		where = sims >= cutoff
		indices = np.nonzero(where)[0]

		# No need for ordering 
		return "\n".join([self.chunks[i] for i in indices])

	async def reset(self):
		self.chunks = []
		self.embeddings = []
