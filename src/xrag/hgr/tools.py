import json
import dspy
import logging
import asyncio
import litellm
import numpy as np
import networkx as nx
from tqdm.asyncio import tqdm_asyncio
from pathlib import Path

from xrag.dataset_processing.preprocess import embed_call_with_retry
from xrag.hgr.models import Domain, Chunk, Query
from xrag.config import config
from xrag.utils import stable_id_hex

logger = logging.getLogger(__name__)

MODEL = config.models["hgr"]

# CONTEXT PREPROCESSING: ONLY USED FOR HGR EVALUATION STYLE --------------------

def _chunk_by_char(text: str, max_chars: int = 10000, overlap_chars: int = 1000) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start += max_chars - overlap_chars
    return chunks

# same chunking as the paper
def _chunk_by_token(text: str, max_tokens: int = 1200, overlap: int = 100) -> list[str]:
    try:
        encode = lambda s: litellm.encode(model=MODEL, text=s)
        decode = lambda t: litellm.decode(model=MODEL, tokens=t)
    except Exception:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        encode, decode = enc.encode, lambda t: enc.decode(t).decode("utf-8", errors="replace")

    tokens = encode(text)
    step = max_tokens - overlap
    return [decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), step)]

def preprocess_hgr_context(context_path: Path, output_path: Path) -> list[Chunk]:
	with open(context_path, "r") as f:
		context_list = json.load(f)

	context_chunks = []
	for context in context_list:
		for chunk_text in _chunk_by_token(text=context):
			context_chunks.append(Chunk(id=stable_id_hex(chunk_text), raw_text=chunk_text))

	with open(output_path, "w") as f:
		json.dump([c.model_dump() for c in context_chunks], f)

# EDGE EXTRACTION: GIVEN TEXT CHUNK, EXTRACT KNOWLEDGE SEGMENTS AND ENTITIES --------------------

class EdgeExtractSignature(dspy.Signature):
	"""
	Given a text document that is potentially relevant to this activity and a list of entitiy types, identify all entities of theose types from the text and all relationships among the identified entities. 

	# Steps

	1. Divide the text into several complete knowledge segments. For each knowledge segment, extract the following information: 
	- knowledge segment: A sentence that describes the context of the knowledge segment. 
	- completeness score: A score form 0 to 10 indicating the completeness of the knowledge segment. 
	Output to the 'knowledge_segments' field a list of all knowledge segments formatted as (<knowledge_segment>,<completeness_score>).
	Make the final item of the list the tuple ("All Done!", -1) when complete.

	2. Identify all entities in each knowledge segment. For each identified entity, extract the following information: 
	- entity name: Name of the entity. 
	- entity type: Type of the entity. 
	- entity description: Comprehensive description of the entity's attributes and activities. 
	- key score: A score from 0 to 100 indicating the importance of the entity in the text. 
	Output to the 'entities' field a list of all entities formatted as (<entity_name>,<entity_type>,<entity_description>,<key_score>).
	Make the final item of the list the tuple ("All Done!", "", "", -1) when complete.

	If the text is meaningless, garbled, or contains nothing to extract, return [("All Done!", -1)] for knowledge_segments and [("All Done!", "", "", -1)] for entities.
	"""

	text: str = dspy.InputField()
	knowledge_segments: list[tuple[str, int]] = dspy.OutputField(desc="a list of knowledge fragments and their completeness scores, formatted as (<knowledge_segment>,<completeness_score>) ")
	entities: list[tuple[str, str, str, int]] = dspy.OutputField(desc="a list of entity names, types, descriptions, and scores, formatted as (<entity_name>,<entity_type>,<entity_description>,<key_score>)")

edge_extract = dspy.Predict(EdgeExtractSignature)

async def _extract_edges_entities(chunk: Chunk, sem: asyncio.Semaphore, max_retries: int = 3):
	if chunk.kb and chunk.e:
		# chunk already has knowledge segments and entities extracted
		return chunk
	async with sem:
		for attempt in range(max_retries):
			max_tokens = 4000 + (attempt * 2000)
			try:
				retry_lm = dspy.LM(model=MODEL, max_tokens=max_tokens)
				with dspy.context(lm=retry_lm):
					res = await edge_extract.acall(text=chunk.raw_text)
				assert res.knowledge_segments[-1][0] == "All Done!" and res.entities[-1][0] == "All Done!", "Output is truncated"
				
				chunk.kb = res.knowledge_segments[:-1]
				chunk.e = res.entities[:-1]
				chunk.failed = False
				break
			except Exception as e:
				if attempt < max_retries - 1:
					logger.warning(f"Chunk {chunk.id} attempt {attempt + 1}/{max_retries} failed, retrying with max_tokens={4000 + ((attempt+1) * 2000)}...")
				else:
					logger.error(f"Chunk {chunk.id} failed after {max_retries} attempts: {e}")
					chunk.failed = True
					dspy.inspect_history(1)
		return chunk

async def extract_all_edges_entities(arg: list[str]| Path, sem_size: int = 16) -> list[Chunk]:
	# miner-style evaluation: input is a list of chunks from the original essay
	if isinstance(arg, list):
		chunks = [ Chunk(id=stable_id_hex(s), raw_text=s) for s in arg]
	# hgr-style evaluation: input is a preprocessed json file (chunked context)
	elif isinstance(arg, Path):
		assert arg.exists()
		with open(arg, "r") as f:
			chunks = [Chunk.model_validate(c) for c in json.load(f)]
			if all(c.kb and c.e for c in chunks):
				logger.info("Done edge and entity extraction for all chunks, using cache")
				return [kb for c in chunks for kb in c.kb], [e for c in chunks for e in c.e]
	else:
		raise ValueError("extract_all_edges_entities accepts Path or list[str]")
	
	sem = asyncio.Semaphore(sem_size)
	results: list[Chunk] = await tqdm_asyncio.gather(
		*[_extract_edges_entities(c, sem) for c in chunks]
	)

	# hgr-style evaluation: cache results
	if isinstance(input, Path):
		with open(input, "w") as f:
			json.dump([c.model_dump() for c in results], f)
			logger.info(f"Overwrote {input} with populated chunks")

	return [kb for c in results for kb in c.kb], [e for c in results for e in c.e]

# CREATE GRAPH: GIVEN KNOWLEDGE SEGMENTS AND ENTITIES, BUILD GRAPH -----------------------

async def aggregate(
	knowledge: list[tuple[str, int]], 
	entities: list[tuple[str, str, str, int]],
	cache_path: Path
):
	# if cache_path.exists():
	# 	logger.info("Graph already generated, using cache")
	# 	with open(cache_path, "r") as f:
	# 		data = json.load(f)
	# 	graph = nx.readwrite.node_link_graph(data)
	# 	return graph

	graph = nx.Graph()
	cur_idx = 0

	logger.info("Generating embeddings ...")
	
	k_to_embed = [k[0] for k in knowledge]
	e_to_embed = [e[2] for e in entities]

	# TODO: maybe embed in batches
	# TODO: cache embeddings
	k_resp, e_resp = await asyncio.gather(
		embed_call_with_retry(to_embed=k_to_embed),
		embed_call_with_retry(to_embed=e_to_embed),
	)

	k_embeddings = k_resp.data
	e_embeddings = e_resp.data
	assert len(k_embeddings) == len(k_to_embed), f"Expected {len(k_to_embed)} knowledge embeddings, got {len(k_embeddings)}"
	assert len(e_embeddings) == len(e_to_embed), f"Expected {len(e_to_embed)} entity embeddings, got {len(e_embeddings)}"

	logger.info("Adding entities to graph...")
	entities_indices = []
	for (e, e_t, e_d, e_s), e_em in zip(entities, e_embeddings):
		graph.add_node(cur_idx)
		entities_indices.append(cur_idx)
		graph.nodes[cur_idx]["hgr_type"] = "entity"
		graph.nodes[cur_idx]["name"] = e
		graph.nodes[cur_idx]["type"] = e_t
		graph.nodes[cur_idx]["desc"] = e_d
		graph.nodes[cur_idx]["score"] = e_s
		graph.nodes[cur_idx]["embedding"] = list(e_em.embedding)
		cur_idx += 1

	logger.info("Adding edges to graph...")
	he_indices = []
	for (k, k_score), k_em in zip(knowledge, k_embeddings):
		graph.add_node(cur_idx)
		he_indices.append(cur_idx)
		graph.nodes[cur_idx]["hgr_type"] = "hyperedge"
		graph.nodes[cur_idx]["text"] = k
		graph.nodes[cur_idx]["score"] = k_score
		graph.nodes[cur_idx]["embedding"] = list(k_em.embedding)

		for (e, _, _, _), e_i in zip(entities, entities_indices):
			if e.lower() in k.lower():
				graph.add_edge(cur_idx, e_i)
		
		cur_idx += 1

	# save to cache
	results = nx.readwrite.node_link_data(graph)
	with open(cache_path, "w") as f:
		json.dump(results, f)

	return graph

class EntityExtractSignature(dspy.Signature):
	""" Given the query, list all entities. """
	query: str = dspy.InputField()
	entities: list[str] = dspy.OutputField()

entity_extract = dspy.ChainOfThought(EntityExtractSignature)

async def extract_entities(text: str) -> list[str]:
	return (await entity_extract.acall(query=text)).entities

# QUERY GRAPH ------------------------------------------------------------------

async def query(
	query: str, 
	hg,
	# From HGR section 5.1
	kv = 60, # Max entities 
	tv = 50, # Threshold for entities 
	kh = 60, # Max hyperedges 
	th = 5, # Threshold for hyperedges 
):
	# get nodes
	i_entities = np.array([i for i, d in hg.nodes.items() if d["hgr_type"] == "entity"])
	i_edges = np.array([i for i, d in hg.nodes.items() if d["hgr_type"] == "hyperedge"])
	logger.debug(f"Found {len(i_entities)} entities in graph")
	logger.debug(f"Found {len(i_edges)} hyperedges in graph")
	assert set(i_edges).isdisjoint(set(i_entities)) # No overlap between them 

	# get embeddings
	e_entities = np.array([hg.nodes[i]["embedding"] for i in i_entities])
	e_edges = np.array([hg.nodes[i]["embedding"] for i in i_edges])

	# embed prompt
	logger.debug("Embedding query...")
	resp = await embed_call_with_retry(to_embed=[query])
	q_em = np.array(resp.data[0].embedding)

	# get similar entities
	cosine = lambda a, B: np.abs(np.dot(B, a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a)))
	e_sims = cosine(q_em, e_entities)
	e_sims = e_sims * np.array([hg.nodes[i]["score"] for i in i_entities]) # multiply by score

	# cut off >= tv
	e_cutoff = np.nonzero(e_sims > tv)[0] 
	e_sims = e_sims[e_cutoff] 

	# use top kv
	e_selection = i_entities[e_cutoff][np.argsort(e_sims)[::-1][:kv]]
	logger.debug(f"Selected {len(e_selection)} relevant entities")

	# get HE set 
	he_sims = cosine(q_em, e_edges)
	
	he_sims = he_sims * np.array([hg.nodes[i]["score"] for i in i_edges]) # multiply by score 
	
	# cut off less than or eq th
	he_cutoff = np.nonzero(he_sims > th)[0]
	he_sims = he_sims[he_cutoff] 
	
	# use top kh
	he_selection = i_edges[he_cutoff][np.argsort(he_sims)[::-1][:kh]]
	logger.debug(f"Selected {len(he_selection)} relevant hyperedges")

	# fusion step
	logger.debug("e", list(e_selection))
	logger.debug("h", list(he_selection))
	assert set(e_selection).isdisjoint(set(he_selection))
	node_set = set(list(e_selection) + list(he_selection))
	logger.debug(f"Have {len(node_set)} information pieces")
	for e in e_selection:
		for i in hg.neighbors(e):
			node_set.add(i)
	for he in he_selection:
		for i in hg.neighbors(he):
			node_set.add(i)
	logger.debug(f"Fusion expand to {len(node_set)} information pieces")
	assert len(node_set) >= len(e_selection) + len(he_selection)
	
	# pull textual information
	k = ""
	for i in node_set:
		if hg.nodes[i]["hgr_type"] == "entity":
			k += f"{hg.nodes[i]["name"]} - {hg.nodes[i]["desc"]}\n"
		elif hg.nodes[i]["hgr_type"] == "hyperedge":
			k += hg.nodes[i]["text"] + "\n"
		else:
			logger.error(f"Something is wrong, '{hg.nodes[i]["hgr_type"]}' is an invalid hgr type")
			exit(1)
	return k

# async def main():
# 	model = "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"
# 	embedding_model = "bedrock/amazon.titan-embed-text-v2:0"
# 	chunks = [
# 		"Time zones have played a crucial role in shaping the way we perceive and organize time across the world. The concept of time zones has evolved over centuries, reflecting the need to standardize time measurement and synchronize activities across vast distances. This essay provides a brief history of time zones, tracing their origins and development into the system we use today.\n\nThe need for time zones arose with the expansion of global trade and transportation networks. Before the advent of standardized time measurement, communities used local solar time based on the position of the sun in the sky. This worked well for localized activities but posed challenges as transportation and communication capabilities improved. Trains and telegraphs, in particular, highlighted the inefficiencies of using local time across long distances.\n\nThe first major development in timekeeping came with the introduction of standard time in the mid-19th century. Before this, each town or city would set its own time based on the local position of the sun. However, this system became increasingly impractical as rail travel became more widespread. In 1847, the British railway companies adopted a single standard time for their schedules, known as Railway Time. This marked the beginning of standardized timekeeping on a regional level.\n\nThe concept of time zones as we know them today was first proposed by Sir Sandford Fleming, a Canadian engineer, in the late 19th century. Fleming suggested dividing the world into 24 time zones, each one representing a one-hour difference in mean solar time. This idea gained traction at the International Meridian Conference held in Washington, D.C. in 1884, where it was agreed to adopt a system of standard time zones based on Greenwich Mean Time (GMT).\n\nThe Greenwich Meridian, located in London, was chosen as the Prime Meridian from which all time zones would be calculated. This decision established GMT as the reference point for coordinating time across the globe. The conference also divided the world into 24 time zones, each spanning 15 degrees of longitude. This system allowed for a more precise and standardized way of measuring time, facilitating global communication and coordination.\n\nFollowing the adoption of standard time zones at the International Meridian Conference, countries around the world began to implement the new system. The United States, for example, divided the country into four time zones: Eastern, Central, Mountain, and Pacific. Other countries followed suit, adjusting their timekeeping practices to align with the new standard.\n\nThe development of time zones was further refined with the introduction of Daylight Saving Time (DST) in the early 20th century. DST is the practice of advancing clocks by one hour during the warmer months to extend daylight hours in the evening. This was first implemented during World War I as a way to conserve energy and make better use of natural daylight. DST has since become a common practice in many countries, with the dates of the time changes varying depending on local regulations.\n\nIn recent years, advances in technology and communication have made coordinating time across different time zones easier than ever. The internet and global telecommunications networks have enabled instant communication and real-time collaboration across vast distances. However, the basic principles of time zones remain essential for organizing activities on a global scale and ensuring that everyone is on the same page when it comes to scheduling events and meetings.\n\nIn conclusion, the history of time zones is a testament to human ingenuity and our ability to adapt to the changing needs of a globalized world. From the early days of local solar time to the standardized system of time zones we use today, the evolution of timekeeping has been driven by the need for efficiency and coordination across vast distances. Time zones have become an integral part of our daily lives, shaping how we interact with each other and organize our activities in a world that is constantly connected and in motion.",
# 	]
# 	q = "Communities used local solar time before standardized time measurement."

# 	dspy.configure(lm=dspy.LM(model))

# 	kb = []
# 	eb = []
# 	for i, chunk in enumerate(chunks):
# 		print(f"Extract {i+1}/{len(chunks)}")
# 		try:
# 			k, e = await _extract_edges_entities(chunk)	
# 		except Exception as e:
# 			print(e)
# 			dspy.inspect_history(2)
# 			raise e
# 		print("Fragments:")
# 		for v in k:
# 			print(v)
# 		print("Entities:")
# 		for v in e:
# 			print(v)
# 		kb += k
# 		eb += e
# 	hg = await aggregate(kb, eb, embedding_model)

# 	k = await query(q, hg, embedding_model)
# 	print("Done!")
# 	print(k)
# 	print("^^^ that's all!")
	


# if __name__ == "__main__":
# 	asyncio.run(main())