import asyncio
# import litellm
import dspy
import networkx as nx
import numpy as np
from xrag.dataset_processing.preprocess import embed_call_with_retry
from xrag.utils import batched
from tqdm import tqdm
import math

# This is prone to errors due to models not following the specification
# We could try the means of this in HGR but I would rather not 
class EdgeExtractSignature(dspy.Signature):
	"""
	Given a text document that is potentially relevant to this activity and a list of entitiy types, identify all entities of theose types from the text and all relationships among the identified entities. 

	# Steps

	1. Divide the text into several complete knowledge segments. For each knowledge segment, extract the following information: 
	- knowledge segment: A sentence that describes the context of the knowledge segment. 
	- completeness score: A score form 0 to 10 indicating the completeness of the knowledge segment. 

	2. Identify all entities in each knowledge segment. For each identified entity, extract the following information: 
	- entity name: Name of the entity. 
	- entity type: Type of the entity. 
	- entity description: Comprehensive description of the entity's attributes and activities. 
	- key score: A score from 0 to 100 indicating the importance of the entity in the text. 
	"""

	text: str = dspy.InputField()
	knowledge_segments: list[tuple[str, int]] = dspy.OutputField(desc="a list of knowledge fragments and their completeness scores")
	entities: list[tuple[str, str, str, int]] = dspy.OutputField(desc="a list of entity names, types, descriptions, and scores")
edge_extract = dspy.ChainOfThought(EdgeExtractSignature)
# edge_extract = dspy.Predict(EdgeExtractSignature)


async def extract_edges_entities(text: str):
	res = await edge_extract.acall(text=text)
	return res.knowledge_segments, res.entities


async def aggregate(
	knowledge: list[tuple[str, int]], 
	entities: list[tuple[str, str, str, int]],
	embedding_model: str,
):
	# print("Aggregate into graph")
	graph = nx.Graph()
	cur_idx = 0

	# print("Generate embeddings")
	embed_knowledge_input = [k[0] for k in knowledge]
	embed_entities_input = [e[2] for e in entities] # Embedding of name or desc? Let's try desc

	batches_to_embed = batched(embed_knowledge_input+embed_entities_input, 100)
	total_batches = math.ceil(len(embed_knowledge_input+embed_entities_input) / 100)
	embeddings = []
	for batch in tqdm(batches_to_embed, total=total_batches):
		resp = await embed_call_with_retry(to_embed=batch)
		embeddings.extend(resp.data)
	# embeddings = litellm.embedding(model=embedding_model, input=embed_knowledge_input+embed_entities_input)
	embed_knowledge = embeddings[:len(embed_knowledge_input)]
	embed_entities = embeddings[len(embed_knowledge_input):]

	# print("Add entities")	
	entities_indices = []
	for (e, e_t, e_d, e_s), e_em in zip(entities, embed_entities):
		# Cannot index by name becuase we need to support having multiple with same name s
		graph.add_node(cur_idx)
		entities_indices.append(cur_idx)
		graph.nodes[cur_idx]["hgr_type"] = "entity"
		graph.nodes[cur_idx]["name"] = e
		graph.nodes[cur_idx]["type"] = e_t
		graph.nodes[cur_idx]["desc"] = e_d
		graph.nodes[cur_idx]["score"] = e_s
		graph.nodes[cur_idx]["embedding"] = np.array(e_em.embedding)
		cur_idx += 1


	# print("Add edges")
	he_indices = []
	for (k, k_score), k_em in zip(knowledge, embed_knowledge):
		graph.add_node(cur_idx)
		he_indices.append(cur_idx)
		graph.nodes[cur_idx]["hgr_type"] = "hyperedge"
		graph.nodes[cur_idx]["text"] = k
		graph.nodes[cur_idx]["score"] = k_score
		graph.nodes[cur_idx]["embedding"] = np.array(k_em.embedding)

		for (e, _, _, _), e_i in zip(entities, entities_indices):
			if e.lower() in k.lower():
				graph.add_edge(cur_idx, e_i)
		
		cur_idx += 1

	return graph


class EntityExtractSignature(dspy.Signature):
	""" Given the query, list all entities. """
	query: str = dspy.InputField()
	entities: list[str] = dspy.OutputField()
entity_extract = dspy.ChainOfThought(EntityExtractSignature)


async def extract_entities(text: str) -> list[str]:
	return (await entity_extract.acall(query=text)).entities


async def query(
	query: str, 
	hg,
	embedding_model: str,
	# From HGR section 5.1
	kv = 60, # Max entities 
	tv = 50, # Threshold for entities 
	kh = 60, # Max hyperedges 
	th = 5, # Threshold for hyperedges 
):
	i_entities = np.array([i for i, d in hg.nodes.items() if d["hgr_type"] == "entity"])
	i_edges = np.array([i for i, d in hg.nodes.items() if d["hgr_type"] == "hyperedge"])
	# print(f"Found {len(i_entities)} entities")
	# print(f"Found {len(i_edges)} hyperedges")
	assert set(i_edges).isdisjoint(set(i_entities)) # No overlap between them 

	e_entities = np.array([hg.nodes[i]["embedding"] for i in i_entities])
	e_edges = np.array([hg.nodes[i]["embedding"] for i in i_edges])

	# Get entities from prompt 
	# q_entities = await extract_entities(query)
	# Make embedding
	# Using json as input seems questionable but it's what the paper describes so ehh
	# q_em_input = json.dumps(q_entities)
	# Nvm it's being really picky
	# q_em_input = ", ".join(q_entities)
	# That's not great so I will try this actually
	q_em_input = query
	q_em = np.array((await embed_call_with_retry(to_embed=q_em_input)).data[0].embedding)
	# q_em = np.array((await litellm.aembedding(model=embedding_model, input=q_em_input)).data[0].embedding)

	cosine = lambda a, B: np.abs(np.dot(B, a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a)))

	# Get entity set 
	e_sims = cosine(q_em, e_entities)
	# Multiply by score
	e_sims = e_sims * np.array([hg.nodes[i]["score"] for i in i_entities])
	# Cut off less than or eq tv
	e_cutoff = np.nonzero(e_sims > tv)[0]
	e_sims = e_sims[e_cutoff] 
	# Use top kv of them
	e_selection = i_entities[e_cutoff][np.argsort(e_sims)[::-1][:kv]]
	# print(f"Selected {len(e_selection)} entities")

	# Get HE set 
	he_sims = cosine(q_em, e_edges)
	# Multiply by score 
	he_sims = he_sims * np.array([hg.nodes[i]["score"] for i in i_edges])
	# Cut off less than or eq th
	he_cutoff = np.nonzero(he_sims > th)[0]
	he_sims = he_sims[he_cutoff] 
	# Use top kh of them
	he_selection = i_edges[he_cutoff][np.argsort(he_sims)[::-1][:kh]]
	# print(f"Selected {len(he_selection)} hyperedges")

	# Fusion step
	# print("e", list(e_selection))
	# print("h", list(he_selection))
	assert set(e_selection).isdisjoint(set(he_selection)) # No overlap between them 
	node_set = set(list(e_selection) + list(he_selection))
	# print(f"Have {len(node_set)} information pieces")
	for e in e_selection:
		for i in hg.neighbors(e):
			node_set.add(i)
	for he in he_selection:
		for i in hg.neighbors(he):
			node_set.add(i)
	# print(f"Fusion expand to {len(node_set)} information pieces")
	assert len(node_set) >= len(e_selection) + len(he_selection)
	
	# Pull textual information
	k = ""
	for i in node_set:
		if hg.nodes[i]["hgr_type"] == "entity":
			k += f"{hg.nodes[i]["name"]} - {hg.nodes[i]["desc"]}\n"
		elif hg.nodes[i]["hgr_type"] == "hyperedge":
			k += hg.nodes[i]["text"] + "\n"
		else:
			print(f"Something is wrong, '{hg.nodes[i]["hgr_type"]}' is an invalid hgr type")
			exit(1)
	return k


async def main():
	model = "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"
	embedding_model = "bedrock/amazon.titan-embed-text-v2:0"
	chunks = [
		"Time zones have played a crucial role in shaping the way we perceive and organize time across the world. The concept of time zones has evolved over centuries, reflecting the need to standardize time measurement and synchronize activities across vast distances. This essay provides a brief history of time zones, tracing their origins and development into the system we use today.\n\nThe need for time zones arose with the expansion of global trade and transportation networks. Before the advent of standardized time measurement, communities used local solar time based on the position of the sun in the sky. This worked well for localized activities but posed challenges as transportation and communication capabilities improved. Trains and telegraphs, in particular, highlighted the inefficiencies of using local time across long distances.\n\nThe first major development in timekeeping came with the introduction of standard time in the mid-19th century. Before this, each town or city would set its own time based on the local position of the sun. However, this system became increasingly impractical as rail travel became more widespread. In 1847, the British railway companies adopted a single standard time for their schedules, known as Railway Time. This marked the beginning of standardized timekeeping on a regional level.\n\nThe concept of time zones as we know them today was first proposed by Sir Sandford Fleming, a Canadian engineer, in the late 19th century. Fleming suggested dividing the world into 24 time zones, each one representing a one-hour difference in mean solar time. This idea gained traction at the International Meridian Conference held in Washington, D.C. in 1884, where it was agreed to adopt a system of standard time zones based on Greenwich Mean Time (GMT).\n\nThe Greenwich Meridian, located in London, was chosen as the Prime Meridian from which all time zones would be calculated. This decision established GMT as the reference point for coordinating time across the globe. The conference also divided the world into 24 time zones, each spanning 15 degrees of longitude. This system allowed for a more precise and standardized way of measuring time, facilitating global communication and coordination.\n\nFollowing the adoption of standard time zones at the International Meridian Conference, countries around the world began to implement the new system. The United States, for example, divided the country into four time zones: Eastern, Central, Mountain, and Pacific. Other countries followed suit, adjusting their timekeeping practices to align with the new standard.\n\nThe development of time zones was further refined with the introduction of Daylight Saving Time (DST) in the early 20th century. DST is the practice of advancing clocks by one hour during the warmer months to extend daylight hours in the evening. This was first implemented during World War I as a way to conserve energy and make better use of natural daylight. DST has since become a common practice in many countries, with the dates of the time changes varying depending on local regulations.\n\nIn recent years, advances in technology and communication have made coordinating time across different time zones easier than ever. The internet and global telecommunications networks have enabled instant communication and real-time collaboration across vast distances. However, the basic principles of time zones remain essential for organizing activities on a global scale and ensuring that everyone is on the same page when it comes to scheduling events and meetings.\n\nIn conclusion, the history of time zones is a testament to human ingenuity and our ability to adapt to the changing needs of a globalized world. From the early days of local solar time to the standardized system of time zones we use today, the evolution of timekeeping has been driven by the need for efficiency and coordination across vast distances. Time zones have become an integral part of our daily lives, shaping how we interact with each other and organize our activities in a world that is constantly connected and in motion.",
	]
	q = "Communities used local solar time before standardized time measurement."

	dspy.configure(lm=dspy.LM(model))

	kb = []
	eb = []
	for i, chunk in enumerate(chunks):
		print(f"Extract {i+1}/{len(chunks)}")
		try:
			k, e = await extract_edges_entities(chunk)	
		except Exception as e:
			print(e)
			dspy.inspect_history(2)
			raise e
		print("Fragments:")
		for v in k:
			print(v)
		print("Entities:")
		for v in e:
			print(v)
		kb += k
		eb += e
	hg = await aggregate(kb, eb, embedding_model)

	k = await query(q, hg, embedding_model)
	print("Done!")
	print(k)
	print("^^^ that's all!")
	


if __name__ == "__main__":
	asyncio.run(main())