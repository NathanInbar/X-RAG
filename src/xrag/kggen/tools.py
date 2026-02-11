import dspy 
import networkx as nx
import litellm
import numpy as np
import unicodedata
import inflect 
from semhash import SemHash

class EntityExtractSignature(dspy.Signature):
	"""
	Extract key entities from the source text. 
	Extracted entities are subjects or objects.
	This is for an extraction task, please be thorough and accurate to the reference text.
	"""
	text: str = dspy.InputField()
	entities: list[str] = dspy.OutputField()
entity_extract = dspy.Predict(EntityExtractSignature)


class RelationshipExtractSignature(dspy.Signature):
	"""
	Extract subject-predicate-object triples from the source text. 
	Subjects adn objects must be from the entities list. 
	Entities provided were previously extacted from the same source text. 
	This is for an extraction task, please be thorough, accurate, and faithful to the reference text.
	"""
	text: str = dspy.InputField()
	entities: list[str] = dspy.InputField()
	relationships: list[tuple[str, str, str]] = dspy.OutputField()
relationship_extract = dspy.Predict(RelationshipExtractSignature)


class ItemResolutionSignature(dspy.Signature):
	"""
	Find duplicate [idk] for the item and an alias that best represents the duplicates. 
	Duplicates ate those that are the same in meaning, such as with variation in tense, plural form, stem form, case, abbreviation, shorthand. 
	Return an empty list if these are none. 
	"""
	text: str = dspy.InputField()
	entities: list[str] = dspy.InputField()
	relationships: list[tuple[str, str, str]] = dspy.OutputField()
relationship_extract = dspy.Predict(RelationshipExtractSignature)


async def extract(text: str) -> tuple[list[str], list[tuple[str, str, str]]]:
	entities = (await entity_extract.acall(text=text)).entities 

	relationships = (await relationship_extract.acall(text=text, entities=entities)).relationships 

	# Check for outsiders 
	r_ents = [v for a, _, b in relationships for v in [a, b]]
	unpresent = []
	for e in r_ents:
		if not (e in entities):
			unpresent.append(e)
	# assert len(unpresent) == 0, f"Oh that's not great {unpresent}"
	entities += unpresent

	return entities, relationships


def singularize(text: str) -> str:
	tokens = []
	for tok in text.split():
		sing = inflect.engine().singular_noun(tok)
		tokens.append(sing if isinstance(sing, str) and sing else tok)
	return " ".join(tokens).strip()


async def resolve(
	entities: list[str], 
	relationships: list[tuple[str, str, str]], 
	threshold=0.95,
):
	# I am aware that this code is bad
	# But like... yeah 

	# Entities! 
	# Normalize 
	ents = [unicodedata.normalize("NFKC", e) for e in entities]
	# Singularize 
	ents = [singularize(e) for e in ents]
	# Make a map 
	ent_map = {e0: e1 for e0, e1 in zip(entities, ents)}
	# Dedup 
	semhash = SemHash.from_records(records=ents)
	deduplication_result = semhash.self_deduplicate(threshold=threshold)
	# Update the map 
	for rec in deduplication_result.filtered:
		for dup, _ in rec.duplicates:
			# print(f"'{dup}' -> '{rec.record}'")
			ent_map[dup] = rec.record

	# Relationships! 
	rels = [r for _, r, _ in relationships]
	rels = [unicodedata.normalize("NFKC", r) for r in rels]
	# Singularize 
	rels = [singularize(r) for r in rels]
	# Map 
	rel_map = {r0: r1 for (_, r0, _), r1 in zip(relationships, rels)}
	# Dedup 
	semhash = SemHash.from_records(records=rels)
	deduplication_result = semhash.self_deduplicate(threshold=threshold)
	# Update the map 
	for rec in deduplication_result.filtered:
		for dup, _ in rec.duplicates:
			# print(f"'{dup}' -> '{rec.record}'")
			rel_map[dup] = rec.record

	# Find occurrences and replace 
	ents = [ent_map[e] for e in entities]
	relationships = [(ent_map[a], rel_map[r], ent_map[b]) for a, r, b in relationships]
	# Reduce
	ents = list(set(ents))
	# print(f"Dedup entities {len(entities)} -> {len(ents)}")
	rels = list(set(relationships))
	# print(f"Dedup relationships {len(relationships)} -> {len(rels)}")
	
	return ents, rels


async def make_graph(
	entities: list[str], 
	relationships: list[tuple[str, str, str]], 
	embedding_model: str,
):
	graph = nx.MultiDiGraph()
	cur_idx = 0

	# print("Generate embeddings")
	embed_knowledge_input = [" ".join(t) for t in relationships]
	embed_entities_input = entities
	embeddings = await litellm.aembedding(model=embedding_model, input=embed_knowledge_input+embed_entities_input)
	embed_knowledge = embeddings.data[:len(embed_knowledge_input)]
	embed_entities = embeddings.data[len(embed_knowledge_input):]

	# print("Add entities")
	entities_indices = {}
	for e, e_em in zip(entities, embed_entities):
		if not (e in entities_indices.keys()):
			graph.add_node(cur_idx)
			entities_indices[e] = cur_idx
			graph.nodes[cur_idx]["name"] = e
			graph.nodes[cur_idx]["embedding"] = np.array(e_em.embedding)
			cur_idx += 1

	# print("Add edges")
	# print(entities)
	for (a, r, b), k_em in zip(relationships, embed_knowledge):
		a_i = entities_indices[a]
		b_i = entities_indices[b]
		graph.add_edge(a_i, b_i, data={
			"relationship": r,
			"embedding": np.array(k_em.embedding),
		})

	return graph


async def query(
	query: str, 
	kg,
	embedding_model: str,
	k=10,
	k2=10,
	hops=2,
) -> list[tuple[str, str, str]]:
	edges = np.array(list(kg.edges.data()))
	embeddings = np.array([d["data"]["embedding"] for _, _, d in edges])

	q_em = np.array((await litellm.aembedding(model=embedding_model, input=[query])).data[0].embedding)

	cosine = lambda a, B: np.abs(np.dot(B, a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a)))

	e_sims = cosine(q_em, embeddings)
	# Top k similar relationships 
	k1_selection = edges[np.argsort(e_sims)[::-1][:k]]
	k1_selection = [(a, b, d["data"]["relationship"]) for a, b, d in k1_selection]

	# Top k 2-hop neighbours to those
	# This is messy but it will work 
	k2_entities = set([v for a, b, _ in k1_selection for v in [a, b]])
	for _ in range(0, hops):
		expanded = set(k2_entities)
		for i in expanded:
			# Can't use neighbours because this is directed! 
			# for n in kg.neighbors(i):
			# 	k2_entities.add(n)
			# Instead use in/out edges 
			for (n, _) in kg.in_edges(i):
				k2_entities.add(n)
			for (_, n) in kg.out_edges(i):
				k2_entities.add(n)
	# Fetch subgraph of expanded neighbours 
	kg_sub = nx.MultiDiGraph()
	kg_sub.add_nodes_from((n, kg.nodes[n]) for n in k2_entities)
	kg_sub.add_edges_from(
		(n, nbr, key, d)
		for n, nbrs in kg.adj.items()
		if n in k2_entities
		for nbr, kd in nbrs.items()
		if nbr in k2_entities
		for key, d in kd.items()
	)
	# Collect all edges (whcih aren't in the original selection)
	k2_edges = np.array(list([(a, b, d) for a, b, d in kg.edges.data() if not (a, b, d["data"]["relationship"]) in k1_selection]))
	k2_embeddings = np.array([d["data"]["embedding"] for _, _, d in k2_edges])
	k2_sims = cosine(q_em, k2_embeddings)
	k2_selection = k2_edges[np.argsort(k2_sims)[::-1][:k2]]
	k2_selection = [(a, b, d["data"]["relationship"]) for a, b, d in k2_selection]

	# print("k1", k1_selection)
	# print("k2", k2_selection)	
	selection = set(list(k1_selection)).union(set(list(k2_selection)))	

	assert set(list(k1_selection)).isdisjoint(set(list(k2_selection))), "Should be disjoint"
	assert len(selection) <= k+k2, "Wrong selection size"

	triples = list((kg.nodes[a]["name"], r, kg.nodes[b]["name"]) for a, b, r in selection) 

	return triples