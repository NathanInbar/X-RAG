import sys
from pathlib import Path
import json
from utils.models import EntityDescEmbed
import litellm
import numpy as np
from utils import mg_driver
from itertools import combinations
import ijson
from utils.signatures import generate_augmented_response

CWD = Path(__name__).resolve().parent
sys.path.append(CWD)

DATASET_FILE = CWD / "result.json"
secrets = CWD / "secrets.env"

EMBED_MODEL = "bedrock/amazon.titan-embed-text-v2:0"
MAX_TOP_CHUNKS = 5

if not secrets.is_file():
    raise ValueError(f"secrets file at '{secrets}' does not exist")

from dotenv import load_dotenv
load_dotenv(secrets)

import logging
logger = logging.getLogger("leanrag-retrieve")
logger.setLevel(logging.INFO)

async def _embed_single(s):
    resp = await litellm.aembedding(model=EMBED_MODEL, input=s)
    data = resp['data'][0]
    return data['embedding'] 

def search_dense(query_vec: list[float], entity_store: list[dict], topk: int= 10):
    if topk <= 0:
        return []

    # Convert to NumPy arrays
    query = np.asarray(query_vec, dtype=np.float32)
    entity_matrix = np.asarray(
        [item["desc_embed"] for item in entity_store], dtype=np.float32
    )

    # Normalize to use cosine similarity (handle zero vectors defensively)
    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        raise ValueError("query vector has zero norm")
    query /= q_norm

    e_norms = np.linalg.norm(entity_matrix, axis=1, keepdims=True)
    e_norms[e_norms == 0] = 1.0
    entity_matrix = entity_matrix / e_norms

    # Dot product gives cosine similarity
    sims = entity_matrix @ query

    # Grab top-k indices
    k = min(topk, len(entity_store))
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    return [
        {**entity_store[i], "score": float(sims[i])}
        for i in top_idx
    ]

async def get_response_context_data(user_query:str, chunks_file:Path) -> str:
    await mg_driver.init()

    embeddings_file = CWD / "g0_embeddings.json"
    with open(embeddings_file, "r") as ef:
        entity_embeddings = json.load(ef)
        
    # 1. embed the query
    prompt_embedding:list[float] = await _embed_single(user_query)
    
    # 2. find top-k seed entities (LeanRAG uses 10)
    seed_entities = search_dense(prompt_embedding, entity_embeddings, topk=10)
    logger.debug(f"found {len(seed_entities)} seed entities")

    ancestor_chains = []
    for s in seed_entities:
        try:
            chain = await mg_driver.get_ancestor_chain(s['key'])
        except Exception:
            continue
        ancestor_chains.append(chain)

    # 3. form lca paths from ancestor chains -- collect information for context data
    reasoning_path_information:dict[str,dict] = {}
    lca_encountered_agg_entities:dict[str, dict] = {}
    entity_parent_map:dict[str,str] = {}
    entity_info_map:dict[str,dict] = {}

    for s1, s2 in combinations(ancestor_chains, 2):
        zipped_chain = zip(s1,s2)
        lca_key:str = None
        for anc1, anc2 in zipped_chain:
            if anc1['key'] == anc2['key']:
                lca_key = anc1['key']
                break
        if not lca_key:
            raise RuntimeError(f"could not find lca for '{s1[0]['key']}' TO '{s2[0]['key']}'")

        # find path from s1 -> lca -> s2
        lca_path = await mg_driver.get_lca_path(s1[0]['key'], s2[0]['key'], lca_key)
        
        # LCA PATH GOES entity_1 -> agg_1 -> ... -> root/aggX -> ... -> agg_1 -> entity_2
        # should always be at least 3 nodes long
        if lca_path[0]['key'] not in entity_parent_map.keys():
            entity_parent_map[lca_path[0]['key']] = lca_path[1] # parent of entity 1
            entity_info_map[lca_path[0]['key']] = lca_path[0]
        if lca_path[-1]['key'] not in entity_parent_map.keys():
            entity_parent_map[lca_path[-1]['key']] = lca_path[-2] # parent of entity 2
            entity_info_map[lca_path[-1]['key']] = lca_path[-1]

        for i, ent in enumerate(lca_path):
            # populate the encountered aggregate entities dict
            if ent['key'] == 'root': # only want to collect seen aggregate nodes , ignore root
                continue
            if ent['key'] in lca_encountered_agg_entities.keys():
                continue
            lca_encountered_agg_entities[ent['key']] = ent

        # collect horizontal intra-relations between nodes in the path
        connection_edges = await mg_driver.get_intra_lca_path_links(s1[0]['key'], s2[0]['key'], lca_key)
        if len(connection_edges) > 0:
            for ce in connection_edges:
                if ce['r_key'] in reasoning_path_information.keys():
                    continue
                reasoning_path_information[ce['r_key']] = ce

    # find the most used chunks among seed entities- refine down to MAX_TOP_CHUNKS chunks
    top_chunks = await mg_driver.get_ranked_provenance_for_entitys([e['key'] for e in seed_entities])
    top_chunks = set([c['chunk_id'] for c in top_chunks[:MAX_TOP_CHUNKS]])
    top_chunks_text = []
    with open(chunks_file, "rb") as in_file:
        for itm in ijson.items(in_file, "item"):
            for chunk in itm['chunks']:
                if chunk['id'] in top_chunks:
                    top_chunks_text.append(chunk['raw_text'])
    top_chunks_text = "\n".join(top_chunks_text)

    context_ent_info = [(entity_info_map[e]['name'], entity_parent_map[e]['name'], entity_info_map[e]['desc']) for e in entity_info_map]
    context_rpath_info = [ vv['r_desc'] for vv in [v for _,v in reasoning_path_information.items()]]
    context_agg_ent_info = [(vv['name'], vv['desc']) for vv in [v for v in lca_encountered_agg_entities.values()]]

    # convert to string tables:
    bei_string, aei_string, rpi_string = [],[],[]
    bei_string.append("entity name, parent, description")# header
    for row in context_ent_info:
        bei_string.append(f"{row[0], row[1], row[2]}")
    bei_string = "\n".join(bei_string)

    aei_string.append("entity name, entity description")# header
    for row in context_agg_ent_info:
        aei_string.append(f"{row[0], row[1]}")
    aei_string = "\n".join(aei_string)

    for row in context_rpath_info:
        rpi_string.append(row)
    rpi_string = "\n".join(rpi_string)

    return bei_string, aei_string, rpi_string, top_chunks_text

async def get_response(user_query:str) -> str:
    bei_string, aei_string, rpi_string, top_chunks_string = await get_response_context_data(user_query)

    response = await generate_augmented_response(
        query = user_query,
        base_entity_info= bei_string,
        agg_entity_info= aei_string,
        reasoning_path_info= rpi_string,
        relevant_chunk_texts = top_chunks_string
    )

    return response