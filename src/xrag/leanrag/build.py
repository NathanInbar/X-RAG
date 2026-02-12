import json
import random
import asyncio
from math import log
from itertools import combinations
import logging
import numpy as np
import litellm
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

from xrag.utils import (
    mg_driver, signatures,
    Tokenizer,
    batched, reduce_embeddings,
    get_optimal_clusters_from_embeddings,
)
from xrag.utils.models import *
from xrag.config import config
from xrag.paths import CACHE_DIR
#TODO: generate + add entity 'type' property

EMBED_MODEL = config.models["embed"]
G0_EMBEDDINGS_FILE = CACHE_DIR / "g0_embeddings.json"
ENTITY_BATCH_SIZE = config.leanrag["entity_batch_size"]
MAX_PARALLEL_EMBED = config.llm_concurrency["embed"]
CLUSTER_SIZE = config.leanrag["cluster_size"]
INITIAL_DELAY = config.llm_retry["initial_delay"]
MAX_ATTEMPTS = config.llm_retry["max_attempts"]

INITIAL_DELAY = config.llm_retry["initial_delay"]
MAX_ATTEMPTS = config.llm_retry["max_attempts"]

# DEBUG_LAYER_START = 0
# DEBUG_LAYER_STOP = 1

type AggEntityKey = str

# backing data stored outside of graph
findings_map: dict[AggEntityKey, list[Finding]] = None
fmap_lock:asyncio.Lock = None

agg_clst_map:dict[AggEntityKey, Cluster] = {}
acm_lock = asyncio.Lock()

root_layer:int = -1

# LOGGER
logging.basicConfig(
    level=logging.WARN,
    format="%(levelname)s - %(message)s",
    force=True
)

logger = logging.getLogger("leanrag-build")
logger.setLevel(logging.INFO)

async def batch_embed_descriptions(batch:list[Entity|AggEntity], acc:AsyncList, embed_sem:asyncio.Semaphore, pbar:AsyncProgressBar) -> None:
    """ embed a single batch of entity descriptions """
    
    retry_delay = INITIAL_DELAY
    for attempt in range(1,MAX_ATTEMPTS+1):
        try:
            async with embed_sem:
                resp = await litellm.aembedding(model=EMBED_MODEL, input=[e['desc'] for e in batch])
            break
        except Exception as e:
            if attempt == MAX_ATTEMPTS:
                error_message = f"Max retry attempts reached. Skipping {len(batch)} descriptions: {e}"
                logger.error(error_message)
                raise RuntimeError(error_message)
            logger.error(f"Embed attempt {attempt} failed for {len(batch)} descriptions. Retrying in {retry_delay}s: {e}")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2
            retry_delay += random.uniform(0, 1)

    batch_embed = resp['data']
    # unpack batch to entity_key -> description pairs
    rows = [
        EntityDescEmbed(key=batch[emb.index]["key"],desc_embed=emb.embedding)
        for emb in sorted(batch_embed, key=lambda e: e.index)
    ]
    await acc.extend(rows)
    await pbar.update(len(batch))

async def embed_all_entity_descriptions(entities:list[Entity], batch_size:int, max_parallel:int, layer:int, pbar:AsyncProgressBar) -> AsyncList[EntityDescEmbed]:
    """
    embed all entity descriptions in batches
    """
    # logger.debug("ENTITIES TO EMBED:\n")
    # logger.debug(str(entities))
    # raise RuntimeError()

    acc = AsyncList()

    embed_sem = asyncio.Semaphore(max_parallel)

    batch_embed_tasks=[]
    for batch in batched(entities, batch_size):
        batch_embed_tasks.append(batch_embed_descriptions(batch, acc, embed_sem, pbar))

    results = await asyncio.gather(*batch_embed_tasks, return_exceptions=True)
    for r in results:
        if r:
            logger.error(f"embed task failure: {r}")

    if layer == 0:
        # write out the entity embeddings (used in retrieval step)
        with open(G0_EMBEDDINGS_FILE, "w") as fp:
            json.dump(acc.get_list(), fp)

    return acc

async def build_aggregate_entity(cluster: Cluster, layer) -> tuple[AggEntity | None, list[Finding] | None]:
    input_rows: list[str] = []
    input_rows.append("ENTITIES: entity_name, entity_description, entity_degree")
    for i, entity in enumerate(cluster):
        input_rows.append(f"{i}: {entity['name']}, {entity['desc']}, {entity['degree']}")
    input_rows.append("")

    intra_rels = await mg_driver.get_intra_cluster_relations(cluster)
    input_rows.append("RELATIONS: source_entity, target_entity, relation_description")
    for i, rel in enumerate(intra_rels):
        input_rows.append(f"{i}: {rel['source_entity']}, {rel['target_entity']}, {rel['relation_description']}")

    input_text = "\n".join(input_rows)
    agg_ent, findings = await signatures.generate_aggregate_node(input_text=input_text)
    agg_ent['key'] = f"{layer}__{agg_ent['key']}"
    
    return agg_ent, findings

async def _aggregate_task(cluster: Cluster, out_aggregates:AsyncList, layer, pbar):
    """
    create new aggregate entity from cluster
    - store it in 'aggregates' list.
    - map new agg entity -> findings list in the findings map
    - map new agg entity -> child entity list (cluster) in agg_clst_map
    """
    new_parent, new_findings = await build_aggregate_entity(cluster, layer)
    if not (new_parent and new_findings):
        print("ERROR! Got none for new parent or new findings")
        await pbar.update(1)
        return

    key: AggEntityKey = new_parent["key"]

    async with fmap_lock:
        async with acm_lock:
            if key in findings_map:
                # duplicate aggregate entity generated: merge into original
                findings_map[key].extend(new_findings)
                agg_clst_map[key].extend(cluster)
                is_dup = True
            else:
                # first occurrence
                findings_map[key] = list(new_findings)
                agg_clst_map[key] = list(cluster)
                is_dup = False

    # only keep one aggregate node per key
    if not is_dup:
        await out_aggregates.append(new_parent)
    await pbar.update(1)

def icr_desc_fallback_concat(inter_cluster_relations:list[IntrClusterRel]) -> str:
    """ LeanRAG F(rel) when connectivity strength is below threshold (tau): use simple concat of inter-cluster-relations """
    return "\n".join([f"relationship<|>{r['source_entity']}<|>{r['target_entity']}<|>{r['relation_description']}" for r in inter_cluster_relations])

async def build_hierarchy(max_depth:int):
    global findings_map, fmap_lock
    global agg_clst_map, acm_lock

    findings_map = {}
    fmap_lock = asyncio.Lock()

    agg_clst_map = {}
    acm_lock = asyncio.Lock()

    await aggregate_layer_recursive(0, max_depth)

async def aggregate_layer_recursive(layer:int, max_depth:int):
    global root_layer
    logger.info(f"Aggregating Layer {layer} ...")
    #debug
    # if DEBUG_LAYER_STOP > 0 and layer == DEBUG_LAYER_STOP:
    #     root_layer = layer
    #     logger.info(f"\tSTOP!: stopped at this layer due to DEBUG_LAYER_STOP flag")
    #     return

    if layer > max_depth:
        root_layer = layer
        logger.info(f"\tSTOP!: stopped at this layer, hit max depth of {max_depth}")
        return
    
    # 1. collect all entities in the layer
    entities:list[Entity] = await mg_driver.get_entities_for_layer(layer)
    n_entities:int = len(entities)
    logger.info(f"\tcollected {n_entities} entites in layer.")

    if n_entities <= 2:
        root_layer = layer
        logger.info(f"\tSTOP!: stopped at this layer, not enough entities (<= 2)")
        return

    logger.info(f"\tgenerating embeddings for {n_entities} entity descriptions ...")
    # 2. get embeddings
    pbar = AsyncProgressBar(total=n_entities, desc="Entity description embeddings")
    entity_desc_embeds:AsyncList[EntityDescEmbed] = await embed_all_entity_descriptions(entities, ENTITY_BATCH_SIZE, MAX_PARALLEL_EMBED, layer, pbar=pbar)
    await pbar.close()

    logger.info(f"\tsuccessfully created {len(entity_desc_embeds)} embeddings!")
    if (len(entity_desc_embeds) != n_entities):
        raise RuntimeError(f"layer {layer}: Tried to embed {n_entities} entity descriptions, but got {len(entity_desc_embeds)} embeddings back!")

    # 3. get embeddings as numpy array, reduce dimesions with umap
    raw_embeds_arr = np.asarray([e["desc_embed"] for e in entity_desc_embeds], dtype=np.float32)
    _shape_before = raw_embeds_arr.shape
    raw_embeds_arr = reduce_embeddings(raw_embeds_arr, reduction_dim=min(2, n_entities-2))
    logger.info(f"\treduced embedding shape from {_shape_before} -> {raw_embeds_arr.shape}")

    # 4. Calculate num of paritions for this layer
    logger.info(f"\tcalculating number of partitions with heuristic and BCI ...")
    heuristic = n_entities // CLUSTER_SIZE
    bci_rec = get_optimal_clusters_from_embeddings(raw_embeds_arr)
    n_components = max(heuristic, bci_rec)
    logger.info(f"\theuristic: {heuristic}, BCI: {bci_rec}. Chose n_components = ({n_components})")

    if layer != 0 and n_components <= 4:
        root_layer = layer
        logger.info(f"\tSTOP!: stopped because number of partitions is small (<=4)")
        return

    clusters: dict[int, Cluster] = None
    if layer == 0 and n_components <= 4:
        # the base kg is too small to partition into multiple clusters. instead, treat full base kg as one cluster.
        clusters = {0: entities}
    else:
        # 5. Partition layer into clusters
        logger.info(f"\tfitting embeddings into GMM with n_components={n_components} ...")
        gmm:GaussianMixture = GaussianMixture(
            n_components= n_components,
            covariance_type="full",
            random_state=0,
            n_init=5
        )
        gmm.fit(raw_embeds_arr)
        responsibilities = gmm.predict_proba(raw_embeds_arr)
        labels = responsibilities.argmax(axis=1)

        clusters:dict[int, Cluster] = {k:[] for k in range(n_components)}
        for i, k in enumerate(labels):
            clusters[int(k)].append(entities[i])

    logger.info(f"\tsuccessfully built {len(clusters.keys())} clusters")

    # 6. Create aggregate nodes
    logger.info(f"\tcreating aggregates nodes from clusters:")
    layer_aggregates:AsyncList = AsyncList()
    pbar = AsyncProgressBar(total=len(clusters), desc=f"agg tasks (layer {layer})")
    _aggregation_tasks = [_aggregate_task(cluster, layer_aggregates, layer, pbar) for cluster in clusters.values()]
    logger.info(f"\rrunning {len(_aggregation_tasks)} aggregation tasks ...")
    await asyncio.gather(*_aggregation_tasks, return_exceptions=True)
    await pbar.close()
    logger.info(f"\tsuccessfully built data for {len(layer_aggregates)} unique aggregate entities! creating nodes in memgraph ...")

    for agg in layer_aggregates:
        await mg_driver.create_aggregate_entity(agg, agg_clst_map[agg['key']], layer+1)
    logger.info("\tsuccesffuly created aggregate entity nodes")

    # 7. Create inter-cluster relations
    if layer == 0 and n_components <= 4:
        # no need for this step for the small-graph case
        root_layer = layer+1
        logger.info(f"\tSTOP!: stopped because this graph was too small (created single aggregate)")
        return
    
    allowed_tokens = ( max_depth - layer ) * 40 * 2
    logger.info(f"\tcreating inter-aggregate relations. (allowed tokens: {allowed_tokens})")
    for aggJ, aggK in tqdm(combinations(layer_aggregates, 2), desc="aggregate entity inter-relations"):
        
        inter_cluster_rel:list[IntrClusterRel] = await mg_driver.get_inter_cluster_relations(agg_clst_map[aggJ['key']], agg_clst_map[aggK['key']])
        if len(inter_cluster_rel) <= 0:
            continue # not well-supported, so we don't need to create a relation between the clusters.

        # cumulative number of tokens for all inter-cluster relation descriptions.. 'connectivity strength'
        n_tokens_intercluster_rel = sum([len(Tokenizer.encode(r['relation_description'])) for r in inter_cluster_rel])
        
        if n_tokens_intercluster_rel > allowed_tokens:
            icr_desc:str = await signatures.generate_aggregate_rel_desc(aggJ, aggK, inter_cluster_rel)
        else:
            icr_desc:str = icr_desc_fallback_concat(inter_cluster_rel)

        await mg_driver.create_inter_cluster_relation(aggJ, aggK, icr_desc, layer+1)
    logger.info(f"\tsuccessfully created inter-aggregate relations! (Done with this layer)")

    await aggregate_layer_recursive(layer+1,max_depth)

async def build():
    global root_layer
    root_layer = -1

    try:
        await mg_driver.init()
        n_layer0_entities = await mg_driver.count_entities()
        max_depth = round(log(n_layer0_entities, CLUSTER_SIZE)) +1
        logger.info(f"building leanrag kg (max depth = {max_depth})")

        # build the graph with recursive hierarchical clustering:
        await build_hierarchy(max_depth)

        await mg_driver.set_root_entities(root_layer)
        logger.info(f"leanrag kg built! ({root_layer} layers)")
        return root_layer
    except Exception as e:
        G0_EMBEDDINGS_FILE.unlink() # delete the g0 embed cache if this build() step failed
        raise e