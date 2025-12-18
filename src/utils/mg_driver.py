import asyncio
from neo4j import AsyncGraphDatabase, AsyncDriver
from utils import normalize_from_name
from utils.models import SPOTriple, Entity, AggEntity, IntrClusterRel, Cluster

_MG_URI = "bolt://localhost:7687"

_driver: AsyncDriver | None = None
_init_lock = asyncio.Lock()

async def init():
    """
    Initialize the global mempgraph client (neo4j driver), connecting to the memgraph instance
    """
    global _driver

    async with _init_lock:
        if _driver is not None:
            pass

        try:
            _driver = AsyncGraphDatabase.driver(uri = _MG_URI, auth = None)
            await _driver.verify_connectivity()
        except Exception as e:
            _driver = None
            raise Exception(f"Couldn't connect to memgraph (uri: '{_MG_URI}', auth: None): {e}") from e

async def close():
    """
    Close the global driver connection
    """
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None

def _get_driver() -> AsyncDriver:
    if _driver is None:
        raise Exception("MG Driver not initialized, call init() first.")
    return _driver

from typing import Mapping, Any
 
async def read(cypher:str, params: Mapping[str, Any]|None = None) -> list[dict[str,Any]]:
    """
    Run a read query, return a list of dict rows
    """
    driver = _get_driver()
    try:
        async with driver.session() as session:
            async def work(tx):
                res = await tx.run(cypher, params or {})
                return [record.data() async for record in res]
            
            return await session.execute_read(work)
    except (OSError, ConnectionError) as e:
        raise Exception(f"Graph read transport error: {e}") from e
    except Exception as e:
        raise Exception(f"Graph read failed: {e}") from e
    
async def write(cypher:str, params: Mapping[str, Any]|None = None) -> list[dict[str,Any]]|None:
    """
    Run a write query, optionally return a list of dict rows if the query returns any
    """
    driver = _get_driver()
    try:
        async with driver.session() as session:

            async def work(tx):
                res = await tx.run(cypher, params or {})
                # Some writes RETURN values, some don't.
                try:
                    return [record.data() async for record in res]
                except Exception:
                    await res.consume()
                    return None

            return await session.execute_write(work)
    except (OSError, ConnectionError) as e:
        raise Exception(f"Graph write transport error: {e}") from e
    except Exception as e:
        raise Exception(f"Graph write failed: {e}") from e

async def clear() -> None:
    """
    Wipes all elements from the graph
    """
    await write(
        """
        MATCH (a)
        DETACH DELETE a
        """
    )

# cypher helpers for KG

async def merge_triple(triple:SPOTriple, triple_descriptions:tuple[str,str,str], source_doc_id:int, source_chunk_id:int, layer:int=0) -> None:
    """
    Upsert an (:Entity) - [:Relation] -> (:Entity) into memgraph from a source SPO triple.
    Performs distinct union on the provenance information (source document id, source chunk ids from document)
    """
    await write(
        """
        MERGE (a:Entity {key:$skey})
          ON CREATE SET a.name = $sname, a.desc = $sdesc, a.layer=$layer
        MERGE (b:Entity {key:$okey})
          ON CREATE SET b.name = $oname, b.desc = $odesc, b.layer = $layer
        MERGE (a)-[r:Relation {key:$pkey}]->(b)
          ON CREATE SET
            r.name = $pname, r.desc = $rdesc, r.layer = $layer
        WITH r,
            coalesce(r.source_doc_ids, []) AS doc_ids, 
            coalesce(r.source_chunk_ids, []) AS chunk_ids
        WITH r,
        (CASE WHEN $doc IN doc_ids THEN doc_ids ELSE doc_ids+[$doc] END) AS merged_doc_ids,
        (CASE WHEN $chunk IN chunk_ids THEN chunk_ids ELSE chunk_ids+[$chunk] END) as merged_chunk_ids
        SET r.source_doc_ids = merged_doc_ids, r.source_chunk_ids = merged_chunk_ids
        
        """,
        {
            "sname": triple['s'],
            "pname": triple['p'],
            "oname": triple["o"],
            "sdesc": triple_descriptions[0],
            "rdesc": triple_descriptions[1],
            "odesc": triple_descriptions[2],
            "skey": normalize_from_name(triple['s']),
            "pkey": normalize_from_name(triple['p']),
            "okey": normalize_from_name(triple['o']),
            "doc": source_doc_id,
            "chunk": source_chunk_id,
            "layer": layer
        }
    )

async def get_entities_for_layer(layer:int) -> list[Entity]:
    """
    Get all entities in a layer.
    Returns list of {key:str, name:str, desc:str, degree:int}
    """
    resp = await read(
        """
        MATCH (a:Entity {layer:$layer})
        RETURN a.key AS key, a.name AS name, a.desc AS desc, degree(a) AS degree
        """,
        {"layer": layer}
    )
    return resp

async def get_intra_cluster_relations(cluster:Cluster) -> list[IntrClusterRel]:
    """
    Get relations between all entities in a cluster
    """
    resp = await read(
        """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.key IN $entity_keys AND b.key IN $entity_keys
        RETURN a.name as source_entity, b.name as target_entity, r.desc as relation_description
        """,
        {
            "entity_keys": [e['key'] for e in cluster]
        }
    )
    return resp

async def get_inter_cluster_relations(cluster_A:Cluster, cluster_B:Cluster) -> list[IntrClusterRel]:
    """
    Get relations across clusters A and B (relations between entities in A and B)
    """
    entities_A = [e['key'] for e in cluster_A]
    entities_B = [e['key'] for e in cluster_B]

    resp = await read(
        """
        MATCH (a:Entity)-[r:Relation]-(b:Entity)
        WHERE a.key IN $a_keys AND b.key IN $b_keys
        RETURN a.name as source_entity, b.name as target_entity, r.desc as relation_description
        """,
        {
            "a_keys" : entities_A,
            "b_keys" : entities_B
        }
    )
    return resp

async def create_aggregate_entity(agg_entity:AggEntity, cluster:Cluster, layer:int) -> None:
    """
    (LeanRAG) create a new aggregate entity for a cluster
    """
    children_entity_keys:list[str] = [e['key'] for e in cluster]
    await write(
        """
        MERGE (n:AggEntity {key: $agg_key})
        ON CREATE SET n.name = $agg_name, n.layer=$layer
        WITH n
        UNWIND $child_entity_keys AS child_key
        MATCH (c:Entity {key: child_key})
        MERGE (c)-[:IS_CHILD_OF]->(n)
        """,
        {
            "agg_key": agg_entity['key'],
            "agg_name": agg_entity['name'],
            "child_entity_keys" : children_entity_keys,
            "layer": layer
        }
    )

async def create_inter_cluster_relation(agg_A:AggEntity, agg_B:AggEntity, rel_desc:str, layer:int) -> None:
    """
    Create a new relation between aggregate nodes A and B, provided a description property
    """
    new_r_key = f"{agg_A['key']}__{agg_B['key']}"

    await write(
        """
        MERGE (a:AggEntity {key: $a_key})-[r:AggRelation {key: $r_key}]->(b:AggEntity {key: $b_key})
        ON CREATE SET r.desc = $r_desc, r.layer = $layer
        """,
        {
            "a_key": agg_A['key'],
            "b_key": agg_B['key'],
            "r_key": new_r_key,
            "r_desc": rel_desc,
            "layer": layer
        }
    )

async def set_root_entities(entity_keys:list[str]) -> None:
    """
    Add the :Root label to the given entities
    """

    await write(
        """
        UNWIND $root_keys as k
        MATCH (n:AggEntity {key: k})
        SET n:Root
        """
    )

async def count_entities() -> int:
    """ Returns the count of nodes with :Entity label """
    resp = await read(
        """
        MATCH (:Entity)
        RETURN count(*) AS count
        """
    )
    return resp[0]['count']