import dspy
from asyncio import Semaphore
from utils.models import Finding, AggEntity, EntityRelation
from utils import normalize_from_name

class _CreateAggregateNode(dspy.Signature):
    """
You are an expert in concept synthesis. Your task is to identify a meaningful aggregate entity from a set of related entities and extract structured insights based solely on provided evidence.

## SKILLS
- Abstraction and naming of collective concepts based on entity types
- Structured summarization and typology recognition
- Comparative analysis across multiple entities
- Strict grounding to provided data (no hallucinated content)

## GOALS
- Derive a meaningful aggregate entity that broadly represents the given entity set
- The aggregate entity name must not match any single entity in the set
- Provide an accurate and concise description of the aggregate entity reflecting shared characteristics
- Extract 5-10 structured findings about the entity set based on grounded evidence

## Rules
- Grounding Rule: All content must be based solely on the provided entity set — no external assumptions
- Naming Rule: The aggregate entity name must not be identical to any single entity; it should reflect a composite structure, function, or theme
- Each finding must include a concise summary and a detailed explanation
- Avoid adding speculative or unsupported interpretations

## Workflows
1. Review the list of entities, focusing on types, descriptions, and relational structure
2. Synthesize a generalized name that best represents the full entity set
3. Write a clear, evidence-based description of the aggregate entity
4. Extract and elaborate on key findings, emphasizing structure, purpose, and interconnections
    """
    input_text:str = dspy.InputField(desc="The input list of entities and their relations")
    entity_name:str = dspy.OutputField(desc="<name>")
    entity_description:str = dspy.OutputField(desc="<brief description summarizing the shared traits and structure>")
    findings:list[Finding] = dspy.OutputField(desc=(
            "JSON array of objects. "
            "Each object MUST have keys: 'summary' (string) and 'explanation' (string). "
            "Example: "
            "["
            "{'summary': '...', 'explanation': '...'}, "
            "{'summary': '...', 'explanation': '...'}"
            "]"))

async def generate_aggregate_node(input_text:str, findings_map:dict[str,list[Finding]], sem:Semaphore=None) -> AggEntity|None:
    try:
        if(sem):
            await sem.acquire()
        pred=await dspy.predict(_CreateAggregateNode(input_text=input_text))
        if(sem):
            sem.release()

        if not pred: raise RuntimeError("empty prediction")
    except Exception:
        return None
    
    agg_entity = AggEntity(
        key=normalize_from_name(pred['entity_name']),
        name=pred['entity_name'],
        description=pred['entity_description']
    )

    # set entry in the findings map
    findings_map[agg_entity['key']] = pred['findings']

    return agg_entity

class _CreateAggregateRel(dspy.Signature):
    """
You specialize in analyzing relationships between two aggregation entities. Your goal is to synthesize one high-level, abstract summary sentence describing how two named aggregations are connected, based solely on their descriptions and sub-entity relationships.

## Skills
- Aggregated reasoning across entity groups
- Abstraction of cross-entity relationships
- Formal summarization under strict constraints
- Strong grounding without repetition or speculation

## Goals
- Produce a single-sentence summary (≤{tokens} words) explaining the nature of the relationship between two aggregation entities
- Avoid reproducing individual sub-entity relationships
- Emphasize structural, functional, or thematic connections at the group level

## Rules
- DO NOT output `relationship<|>` lines or copy sub-entity relationship descriptions
- DO NOT name specific sub-entities (e.g., individuals)
- DO NOT use the term “community”; always refer to “aggregation,” “group,” “collection,” or thematic equivalents
- DO use collective terms (e.g., “external reviewers,” “trade policy actors”)
- The sentence must be ≤{tokens} words, factual, grounded, and in formal English
- The relationship must reflect an **aggregation-level abstraction**, such as:
  - support/collaboration
  - review/feedback
  - functional alignment
  - domain linkage (e.g., one produces work, the other evaluates it)
## Example

### Input:
Aggregation A Name: WTO External Contributors  
Aggregation A Description: A group of economists and trade policy experts who provided feedback on early drafts of WTO reports.  

Aggregation B Name: WTO Flagship Reports  
Aggregation B Description: Core analytical publications from the WTO addressing international trade issues.  

Sub-Entity Relationships:
- Person A → early drafts of WTO report → gave feedback  
- Person B → early drafts → reviewed document  
...

### ✅ Output:
WTO External Contributors played an advisory role to the WTO Flagship Reports aggregation by offering critical expert feedback on preliminary drafts, strengthening the analytical rigor and credibility of the final publications.
"""
    aggregate_a_name:str = dspy.InputField(desc="Name of Aggregate A")
    aggregate_a_desc:str = dspy.InputField(desc="Description of Aggregate A")
    aggregate_b_name:str = dspy.InputField(desc="Name of Aggregate B")
    aggregate_b_desc:str = dspy.InputField(desc="Description of Aggregate B")
    sub_entity_relationships:list[str] = dspy.InputField(desc="The sub-entity relationships between entities in aggregate A and B")
    summary:str = dspy.OutputField(desc="The high-level, abstract summary sentence describing how two named aggregations are connected")

async def generate_aggregate_rel_desc(agg_a:AggEntity,agg_b:AggEntity, inter_cluster_relations=list[EntityRelation], sem:Semaphore=None) -> str:
    """ Using 2 aggregate entities, and a list of the inter-cluster relations, create a summary for the relation between aggregate A and B"""

    # form prompt-compatible string list from inter cluster relations
    sub_e_rel = [f""]
    raise NotImplementedError("implement sub_e_rel")
    try:
        if(sem):
            await sem.acquire()

        pred=await dspy.predict(_CreateAggregateRel(
            aggregate_a_name = agg_a["name"],
            aggregate_a_desc = agg_a["description"],
            aggregate_b_name = agg_b["name"],
            aggregate_b_desc = agg_b["description"],
            sub_entity_relationships = sub_e_rel
        ))

        if(sem):
            sem.release()

        if not pred: raise RuntimeError("empty prediction")
    except Exception:
        return None
    
    return pred['summary']