import dspy
from asyncio import Semaphore, Lock
from utils.models import Finding, AggEntity, IntrClusterRel
from utils import normalize_from_name

AGGREGATION_MODEL = "bedrock/us.amazon.nova-pro-v1:0"
RESPONSE_MODEL = "bedrock/us.amazon.nova-pro-v1:0"

_lm_aggregator= dspy.LM(model = AGGREGATION_MODEL)
_lm_responder = dspy.LM(model = RESPONSE_MODEL)

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
    
PRED_CREATE_AGG_NODE = dspy.Predict(_CreateAggregateNode)
async def generate_aggregate_node(
    input_text: str,
    sem: Semaphore | None = None
) -> tuple[AggEntity | None, list[Finding] | None]:
    try:
        if sem:
            async with sem:
                with dspy.context(lm=_lm_aggregator):
                    pred = await PRED_CREATE_AGG_NODE.acall(input_text=input_text)
        else:
            with dspy.context(lm=_lm_aggregator):
                pred = await PRED_CREATE_AGG_NODE.acall(input_text=input_text)

        if not pred:
            raise RuntimeError("empty prediction")

    except NotImplementedError: # ignore error silencing for now
        return (None, None)

    agg_entity = AggEntity(
        key=normalize_from_name(pred["entity_name"]),
        name=pred["entity_name"],
        desc=pred["entity_description"],
    )
    return (agg_entity, pred["findings"])

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

PRED_CREATE_AGG_REL = dspy.Predict(_CreateAggregateRel)

async def generate_aggregate_rel_desc(agg_a:AggEntity,agg_b:AggEntity, inter_cluster_relations=list[IntrClusterRel], sem:Semaphore=None) -> str:
    """ Using 2 aggregate entities, and a list of the inter-cluster relations, create a summary for the relation between aggregate A and B"""

    # form prompt-compatible string list from inter cluster relations
    sub_e_rel = [f"{icr['source_entity']}→{icr['target_entity']}→{icr['relation']}" for icr in inter_cluster_relations]
    if(sem):
        await sem.acquire()
    try:
        with dspy.context(lm=_lm_aggregator):
            pred=await PRED_CREATE_AGG_REL.acall(
                aggregate_a_name = agg_a["name"],
                aggregate_a_desc = agg_a["desc"],
                aggregate_b_name = agg_b["name"],
                aggregate_b_desc = agg_b["desc"],
                sub_entity_relationships = sub_e_rel
            )
        if not pred: raise RuntimeError("empty prediction")
    except NotImplementedError: # temp skip error silencing (throw exceptions)
        return None
    finally:
        if(sem):
            sem.release()
    
    return pred['summary']


class _AugmentedResponse(dspy.Signature):
    """
# Role
You are a helpful assistant responding to questions about data in the tables provided.

## Goal
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
Add sections and commentary to the response as appropriate for the length and format.

## Target Length and Response Format
Target Lenth: Multiple Paragraphs
Response Format: Style the response in markdown.
    """
    user_question = dspy.InputField()
    base_entity_information = dspy.InputField()
    aggregate_entity_information = dspy.InputField()
    reasoning_path_information = dspy.InputField()
    relevant_chunks = dspy.InputField()
    response = dspy.OutputField()

PRED_CREATE_AUG_RESPONSE = dspy.Predict(_AugmentedResponse)

async def generate_augmented_response(query:str,base_entity_info, agg_entity_info, reasoning_path_info, relevant_chunk_texts, sem:Semaphore) -> str:
    
    if(sem):
        await sem.acquire()
    try:
        with dspy.context(lm=_lm_responder):
            pred = await PRED_CREATE_AUG_RESPONSE.acall(
                user_question=query,
                base_entity_information=base_entity_info,
                aggregate_entity_information=agg_entity_info,
                reasoning_path_information=reasoning_path_info,
                relevant_chunks = relevant_chunk_texts
            )
    except NotImplementedError: # temp: let exceptions raise
        return None 
    finally:
        if(sem):
            await sem.release()
    return pred['response']