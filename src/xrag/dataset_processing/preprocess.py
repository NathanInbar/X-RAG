from pathlib import Path
import random
import asyncio
import random
from asyncio import Semaphore, Lock, Task, create_task, gather
import litellm
import dspy
import json
from tqdm import tqdm
from wtpsplit import SaT
from tokenizers import Tokenizer
import logging

from xrag.utils import normalize_from_name, stable_id_hex
from xrag.utils.models import SPOTriple, Chunk

from xrag.config import config
from xrag.paths import CACHE_DIR

EXTRACTION_MODEL = config.models["trip_extract"]
MAX_PARALLEL_EXTRACT = config.llm_concurrency["trip_extract"] # maximum extraction calls to send concurrently

TOKEN_MODEL = config.models["tokenizer"]
SEGMENTER_MODEL = config.models["segmenter"] # SaT model
PARAGRAPH_THRESHOLD = config.preprocess["paragraph_thresh"] # bias for paragraph length
MIN_CHUNK_TOKENS_THRESH = config.preprocess["min_chunk_tokens_thresh"] # minimum tokens before a segment is considered a chunk

EMBED_MODEL = config.models["embed"]
MAX_PARALLEL_EMBED = config.llm_concurrency["embed"] # maximum concurrent embedding model requests
BATCH_TOKEN_TARGET = config.preprocess["batch_token_target"] # pack chunks into a batch until we cross this threshold
MAX_BATCH_ITEMS = config.preprocess["max_batch_items"] # cap on request size

DESCRIPTION_GEN_MODEL = config.models["description_gen"]
MAX_DESCRIPTION_LENGTH = config.preprocess["max_desc_length"] # max description length in tokens before condensing into a summary
MAX_CONCURRENT_REQUESTS = config.llm_concurrency["description_gen"] # max concurrent requests for LLM inference
CHUNK_BATCH_SIZE = config.preprocess["max_desc_chunks"] # how many chunks to process in each batch

INITIAL_DELAY = config.llm_retry["initial_delay"]
MAX_ATTEMPTS = config.llm_retry["max_attempts"]

_embed_sem = Semaphore(MAX_PARALLEL_EMBED)
triple_sem = Semaphore(MAX_PARALLEL_EXTRACT)

# - - - - -- - dspy PROMPTS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(levelname)s | %(filename)s:%(lineno)d | %(message)s"
))

logger.addHandler(handler)

class _SummarizeDescription(dspy.Signature):
    """Generate a short summary from the description."""
    description = dspy.InputField(desc="The description to summarize")
    summary = dspy.OutputField(desc="The summary of the description")

class _GenerateDescriptions(dspy.Signature):
    """You are provided with a subject-predicate-object triple along with the context from which it was extracted from.
    Using only this information, generate a 2-3 sentence description of the each of the subject, predicate, and object elements in the target triple.
    Be faithful to the source context, do not extrapolate or invent facts. For each description, output only the description without preface.
    """

    triple = dspy.InputField(desc="The target triple extracted from the context.")
    context = dspy.InputField(desc="The context from which to describe the subject, predicate, and object elements of the triple.")
    subject_description = dspy.OutputField(desc="A 2-3 sentence description of the subject entity from the triple.")
    predicate_description = dspy.OutputField(desc="A 2-3 sentence description of the predicate (relation) between the subject and object entities")
    object_description = dspy.OutputField(desc="A 2-3 sentence description of the object entity from the triple.")


# - - - - - - - - CLASS DEFINITIONS

class TextSegmenter:
    """
    Segments text into chunks using wtpsplit ("Segment Any Text"):
    https://github.com/segment-any-text/wtpsplit
    """
    _model: str | None = None
    _instance: SaT | None = None

    @classmethod
    def configure(cls, model: str) -> None:
        cls._model = model
        cls._instance = None

    @classmethod
    def instance(cls) -> SaT:
        if cls._model is None:
            raise RuntimeError("TextSegmenter model not configured")

        if cls._instance is None:
            cls._instance = SaT(
                cls._model,
                ort_providers=["CPUExecutionProvider"],
            )

        return cls._instance

    @classmethod
    def _normalize_segments(cls, segments: list[str]) -> None:
        """
        mutably collapse 2 segments when the boundary is a dash, sanitize
        """
        out = []
        i = 0
        n = len(segments)

        while i < n:
            s = segments[i]
            if s.endswith('-') and i + 1 < n:
                s = s[:-1] + segments[i + 1]
                i += 2
            else:
                i += 1

            out.append(s)

        segments.clear()
        segments.extend(out)

    @classmethod
    def create_segments(cls, text: str) -> list[str]:

        instance = cls.instance()
        segments = instance.split(text,
            strip_whitespace=True, 
            remove_whitespace_before_inference=True, 
            paragraph_threshold=PARAGRAPH_THRESHOLD)
        
        cls._normalize_segments(segments)

        return segments


class TripleExtractor:
    """
    Async triplet extractor from chunk text
    """

    def __init__(self, model: str) -> None:
        self._lm = dspy.LM(model)

    async def extract(self, text: str) -> list[dict[str, str]]:
        # Async DSPy call within a per-task context to avoid global settings
        try:
            with dspy.context(lm=self._lm):
                pred = await dspy.Predict(_ExtractTriples).acall(source_text=text)
        except Exception as e:
            logger.error(f"Triplet extraction error: {type(e).__name__}: {e}")
            return []

        raw = getattr(pred, "triples_json", "") or "[]"

        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("triples_json must be a JSON array")
        except Exception as e:
            logger.error(f"DSPy triples_json parse failure: {e}; raw={raw[:200]!r}")
            return []

        out: list[SPOTriple] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            s = (item.get("subject") or "").strip()
            p = (item.get("predicate") or "").strip()
            o = (item.get("object") or "").strip()

            if not (s and p and o):
                continue # skip empty / broken triples

            out.append({"s": s, "p": p, "o": o}) # SPO Triple

        return out


class ElementDescription:
    """async-safe description of a Triple element (S,P,or O)"""
    def __init__(self, description: str = ""):
        self.lock = asyncio.Lock()
        self.description = description

class DescriptionMap:
    def __init__(self):
        self.map: dict[str, ElementDescription] = {}  # key -> async-safe description
        self.map_lock: asyncio.Lock = asyncio.Lock()

        # keys currently being summarized
        self.summary_tasks: set[str] = set()
        self.summary_task_lock: asyncio.Lock = asyncio.Lock()

    async def has_key(self, key: str) -> bool:
        async with self.map_lock:
            return key in self.map

    async def append_description(self, key: str, desc: str) -> str:
        """append a description to an entity and return updated"""
        # ensure entry exists
        async with self.map_lock:
            if key not in self.map:
                self.map[key] = ElementDescription(desc)
                return self.map[key].description

            entry = self.map[key]

        # append under per-key lock
        async with entry.lock:
            entry.description += "|Here is another description: " + desc
            updated = entry.description

        # token length check (dirty read acceptable)
        n_tok = len(_tokenizer.encode(updated))
        if n_tok > MAX_DESCRIPTION_LENGTH:
            await self.request_summary(key)

        return updated

    async def request_summary(self, key: str) -> None:
        async with self.summary_task_lock:
            if key in self.summary_tasks:  # already generating a summary
                return

            self.summary_tasks.add(key)
            task = asyncio.create_task(self.summarize_description(key))
            task.add_done_callback(self._summary_done) # prevent "Task exception was never retrieved" if exception occurs

    def _summary_done(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception:
            pass

    async def summarize_description(self, key: str) -> None:
        try:
            entry = self.map[key]
            async with entry.lock:
                summ = await description_generator.summarize(entry.description)
                if summ:
                    entry.description = summ
        finally:
            async with self.summary_task_lock:
                self.summary_tasks.discard(key)

    async def wait_for_summaries(self) -> None:
        # spin until the set is empty (summarize tasks complete)
        while True:
            async with self.summary_task_lock:
                pending = len(self.summary_tasks)
            if pending == 0:
                return
            await asyncio.sleep(0)  # yield to allow summary tasks to run


class DescriptionGenerator:
    """
    Generate descriptions for an entity or relation
    """

    def __init__(self, model: str, max_concurrent_requests:int) -> None:
        self._lm = dspy.LM(model)
        self.sem = asyncio.Semaphore(max_concurrent_requests)

    async def generate(self, triple:SPOTriple, context:str) -> str|None:
        try:
            triple_flat = f"subject: {triple['s']}, predicate: {triple['p']}, object: {triple['o']}"
            async with self.sem:
                with dspy.context(lm=self._lm):
                    pred = await dspy.Predict(_GenerateDescriptions).acall(triple=triple_flat, context=context)

            s_desc = pred['subject_description'] or ""
            p_desc = pred['predicate_description'] or ""
            o_desc = pred['object_description'] or ""
        except Exception as e:
            print(f"SPO Triplet description generation error: {type(e).__name__}: {e}")
            return None
        
        return (s_desc, p_desc, o_desc)
    
    async def summarize(self, desc:str) -> str|None:
        try:
            async with self.sem:
                with dspy.context(lm=self._lm):
                    pred = await dspy.Predict(_SummarizeDescription).acall(description=desc)
            summ = pred['summary']
            if not summ:
                print(f"Error summarizing description, got empty summary")
        except Exception as e:
            print(f"Error summarizing description")
            return None
        return summ










# module level vars
_tokenizer = Tokenizer.from_pretrained(TOKEN_MODEL)
triple_extractor = TripleExtractor(model=EXTRACTION_MODEL)
description_generator = DescriptionGenerator(model=DESCRIPTION_GEN_MODEL, max_concurrent_requests=MAX_CONCURRENT_REQUESTS)
description_map = DescriptionMap()


# functions
def process_chunk_text(chunk_text:str) -> Chunk:
    """ compute token statistics & trim chunk list by minimum token length"""
    chunk_text = chunk_text.strip()
    enc = _tokenizer.encode(chunk_text)
    raw_chunk = {
        "id": stable_id_hex(chunk_text),
        "raw_text": chunk_text,
        "approx_n_tokens": len(enc),
    }
    return raw_chunk

async def embed_call_with_retry(to_embed):
    retry_delay = INITIAL_DELAY
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with _embed_sem:
                resp = await litellm.aembedding(model=EMBED_MODEL, input=to_embed)
            return resp
        except Exception as e:
            if attempt == MAX_ATTEMPTS:
                error_message = f"Max retry attempts reached. Skipping {len(to_embed)} embeddings: {e}"
                logger.error(error_message)
                raise RuntimeError(error_message)
            logger.error(f"Embed attempt {attempt} failed for {len(to_embed)} descriptions. Retrying in {retry_delay}s: {e}")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2
            retry_delay += random.uniform(0, 1)

async def _embed_batch(chunks:list[Chunk]) -> list[list[float]]:
    raw_texts = [c["raw_text"] for c in chunks]
    resp = await embed_call_with_retry(raw_texts)
    data = resp['data']

    if len(data) != len(chunks):
        raise RuntimeError("Embedding count mismatch.. cant map embeddings to owning chunks")
    
    # NOTE: order is preserved such that the embedding at output[i] corresponds to the chunk at chunks[i]
    return [item['embedding'] for item in data]


async def _embed_and_update_batch(batch:list[Chunk]) -> None:
    # get embedding for batch
    embs = await _embed_batch(batch)

    for chunk, embedding in zip(batch, embs, strict=True):
        chunk["embedding"] = embedding

def _make_batches(chunks:list[Chunk]) -> list[list[Chunk]]:
    batches: list[list[Chunk]] = []
    i = 0
    n = len(chunks)

    while i < n :
        batch: list[Chunk] = []
        tok = 0

        while i < n and len(batch) < MAX_BATCH_ITEMS and tok < BATCH_TOKEN_TARGET:
            batch.append(chunks[i])
            tok += chunks[i]['approx_n_tokens']
            i+=1
        
        # if starting with a chunk that crosses target then batch will contain just that chunk
        if not batch:
            batch = [chunks[i]]
            i += 1
        
        batches.append(batch)

    return batches

async def generate_chunk_embeddings(chunks:list[Chunk])->None:

    # schedules embedding request coroutines in waves (avoid unbounded scheduling)
    batches = _make_batches(chunks)
    in_flight: list[Task[None]] = []
    WAVE = MAX_PARALLEL_EMBED * 2

    for batch in batches:
        in_flight.append(create_task(_embed_and_update_batch(batch)))
        if len(in_flight) >= WAVE:
            await gather(*in_flight)
            in_flight.clear()

    if in_flight:
        await gather(*in_flight)



class _ExtractTriples(dspy.Signature):
    """
    Extract subject predicate object triples from the source text.
    Be thorough, accurate, and faithful to the source text.
    Return a JSON array under 'triples_json' like:
    [
      {"subject": "...", "predicate": "...", "object": "..."},
      ...
    ]
    """
    source_text = dspy.InputField()
    triples_json = dspy.OutputField(desc="JSON array of {subject, predicate, object}")



async def _extract_triple_from_chunk(chunk: Chunk) -> None:
    async with triple_sem:
        triples = await triple_extractor.extract(chunk['raw_text'])

    chunk['triples'] = triples
    
    
async def extract_chunk_triples(chunks:list[Chunk]) -> None:
    # schedule extraction tasks in waves
    WAVE = MAX_PARALLEL_EXTRACT * 2
    
    in_flight: list[Task[None]] = []

    for chunk in chunks:
        in_flight.append(create_task(_extract_triple_from_chunk(chunk)))
        if len(in_flight) >= WAVE:
            await gather(*in_flight)
            in_flight.clear()

    if in_flight:
        await gather(*in_flight)


async def describe_trips_in_chunk(chunk: Chunk):
    logger.debug(f"describing triples: '{chunk['triples']}'")
    if chunk['triples'] == []:
        return
    
    async def _describe_triple(triple: SPOTriple):
        descriptions = await description_generator.generate(triple, chunk['raw_text'])
        if descriptions is None: #err
            return
        s_key = normalize_from_name(triple["s"])
        o_key = normalize_from_name(triple["o"])
        p_key = s_key + "__" + normalize_from_name(triple["p"]) + "__" + o_key

        # run the three appends concurrently, but wait for them
        await asyncio.gather(
            description_map.append_description(key=s_key, desc=descriptions[0]),
            description_map.append_description(key=p_key, desc=descriptions[1]),
            description_map.append_description(key=o_key, desc=descriptions[2]),
        )

    seen: set[str] = set()
    tasks: list[asyncio.Task] = []

    for triple in chunk["triples"]:
        s = normalize_from_name(triple["s"])
        p = normalize_from_name(triple["p"])
        o = normalize_from_name(triple["o"])
        t_hash = stable_id_hex(s + p + o)

        if t_hash not in seen:
            seen.add(t_hash)
            tasks.append(asyncio.create_task(_describe_triple(triple)))

    # ensure all triple descriptions in this chunk complete
    if tasks:
        await asyncio.gather(*tasks)
async def describe_all_chunks(chunks):
    pbar = tqdm(total=len(chunks), desc="Chunks")

    try:
        for i in range(0, len(chunks), CHUNK_BATCH_SIZE):
            batch = chunks[i : i + CHUNK_BATCH_SIZE]

            async def run_and_update(chunk):
                try:
                    await describe_trips_in_chunk(chunk)
                finally:
                    pbar.update(1)

            await asyncio.gather(*(run_and_update(c) for c in batch))
    finally:
        pbar.close()

async def process_dataset_file(file:Path):
    logger.debug(f"Processing file: '{file.stem}'")
    description_map.map.clear()
    out_results_filename = CACHE_DIR / f"{file.stem}__chunks.json"
    out_descriptions_filename = CACHE_DIR / f"{file.stem}__g0_descriptions.json"

    # dataset file should have an 'essay' field
    with open(file, "r") as fp:
        data = json.load(fp)
    essay = data['essay']

    TextSegmenter.configure(model=SEGMENTER_MODEL)
    chunk_texts:list[str] = TextSegmenter.create_segments(essay)
    if not chunk_texts:
        raise Exception(f"source '{file.name}' did not generate chunks")
    
    token_counts = []
    min_tokens, max_tokens = float('inf'), 0
    chunks:list[Chunk] = []
    for text in chunk_texts:
        chunk = process_chunk_text(text)
        n_tokens = chunk['approx_n_tokens']

        if not chunk or n_tokens < MIN_CHUNK_TOKENS_THRESH:
            continue # skip empty and small chunks
        
        if chunk['id'] in [c['id'] for c in chunks]:
            continue # skip duplicate chunks

        chunks.append(chunk)

        # token stats tracking
        token_counts.append(n_tokens)
        if(n_tokens < min_tokens):
            min_tokens = n_tokens
        if(n_tokens > max_tokens):
            max_tokens = n_tokens

    logger.debug(f"Generating chunk embeddings ...")
    await generate_chunk_embeddings(chunks)
    logger.debug(f"Extracting chunk triples ...")
    await extract_chunk_triples(chunks)

    # results.json file
    with open(out_results_filename, "w") as out_file:
        out_file.write("[")

        # single source for now
        out_file.write("{ "+f'"id":"{stable_id_hex(str(file.name))}", "source":"{file.name}",\n')
        out_file.write('"chunks": [\n')

        for i,chunk in enumerate(chunks):
            json.dump(chunk, out_file, ensure_ascii=False)
            if i != len(chunks)-1:
                out_file.write(",\n")
            out_file.flush()

        out_file.write("\n]\n}")
        out_file.write("]")

    await describe_all_chunks(chunks)
    await description_map.wait_for_summaries()
    with open(out_descriptions_filename, "w") as outfile:
        outfile.write("[\n")

        items = list(description_map.map.items())
        for i, (e_key, e_desc) in enumerate(items):
            json.dump({e_key: e_desc.description}, outfile)
            if i < len(items) - 1:
                outfile.write(",")
            outfile.write("\n")

        outfile.write("]")