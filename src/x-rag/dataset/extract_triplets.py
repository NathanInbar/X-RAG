# extract triplets for chunks (async concurrent per-chunk; serialized file writes)

import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import dspy
import json
import ijson
import litellm

SRC = Path(__file__).resolve().parents[2]

EXTRACTION_MODEL = "anthropic.claude-opus-4-5-20251101-v1:0"

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


class TripletExtractor:
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
            print(f"Triplet extraction error: {type(e).__name__}: {e}")
            return []

        raw = getattr(pred, "triples_json", "") or "[]"

        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("triples_json must be a JSON array")
        except Exception as e:
            print(f"DSPy triples_json parse failure: {e}; raw={raw[:200]!r}")
            return []

        out: list[dict[str, str]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            s = (item.get("subject") or "").strip()
            p = (item.get("predicate") or "").strip()
            o = (item.get("object") or "").strip()

            if not (s and p and o):
                continue
                #TODO normalize surfaces
                #TODO build canonical entity dictionary

            out.append({"subject": s, "predicate": p, "object": o})

        return out


async def process(
    input_filename: str,
    output_filename: str,
    extractor: TripletExtractor,
    *,
    max_parallel: int = 8,
):
    """ 
    - Reads input as a stream (ijson)
    - For each source object, runs chunk extraction concurrently (bounded)
    - Writes output sequentially (single coroutine) to avoid race conditions
    """
    input_path = Path(input_filename)
    sem = asyncio.Semaphore(max_parallel)

    async def extract_one(chunk_text: str) -> dict:
        # Bound concurrency to avoid hammering Bedrock / running out of sockets.
        async with sem:
            triplets = await extractor.extract(chunk_text)
            return {"chunk_text": chunk_text, "triplets": triplets}

    with open(input_path, "rb") as input_file, open(output_filename, "w", encoding="utf-8") as output_file:
        output_file.write("[\n")
        first = True

        # Expecting: [ { "source": "...", "chunks": ["foo", "bar", ...] }, ... ]
        for jobj in ijson.items(input_file, "item"):
            source = jobj["source"]
            chunks = jobj.get("chunks") or []
            if not isinstance(chunks, list):
                chunks = []

            tasks = [asyncio.create_task(extract_one(ct)) for ct in chunks]
            out_chunks = await asyncio.gather(*tasks, return_exceptions=True)

            # Normalize exceptions into empty triplet results (keeps output schema stable)
            normalized_chunks: list[dict] = []
            for idx, result in enumerate(out_chunks):
                if isinstance(result, Exception):
                    print(f"[{source}] chunk {idx} task failed: {type(result).__name__}: {result}")
                    normalized_chunks.append({"chunk_text": chunks[idx], "triplets": []})
                else:
                    normalized_chunks.append(result)

            out_jobj = {
                "source": source,
                "chunks": normalized_chunks,
            }

            if not first:
                output_file.write(",\n")
            first = False

            json.dump(out_jobj, output_file, ensure_ascii=False)
            output_file.flush()

        output_file.write("\n]")


def main():
    parser = argparse.ArgumentParser(description="Extract triplets for chunks")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", default="output_final.json")
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=8,
        help="Max concurrent chunk extraction calls (per process).",
    )
    args = parser.parse_args()

    load_dotenv(f"{SRC}/secrets.env")

    extractor = TripletExtractor(model=EXTRACTION_MODEL)

    asyncio.run(process(args.input_file, args.output_file, extractor, max_parallel=args.max_parallel))


if __name__ == "__main__":
    main()
