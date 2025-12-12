# Utility to take a json array with pdf text and split it into chunks
import argparse
import json
from pathlib import Path

import ijson
from wtpsplit import SaT
from tokenizers import Tokenizer

SEGMENTER_MODEL = "sat-3l-sm"
PARAGRAPH_THRESHOLD = 0.5 # default = 0.5

DISPLAY_TOKEN_STATS = True
if DISPLAY_TOKEN_STATS:
    _tokenizer = Tokenizer.from_pretrained("gpt2")

import statistics

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

def process(input_filename: str, output_filename: str) -> None:
    input_path = Path(input_filename)
    output_path = Path(output_filename)

    with input_path.open("rb") as in_file, \
         output_path.open("w", encoding="utf-8") as out_file:

        out_file.write("[\n")

        first = True
        # top-level JSON is: [ { "source": ..., "text": ... }, ... ]
        for jobj in ijson.items(in_file, "item"):
            try:
                chunks = TextSegmenter.create_segments(jobj["text"])
                if not chunks:
                    raise Exception(f"source '{jobj["source"]}' did not generate chunks")
                out_jobj = {
                    "source": jobj["source"],
                    "chunks": chunks,
                }

                if not first:
                    out_file.write(",\n")
                json.dump(out_jobj, out_file, ensure_ascii=False)
                out_file.flush()

                first = False

            except Exception as e:
                print(f"Error segmenting pdf '{jobj['source']}': {e}")


            if DISPLAY_TOKEN_STATS:
                token_counts = []
                min_tokens = float('inf')
                max_tokens = 0

                for chunk_text in chunks:
                    enc = _tokenizer.encode(chunk_text)
                    num_tokens = len(enc)
                    token_counts.append(num_tokens)

                    if(num_tokens < min_tokens):
                        min_tokens = num_tokens
                    if(num_tokens > max_tokens):
                        max_tokens = num_tokens

                print(f"\n====STATS FOR '{jobj["source"]}'====")
                print(f"# Chunks: {len(token_counts)}")
                print(f"Median chunk tokens: {statistics.median(token_counts)}")
                print(f"Mean chunk tokens: {statistics.mean(token_counts)}")
                print(f"Min chunk token count: {min_tokens}")
                print(f"Max chunk token count: {max_tokens}")
                print("=============\n")

        out_file.write("\n]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment pdf-text JSON into chunked JSON using wtpsplit."
    )
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_file", default="out_chunks.json", type=str)

    args = parser.parse_args()

    TextSegmenter.configure(SEGMENTER_MODEL)
    TextSegmenter.instance() #warm up

    process(args.input_file, args.output_file)


if __name__ == "__main__":
    main()