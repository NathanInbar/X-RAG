import json
import dspy
import logging

from pathlib import Path

from xrag.config import config
from xrag.paths import DATASETS_DIR

IN_DIR = DATASETS_DIR / "OURS/JSON Mine Dataset"
OUT_DIR = DATASETS_DIR / "OURS_RATIONALE"
TRIM_MODEL = dspy.LM(config.models["trim_model"])

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def chunk_context(text, chunk_size):
    """Split text into chunks at paragraph/sentence boundaries."""
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        end_pos = current_pos + chunk_size
        
        if end_pos >= len(text):
            chunks.append(text[current_pos:])
            break
        
        chunk_text = text[current_pos:end_pos]
        last_break = chunk_text.rfind('\n\n')
        
        if last_break > chunk_size * 0.5:
            end_pos = current_pos + last_break
        else:
            last_sentence = max(
                chunk_text.rfind('. '),
                chunk_text.rfind('.\n'),
                chunk_text.rfind('! '),
                chunk_text.rfind('? ')
            )
            if last_sentence > chunk_size * 0.5:
                end_pos = current_pos + last_sentence + 1
        
        chunks.append(text[current_pos:end_pos])
        current_pos = end_pos
    
    return chunks

class TrimSignature(dspy.Signature):
    """Trim the context to only include information necessary to answer the query.
    Remove irrelevant sections while preserving all relevant content exactly as written."""
    
    actual_context: str = dspy.InputField(
        desc = "The full context that may contain both relevant and irrelevant information"
    )
    statement: str = dspy.InputField(
        desc = "The statement that needs to be derived from the context."
    )
    optimal_context: str = dspy.OutputField(
        desc = (
            "The trimmed context containing only relevant information needed to derive the statement."
            "If no relevant information exists in the text, make optimal_context empty."
            "Use the exact wording from the actual_context."
            "Do not paraphrase, summarize, or rewrite. Remove only the irrelevant sections."
            "Output plain text with no formatting (bold, italics, or markdown)."
        )
    )

trim = dspy.Predict(TrimSignature)

def trim_to_optimal(context, statement, max_chunk_size = 8000):
    chunks = chunk_context(context, max_chunk_size)
    logger.debug(f"total chunks: {len(chunks)}")
    all_optimal = []
    with dspy.context(lm=TRIM_MODEL):
        for context in chunks:
            result = trim(
                actual_context=context,
                statement=statement
            )

            if not result.optimal_context == '':
                all_optimal.append(result.optimal_context)

    logger.debug(f"total optimal from chunks: {len(all_optimal)}")
    optimal_context = " ".join(all_optimal)
    return optimal_context

if __name__ == "__main__":

    if not (IN_DIR.exists() and OUT_DIR.exists()):
        raise FileNotFoundError(f"{IN_DIR} or {OUT_DIR} does not exist.")
    
    for i,f in enumerate(IN_DIR.iterdir()):

        outfile = OUT_DIR / f.name
        if outfile.exists():
            logger.info(f"{f.name} ({i+1}/53) already done, skipping")
            continue
        
        logger.info(f"Starting {f.name} ({i+1}/39)")

        try:
            with open(f, "r") as fp:
                doc:dict = json.load(fp)
            essay:str = doc["essay"]
            answers:list[str] = doc["answers"]
        except Exception as e:
            logger.error(f"Couldn't get data from {f.stem}, skipping")
            continue
        
        sum_length = 0
        for j, a in enumerate(answers):
            logger.info(f"Answer {j+1}/15")
            rationale = trim_to_optimal(essay, a)
            sum_length += len(rationale)
            answers[j] = { "answer": a, "rationale": rationale }

        avg_rationale = sum_length / len(answers)
        logger.info(f"Avg. rationale length for {f.stem}: {avg_rationale:2f}")
        
        logger.info(f"Writing new file to {outfile}")
        try:
            with open(outfile, "w") as f:
                json.dump(doc, f)
                logger.info(f"Done {f.name}")
        except Exception:
            logger.error(f"Failed to write output for {f.name}, skipping")

        


        