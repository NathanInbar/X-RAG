import dspy
import json
import asyncio
import logging
import warnings

from tqdm import tqdm
from pathlib import Path

from xrag.config import config
from xrag.paths import DATASETS_DIR
from xrag.utils.eval import dspy_evaluate

warnings.filterwarnings("ignore", category=DeprecationWarning)

IN_DIR = DATASETS_DIR / "OURS" / "JSON Mine Dataset"
OUT_DIR = DATASETS_DIR / "OURS" / "With Rationale"

TRIM_MODEL = dspy.LM(config.models["trim_model"])
JUDGE_MODEL = dspy.LM(config.models["eval_judge"])

MAX_TRIM_DEPTH = 3

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------

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

class _TrimSignature(dspy.Signature):
    """Extract only the portions of context needed to support or derive the given statement.
    You are seeing a PORTION of a larger document, not the complete text.
    Only return information that directly helps derive the statement - be strict about relevance."""
    
    actual_context: str = dspy.InputField(
        desc=(
            "A PORTION of a larger text that may or may not contain information relevant to the statement"
        )
    )
    statement: str = dspy.InputField(
        desc="The target statement or claim that should be supported by information in the context"
    )
    optimal_context: str = dspy.OutputField(
        desc=(
            "The trimmed context containing ONLY portions that DIRECTLY support deriving the statement. "
            "Requirements:\n"
            "- This portion may not contain any relevant information - if so, return EMPTY STRING\n"
            "- Be STRICT: only include text that directly helps derive the statement\n"
            "- Remove ALL sentences/passages that don't directly support the statement\n"
            "- If all information is directly relevant, don't trim unnecessarily\n"
            "- Copy relevant text verbatim from actual_context (exact wording, no paraphrasing)\n"
            "- Output plain text only (no markdown, bold, italics, or special formatting)\n"
            "- Preserve original sentence structure and punctuation for included text"
        )
    )

trim = dspy.Predict(_TrimSignature)

async def trim_to_optimal(context, statement, depth, max_chunk_size = 8000):
    chunks = chunk_context(context, max_chunk_size)
    all_optimal = []
    with dspy.context(lm=TRIM_MODEL):
        for chunk in chunks:
            result = trim(actual_context=chunk,statement=statement)
            if result.optimal_context == "": continue
            all_optimal.append(result.optimal_context)

    optimal_context = " ".join(all_optimal)

    with dspy.context(lm=JUDGE_MODEL):
        contained = (await dspy_evaluate.acall(context=optimal_context, statement=statement)).context_contains_statement
    if contained and depth < MAX_TRIM_DEPTH:
        return await trim_to_optimal(optimal_context, statement, depth + 1)
    else:
        return context

async def process_single_file(fp: Path):

    # skip if already processed
    outfile = OUT_DIR / fp.name
    if outfile.exists():
        return 1, ""

    # load dataset doc
    try:
        with open(fp, "r") as f:
            doc:dict = json.load(f)
        essay:str = doc["essay"]
        answers:list[str] = doc["answers"]
    except Exception as e:
        return  0,  f"Failed read from {fp.stem}: {e}"
    
    # get rationale for each statement
    for i, a in enumerate(tqdm(answers, desc=f"Trimming Context")):
        rationale = await trim_to_optimal(context=essay, statement=a, depth=0)
        answers[i] = { "answer": a, "rationale": rationale }

    # save output with rationale
    try:
        with open(outfile, "w") as f:
            json.dump(doc, f)
    except Exception as e:
        return 0, f"failed write to {outfile.name}: {e}"

    return 1, ""

async def process_all_files(files):
    errors = []
    total = len(files)
    
    for i, f in enumerate(files):
        logger.info(f"Starting {f.stem} ({i+1}/{total})")
        status, message = await process_single_file(f)
        if status == 0:
            errors.append({"file": f.name, "error": message})
    
    return errors

# ------------------------------------------------------------------------------

if __name__ == "__main__":

    if not (IN_DIR.exists() and OUT_DIR.exists()):
        raise FileNotFoundError(f"{IN_DIR} or {OUT_DIR} does not exist.")
    
    files = list(IN_DIR.iterdir())
    total = len(files)
    logger.info(f"{total} files to process")

    errors = asyncio.run(process_all_files(files))

    try:
        error_file = "rationale_errors.json"
        with open(error_file, "w") as f:
            json.dump(errors, f)
        logger.info(f"{len(errors)} errors output to {error_file}")
    except Exception as e:
        logger.error(f"Could not dump errors to {error_file}: {e}")
        logger.error(errors)

    logger.info(f"============= All Done :) ===========")
    logger.info(f"{total-len(errors)}/{total} succeeded")
    

        


        