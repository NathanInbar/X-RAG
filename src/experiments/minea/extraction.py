"""
Triple extraction runner for MINEA evaluation.

Handles running triple extraction on needle-enriched chunks
with retry logic and error handling.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

import dspy
import litellm

from xrag.dataset_processing.preprocess import _ExtractTriples

logger = logging.getLogger(__name__)


async def run_extraction_on_enriched_chunk(
    model_name: str,
    model_id: str,
    original_text: str,
    enriched_text: str,
    initial_delay: float,
    max_attempts: int,
) -> dict[str, Any]:
    """Run triple extraction on a needle-enriched chunk."""
    try:
        model_info = litellm.get_model_info(model_id)
        model_max = model_info.get("max_output_tokens", 4000)
    except Exception as e:
        logger.error(f"[{model_name}] invalid model ID {model_id}: {e}")
        return {
            "model_name": model_name,
            "model_id": model_id,
            "triple_count": 0,
            "triples": [],
            "parse_success": False,
            "attempts": 0,
            "elapsed_seconds": 0.0,
            "error": f"Invalid model ID: {e}",
        }

    lm = dspy.LM(model_id, max_tokens=min(16000, model_max))
    predict = dspy.Predict(_ExtractTriples)

    t0 = time.monotonic()
    triples: list[dict] = []
    parse_success = False
    attempts_used = 0

    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        triples = []
        try:
            with dspy.context(lm=lm):
                pred = await predict.acall(source_text=enriched_text)
            raw_output = getattr(pred, "triples_json", "") or "[]"
            raw_output = raw_output.strip()

            if raw_output.startswith("```"):
                raw_output = re.sub(r"^```\w*\n?", "", raw_output)
                raw_output = re.sub(r"\n?```\s*$", "", raw_output)
                raw_output = raw_output.strip()

            data = json.loads(raw_output)
            if not isinstance(data, list):
                raise ValueError("not a list")

            for item in data:
                if not isinstance(item, dict):
                    continue
                # Convert to strings and strip
                s = str(item.get("subject", "")).strip()
                p = str(item.get("predicate", "")).strip()
                o = str(item.get("object", "")).strip()
                if s and p and o:
                    triples.append({"s": s, "p": p, "o": o})

            parse_success = True
            break

        except Exception as e:
            if attempt < max_attempts:
                logger.warning(
                    f"[{model_name}] attempt {attempt} failed: {e}, retrying in {delay}s"
                )
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error(f"[{model_name}] all attempts failed: {e}")

    elapsed = time.monotonic() - t0
    result = {
        "model_name": model_name,
        "model_id": model_id,
        "triple_count": len(triples),
        "triples": triples,
        "parse_success": parse_success,
        "attempts": attempts_used,
        "elapsed_seconds": round(elapsed, 3),
    }

    # Add error indicator if extraction failed
    if not parse_success:
        result["error"] = "Extraction failed after all retry attempts"

    return result
