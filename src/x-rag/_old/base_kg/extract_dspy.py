import dspy
import json
import logging

from utils.models import subject, predicate, obj

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class _ExtractTriples(dspy.Signature):
    """
    Extract subject predicate object triples from the source text.
    Be thorough, accurate, and faithful to the source text.
    Return a JSON array under 'triples_json' with objects:
      {"subject": "...", "predicate": "...", "object": "..."}
    """

    source_text = dspy.InputField()
    triples_json = dspy.OutputField(desc="JSON array of {subject,predicate,object}")


class TripletExtractor:
    """
    Async triplet extractor from chunk text
    """

    def __init__(self, model:str) -> None:
        self._lm = dspy.LM(model)

    async def setup(self) -> None:
        return

    async def teardown(self) -> None:
        return

    async def extract(self, text: str) -> list[tuple[subject, predicate, obj]]:
        # Async DSPy call within a per-task context to avoid global settings
        try:
            with dspy.context(lm=self._lm):
                pred = await dspy.Predict(_ExtractTriples).acall(source_text=text)
        except Exception as e:
            logger.error(f"Triplet extraction error: {type(e)}")
            return [] 

        raw = getattr(pred, "triples_json", "") or "[]"

        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("triples_json must be a JSON array")
        except Exception as e:
            logger.error("DSPy triples_json parse failure: %s; raw=%r", e, raw[:200])
            return []

        out: list[tuple[subject, predicate, obj]] = []
        for item in data:
            s = (item.get("subject") or "").strip()
            p = (item.get("predicate") or "").strip()
            o = (item.get("object") or "").strip()

            if not (s and p and o):
                continue  # empty or broken triplet

            out.append((s, p, o))
        return out
