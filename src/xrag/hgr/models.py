from typing import Literal
from pydantic import BaseModel
from xrag.paths import DATASETS_DIR

Domain = Literal["agriculture", "cs", "hypertension", "legal", "mix"]

class Chunk(BaseModel):
    id: str
    raw_text: str
    kb: list[tuple[str, int]] = []
    e: list[tuple[str, str, str, int]] = []
    failed: bool = False
	
class Query(BaseModel):
    question: str
    golden_answers: list[str]
    context: list[str]

    # retrieval
    retrieved_context: str | None = None
    retrieval_error: bool | None = None

    # generation
    response: str | None = None
    generation_error: bool | None = None

    # evaluation
    em: float | None = None
    f1: float | None = None
    rsim: float | None = None
    gen_exp: str | None = None



