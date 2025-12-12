from typing import TypeAlias
from pydantic import BaseModel

subject: TypeAlias = str
predicate: TypeAlias = str
obj: TypeAlias = str

class Chunk(BaseModel):
    text: str
    embedding: list[float]