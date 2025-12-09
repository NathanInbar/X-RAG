from pydantic import BaseModel

class Chunk(BaseModel):
    text: str
    embedding: list[float]