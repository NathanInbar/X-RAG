from typing import TypedDict

type HexID = str

class ExtractedSource(dict):
    id:HexID
    source: str
    normalized_text: str

class SPOTriple (TypedDict):
    s:str # subject
    p:str # predicate
    o:str # object

class Chunk (TypedDict):
    id:HexID
    raw_text:str
    approx_n_tokens:int
    embedding:list[float]
    triples:list[SPOTriple]