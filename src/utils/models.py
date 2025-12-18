import asyncio
from typing import TypedDict, TypeVar, Generic
from tqdm import tqdm

T = TypeVar("T")
type HexID = str

#TODO: remove this
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

class DocData (TypedDict):
    id: HexID
    source: str # name
    chunks: list[Chunk]

# - - - - - - -- - - - - - END OLD

class Entity(TypedDict):
    key:str
    name:str
    desc:str
    degree:int

type Cluster = list[Entity]

class AggEntity(TypedDict):
    key:str
    name:str
    description:str

class EntityRelation(TypedDict):
    key:str
    name:str
    desc:str

class IntrClusterRel(TypedDict):
    """ Intra / Inter cluster relation"""
    source_entity:str
    target_entity:str
    relation:str
    relation_description:str

class Finding(TypedDict):
    summary:str
    explanation:str

class EntityDescEmbed(TypedDict):
    key:str
    desc_embed:list[float]



class AsyncList(Generic[T]):
    """write-protected list via asyncio lock"""
    def __init__(self):
        self._list:list[T] = []
        self._lock = asyncio.Lock()

    def __getitem__(self, index:int) -> T:
        """only read from this. elements are not write-protected"""
        return self._list[index]

    def __len__(self): return len(self._list)

    def __str__(self): return str(self._list)

    async def extend(self, rows):
        async with self._lock:
            self._list.extend(rows)
    
    async def append(self, itm):
        async with self._lock:
            self._list.append(itm)

    def get_list(self):
        """only perform read operations from this"""
        return self._list
    
    async def clear(self):
        async with self._lock:
            self._list.clear() 
    
class AsyncProgressBar:
    """ simple asyncio locked tqdm progress bar """
    def __init__(self, total:int, desc:str):
        self._pbar = tqdm(total=total, desc=desc)
        self._lock = asyncio.Lock()

    async def update(self, amt:int):
        async with self._lock:
            self._pbar.update(amt)

    async def close(self):
        async with self._lock:
            self._pbar.close()