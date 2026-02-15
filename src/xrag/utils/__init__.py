from hashlib import blake2b
from .models import Chunk, HexID, DocData
import re
from itertools import islice
from tokenizers import Tokenizer as _Tokenizer
from typing import Sequence
import numpy as np
from sklearn.mixture import GaussianMixture

def normalize_from_name(name:str) -> str:
    """
    Deterministically normalize a name into a distinct key
    """
    s = name.strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "_" # never empty

def stable_id_hex(s: str, nbytes: int = 16) -> HexID:
    """
    stable hash to create IDs as strings - want to avoid precision/conversion errors with memgraph
    """
    return blake2b(s.encode("utf-8"), digest_size=nbytes).hexdigest()
    
def print_chunks_view(chunks:list[Chunk],max_chunks_print = 10, max_field_len_print = 50):
    """
    helper to print small view into the chunk objects
    """
    def _ellipsize(s: str, max_len: int) -> str:
        s = str(s)
        return s if len(s) <= max_len else s[:max_len - 3] + f"...(+{len(s[max_len-3:])})"
    
    print("====CHUNKS (VIEW)====")
    for i in range(min(len(chunks), max_chunks_print)):
        _chunk = chunks[i]
        print(f"id (#{i+1}):\t{_ellipsize(_chunk['id'], max_field_len_print)}")
        print(f"text:\t\t{_ellipsize(_chunk['raw_text'], max_field_len_print)}")
        print(f"tokens:\t\t{_ellipsize(_chunk['approx_n_tokens'], max_field_len_print)}")
        if 'embedding' in _chunk.keys():
            print(f"embed:\t\t{_ellipsize(_chunk['embedding'], max_field_len_print)}")
        if 'triples' in _chunk.keys():
            print(f"trips:\t\t{_ellipsize(_chunk['triples'], max_field_len_print)}")
        print("\n")

def get_chunk_from_id(dataset:DocData, id:HexID) -> Chunk|None:
    """ O(n) search through chunks"""

    for doc_data in dataset:
        for chunk in doc_data['chunks']:
            if chunk['id'] == id:
                return chunk
    return None

def infer_schema(obj):
    """infer schema of a json obj"""
    if isinstance(obj, dict):
        return {
            k: infer_schema(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        if not obj:
            return []
        return [infer_schema(obj[0])]
    return type(obj).__name__

def batched(iterable, size):
    if size < 1:
        raise ValueError("size must be at least 1")
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch

class Tokenizer:
    model = "gpt2"
    _tokenizer:_Tokenizer = None

    @classmethod
    def encode(self, input:str):
        if not Tokenizer._tokenizer:
            Tokenizer._tokenizer = _Tokenizer.from_pretrained(Tokenizer.model)

        return Tokenizer._tokenizer.encode(input)
    


from typing import Sequence
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

try:  # UMAP pulls in numba/llvmlite which aren't available on Python 3.12 yet.
    import umap
except ImportError:  # pragma: no cover - only hit when UMAP isn't installed.
    umap = None

def reduce_embeddings(embeddings:np.ndarray, reduction_dim=2, random_state=0):
    # LeanRAG reduces dimensionality before clustering; prefer UMAP when it's available.
    if embeddings.size == 0:
        return embeddings

    target_dim = min(reduction_dim, embeddings.shape[0] - 2) if embeddings.shape[0] > 2 else 1
    target_dim = max(1, target_dim)

    if umap is not None:
        reducer = umap.UMAP(
            n_components=target_dim,
            n_neighbors=15,
            metric="cosine",
            random_state=random_state,
        )
        return reducer.fit_transform(embeddings)

    # Fall back to PCA so Python 3.12 environments (where UMAP can't be installed)
    # still get deterministic dimensionality reduction for clustering.
    max_components = max(1, min(embeddings.shape[0], embeddings.shape[1]))
    fallback_dim = min(target_dim, max_components)
    reducer = PCA(n_components=fallback_dim, random_state=random_state)
    return reducer.fit_transform(embeddings)

def get_optimal_clusters_from_embeddings(
    reduced_embeddings: np.ndarray,
    *,
    max_clusters: int = 50,
    rel_tol: float = 1e-3,
    random_state: int = 0,
) -> int:
    """
    LeanRAG-style cluster-count selection; desc_embeddings is a list of per-entity
    description vectors (e.g., [{'key': ..., 'desc_embed': [...]}, ...] →pass only
    the desc_embed values here).

    Returns the number of clusters to request from the GMM.
    """


    capped_max = min(len(reduced_embeddings), max_clusters)
    bics = []
    prev_bic = float("inf")

    for n_components in range(1, capped_max + 1):
        gm = GaussianMixture(
            n_components=n_components,
            random_state=random_state,
            n_init=5,
            init_params="k-means++",
        )
        gm.fit(reduced_embeddings)
        bic = gm.bic(reduced_embeddings)
        bics.append(bic)

        if abs(prev_bic - bic) / (abs(prev_bic) + 1e-12) < rel_tol:
            break
        prev_bic = bic

    best_idx = int(np.argmin(bics))
    return best_idx + 1  # because we started counting at 1
