from hashlib import blake2b
from .models import Chunk, HexID, DocData

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
