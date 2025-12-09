from models import Chunk

class ChunkDatabase():
    """
    In-memory database of chunks
    """
    
    def __init__(self):
        self.chunks = dict[int, Chunk]
        self.idx_itr = 0

    def chunks(self) -> dict[int,Chunk]:
        return self.chunks

    def add_record(self, chunk:Chunk) -> None:
        self.idx_itr += 1
        self.chunks[self.idx_itr] = chunk
    
    def get_record(self, idx:int) -> Chunk|None:
        if not idx in self.chunks.keys:
            return None
        
        return self.chunks[idx]