from wtpsplit import SaT

class TextSegmenter:
    """
    Segments text into chunks using wtpsplit ("Segment Any Text"):
    https://github.com/segment-any-text/wtpsplit
    """
    _model = None
    _instance = None

    @classmethod
    def configure(cls, model: str) -> None:
        cls._model = model
        cls._instance = None

    @classmethod
    def instance(cls):
        if cls._model is None:
            raise RuntimeError("Model not configured")
        
        if cls._instance is None:
            cls._instance = SaT(cls._model)
        
        return cls._instance

    @classmethod
    def create_segments(cls, text: str) -> list[str]:
        seg = cls.instance()
        return [str(s) for s in seg.split(text)]
