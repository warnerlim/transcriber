import numpy as np
from typing import Iterable, Optional
from nlp.device import pick_device
from settings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIM

# --------------------------------------------------------------------------- #
# Text embedder
# --------------------------------------------------------------------------- #
class TextEmbedder:
    """Text -> 768-dim, L2-normalised float32 vector (numpy array).

    Vectors are normalised, so the L2 metric used by the Milvus index ranks results
    exactly like cosine similarity would.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_seq_length: int = 256,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = pick_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length

        # Fail early (and clearly) if the model does not match the Milvus schema.
        self.dim = int(self.model.encode(["dimension probe"], convert_to_numpy=True).shape[1])
        if self.dim != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding model '{model_name}' outputs {self.dim}-dim vectors but the Milvus "
                f"schema expects {EMBEDDING_DIM}. Pick a 768-dim model (WECHAT_EMBEDDING_MODEL) "
                f"or change the dim in milvus.py."
            )

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Embed one text -> array of shape (768,). Call ``.tolist()`` before inserting."""
        return self.get_text_embeddings([text])[0]

    def get_text_embeddings(self, texts: Iterable[str]) -> np.ndarray:
        """Embed many texts at once (much faster than a loop) -> array of shape (n, 768)."""
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)