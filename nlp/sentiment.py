from typing import Dict, Iterable, List, Optional, Sequence
from nlp.device import pick_device
from settings import DEFAULT_SENTIMENT_MODEL, SENTIMENT_LABELS, SENTIMENT_DIM, NEUTRAL_SCORES

# --------------------------------------------------------------------------- #
# Sentiment model
# --------------------------------------------------------------------------- #
def _resolve_label_order(id2label: Dict[int, str]) -> List[int]:
    """Return the model's output indices for [negative, neutral, positive]."""
    found: Dict[str, int] = {}
    for index, name in id2label.items():
        lowered = str(name).strip().lower()
        for label in SENTIMENT_LABELS:
            if lowered.startswith(label[:3]):
                found[label] = int(index)
    if len(found) != len(SENTIMENT_LABELS):
        raise ValueError(
            f"Cannot map the model labels {dict(id2label)} to negative/neutral/positive. "
            f"Pass label_order=[idx_negative, idx_neutral, idx_positive] to SentimentModel."
        )
    return [found[label] for label in SENTIMENT_LABELS]

class SentimentModel:
    """Text -> [P(negative), P(neutral), P(positive)] as a plain list of 3 floats.

    The three values sum to 1 and are stored in the ``sentiment_scores`` vector field.
    Whatever order the underlying model uses, the output order is always
    negative, neutral, positive.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SENTIMENT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 256,
        label_order: Optional[Sequence[int]] = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = pick_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        if self.model.config.num_labels != SENTIMENT_DIM:
            raise ValueError(
                f"Sentiment model '{model_name}' has {self.model.config.num_labels} labels but the "
                f"Milvus schema expects {SENTIMENT_DIM} (negative/neutral/positive)."
            )
        self._order = (
            list(label_order) if label_order is not None else _resolve_label_order(self.model.config.id2label)
        )

    def sent_analysis(self, text: str) -> List[float]:
        """Score one text -> [negative, neutral, positive]."""
        return self.sent_analysis_batch([text])[0]

    def sent_analysis_batch(self, texts: Iterable[str]) -> List[List[float]]:
        """Score many texts at once. Empty/blank texts get a neutral score."""
        texts = list(texts)
        results = [list(NEUTRAL_SCORES) for _ in texts]
        todo = [i for i, t in enumerate(texts) if t and t.strip()]

        for start in range(0, len(todo), self.batch_size):
            indices = todo[start:start + self.batch_size]
            encoded = self.tokenizer(
                [texts[i] for i in indices],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.inference_mode():
                logits = self.model(**encoded).logits
            probs = self._torch.softmax(logits.float(), dim=-1)[:, self._order]
            for i, row in zip(indices, probs.cpu().tolist()):
                results[i] = [float(p) for p in row]
        return results