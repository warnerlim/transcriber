# pdf_ingest.py
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pymupdf
from PIL import Image
from tqdm import tqdm

from text_utils import preprocess_wechat_text

logger = logging.getLogger("pdf_ingest")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_OCR_MODEL = "StarDoc-AI/NaviDC-OCR"
OCR_PROMPT = "<image>\nPerform layout and document OCR parsing:"
DEFAULT_MIN_TEXT_CHARS = 20
DEFAULT_RENDER_DPI = 200
DEFAULT_BATCH_SIZE = 100
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
MAX_TEXT_PAYLOAD_CHARS = 2040
OCR_MAX_NEW_TOKENS = 4096

SENDER = "PDF_Ingest"
RECEIVER = "System"


# --------------------------------------------------------------------------- #
# NaviDC-OCR / TeleOCR vision-language service (lazy instantiation)
# --------------------------------------------------------------------------- #

class NaviDCOCRService:
    """Thin wrapper around the StarDoc-AI/NaviDC-OCR (TeleOCR) VLM.

    The ~1.2B-parameter model and its processor are **not** loaded at
    construction time; weights are materialised on the first call to
    :meth:`parse_page_image`.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_OCR_MODEL,
        max_new_tokens: int = OCR_MAX_NEW_TOKENS,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self._device_override = device
        self._processor: Any = None
        self._model: Any = None
        self._device: str = ""
        self._dtype: Any = None

    def _ensure_loaded(self) -> None:
        """Load processor + model into memory exactly once, on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if self._device_override:
            self._device = self._device_override
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32

        logger.info(
            "Loading OCR model '%s' on %s (%s) — first scanned page encountered.",
            self.model_path,
            self._device,
            self._dtype,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self._dtype,
        ).to(self._device)
        self._model.eval()
        logger.info("OCR model ready.")

    def parse_page_image(self, pil_image: Image.Image) -> str:
        """Run layout/document OCR parsing on a rendered page image."""
        self._ensure_loaded()
        import torch

        image = pil_image.convert("RGB")
        inputs = self._processor(
            images=image, text=OCR_PROMPT, return_tensors="pt"
        )
        inputs = {
            k: (
                v.to(self._device, self._dtype)
                if getattr(v, "dtype", None) is not None
                and str(v.dtype).startswith("torch.float")
                else v.to(self._device)
            )
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated[:, prompt_len:]
        text = self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return text.strip()


# --------------------------------------------------------------------------- #
# PDF page handling
# --------------------------------------------------------------------------- #

def render_page_to_image(
    page: pymupdf.Page, dpi: int = DEFAULT_RENDER_DPI
) -> Image.Image:
    """Render a PDF page to an in-memory PIL.Image at ``dpi`` resolution."""
    pix = page.get_pixmap(dpi=max(dpi, DEFAULT_RENDER_DPI))
    mode = "RGBA" if pix.alpha else ("RGB" if pix.n >= 3 else "L")
    image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif mode == "L":
        image = image.convert("RGB")
    return image


def extract_page_text(
    page: pymupdf.Page,
    ocr_service: NaviDCOCRService,
    min_text_chars: int,
    dpi: int,
) -> Tuple[str, str]:
    """Extract text from a single page, falling back to OCR when needed.

    Returns:
        ``(text, method)`` where ``method`` is ``"native"``, ``"ocr"`` or
        ``"empty"``.
    """
    native_text = ""
    try:
        native_text = page.get_text("text") or ""
    except Exception as exc:
        logger.warning(
            "Native extraction failed on page %d: %s", page.number + 1, exc
        )

    if len(native_text.strip()) >= min_text_chars:
        return native_text, "native"

    logger.info(
        "Page %d below min_text_chars (%d chars); using OCR fallback.",
        page.number + 1,
        len(native_text.strip()),
    )
    try:
        image = render_page_to_image(page, dpi=dpi)
        ocr_text = ocr_service.parse_page_image(image)
    except Exception as exc:
        logger.error("OCR failed on page %d: %s", page.number + 1, exc)
        return native_text.strip(), "empty"

    if ocr_text:
        return ocr_text, "ocr"
    return native_text.strip(), "empty"


# --------------------------------------------------------------------------- #
# Chunking & record construction
# --------------------------------------------------------------------------- #

def build_text_splitter(chunk_size: int, chunk_overlap: int):
    """Token-exact splitter backed by tiktoken's ``cl100k_base`` encoding."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})."
        )
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def make_msg_id(
    file_name: str, page_idx: int, chunk_idx: int, chunk_content: str
) -> str:
    """Deterministic SHA-256 record id — stable across pipeline re-runs."""
    raw = f"{file_name}|page_{page_idx + 1}|chunk_{chunk_idx}|{chunk_content[:50]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_records_for_pdf(
    pdf_path: Path,
    splitter: Any,
    ocr_service: NaviDCOCRService,
    min_text_chars: int,
    dpi: int,
) -> Iterator[Dict[str, Any]]:
    """Yield fully-formed Milvus records for every chunk of one PDF."""
    file_name = pdf_path.name
    try:
        mtime = int(pdf_path.stat().st_mtime)
    except OSError:
        mtime = 0

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        logger.error("Cannot open %s: %s — skipping file.", pdf_path, exc)
        return

    stats = {"native": 0, "ocr": 0, "empty": 0}
    with doc:
        total_pages = doc.page_count  # Captured safely inside the context manager
        for page_idx in range(total_pages):
            page = doc.load_page(page_idx)
            raw_text, method = extract_page_text(
                page, ocr_service, min_text_chars, dpi
            )
            stats[method] += 1

            cleaned = preprocess_wechat_text(raw_text)
            if not cleaned:
                continue

            for chunk_idx, chunk in enumerate(splitter.split_text(cleaned)):
                chunk = chunk.strip()
                if not chunk:
                    continue
                yield {
                    "msg_id": make_msg_id(file_name, page_idx, chunk_idx, chunk),
                    "sender": SENDER,
                    "receiver": RECEIVER,
                    "chat_id": file_name,
                    "text": chunk[:MAX_TEXT_PAYLOAD_CHARS],
                    "timestamp": mtime,
                    "metadata": {
                        "source_path": str(pdf_path.resolve()),
                        "page_number": page_idx + 1,
                        "chunk_index": chunk_idx,
                    },
                }

    logger.info(
        "%s: %d pages (native=%d, ocr=%d, empty=%d).",
        file_name,
        total_pages,
        stats["native"],
        stats["ocr"],
        stats["empty"],
    )


# --------------------------------------------------------------------------- #
# Directory scan + batch upsert
# --------------------------------------------------------------------------- #

def find_pdfs(root: Path) -> List[Path]:
    """Recursively collect ``*.pdf`` files under ``root`` (sorted, stable)."""
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"
    )


def ingest_directory(
    root: Path,
    insert_batch: Callable[[list[dict[str, Any]]], Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ocr_model: str = DEFAULT_OCR_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS,
    dpi: int = DEFAULT_RENDER_DPI,
    status_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, int]:
    """Run the full ingestion pipeline over ``root``.

    Args:
        root: Directory scanned recursively for *.pdf files.
        insert_batch: Required callback that receives a list of records to insert.
        chunk_size: Chunk size in cl100k_base tokens.
        chunk_overlap: Chunk overlap in cl100k_base tokens.
        ocr_model: HuggingFace model path for the OCR VLM.
        batch_size: Records per Milvus upsert batch.
        min_text_chars: Pages with less native text than this are sent to OCR.
        dpi: Render resolution for OCR fallback pages.
        status_cb: Optional callback for status messages.

    Returns:
        Counters: files seen, records generated, batches upserted.
    """
    _status = status_cb or (lambda _msg: None)

    pdfs = find_pdfs(root)
    if not pdfs:
        logger.warning("No PDF files found under %s", root)
        _status(f"📁 No PDF files found under {root}\n")
        return {"files": 0, "records": 0, "batches": 0}

    logger.info("Found %d PDF file(s) under %s", len(pdfs), root)
    _status(f"📁 Found {len(pdfs)} PDF file(s) under %s\n")

    splitter = build_text_splitter(chunk_size, chunk_overlap)
    ocr_service = NaviDCOCRService(model_path=ocr_model)

    counters = {"files": 0, "records": 0, "batches": 0}
    batch_queue: List[Dict[str, Any]] = []

    def flush() -> None:
        if not batch_queue:
            return
        insert_batch(list(batch_queue))
        counters["batches"] += 1
        batch_queue.clear()

    for pdf_path in tqdm(pdfs, desc="Ingesting PDFs", unit="file"):
        try:
            for record in build_records_for_pdf(
                pdf_path,
                splitter,
                ocr_service,
                min_text_chars,
                dpi,
            ):
                batch_queue.append(record)
                counters["records"] += 1
                if len(batch_queue) >= batch_size:
                    flush()
            counters["files"] += 1
            _status(f"  ✅ Processed {pdf_path.name}\n")
        except Exception:
            logger.exception("Unhandled error while processing %s", pdf_path)
            _status(f"  ❌ Error processing {pdf_path.name}\n")

    flush()

    summary = (
        f"✅ Ingestion complete: {counters['files']} file(s), "
        f"{counters['records']} record(s), {counters['batches']} batch(es).\n"
    )
    logger.info(summary.strip())
    _status(summary)
    return counters