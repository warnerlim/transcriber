import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from settings import Config

logger = logging.getLogger("voice_assistant")


class PdfIngestionService:
    """
    Wraps the PDF-to-Milvus ingestion pipeline.

    - Runs on startup (if configured) to process any PDFs waiting in the
      ``files_to_database_here`` folder.
    - Can be triggered on-demand via the UI when the user drops new files in.
    - Prevents concurrent runs with a threading lock.
    - Writes into the dedicated ``pdf_documents`` collection (chat messages
      live separately in ``wechat_messages``).
    """

    def __init__(
        self,
        config: Config,
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._cfg = config
        self._status_cb = status_cb or (lambda _msg: None)
        self._busy = threading.Lock()

    def ingest(
        self,
        client: Any,
        embedder: Any,
        root: Optional[Path] = None,
    ) -> Dict[str, int]:
        """
        Run PDF ingestion for *root* (defaults to config.pdf_input_dir).

        Returns counters dict. Safe to call from any thread; will silently
        decline if another ingestion is already running.
        """
        if not self._busy.acquire(blocking=False):
            self._status_cb("🔄 PDF ingestion already in progress...\n")
            return {"files": 0, "records": 0, "batches": 0, "skipped": True}

        target = root or self._cfg.pdf_input_dir
        try:
            if not target.exists():
                self._status_cb(f"⚠️ PDF directory not found: {target}\n")
                return {"files": 0, "records": 0, "batches": 0}

            self._status_cb(f"🔄 Starting PDF ingestion from {target}...\n")

            from storage.pdf_store import insert_pdf_chunks
            from ingestion.pdf_ingest import ingest_directory

            def insert_batch(records: list[dict[str, Any]]) -> None:
                insert_pdf_chunks(records, client, embedder)

            counters = ingest_directory(
                root=target,
                insert_batch=insert_batch,
                chunk_size=self._cfg.pdf_chunk_size,
                chunk_overlap=self._cfg.pdf_chunk_overlap,
                ocr_model=self._cfg.pdf_ocr_model,
                batch_size=self._cfg.pdf_batch_size,
                min_text_chars=self._cfg.pdf_min_text_chars,
                dpi=self._cfg.pdf_render_dpi,
                status_cb=self._status_cb,
            )
            return counters
        except Exception as exc:
            logger.exception("PDF ingestion failed")
            self._status_cb(f"❌ PDF ingestion failed: {exc}\n")
            return {"files": 0, "records": 0, "batches": 0, "error": str(exc)}
        finally:
            self._busy.release()