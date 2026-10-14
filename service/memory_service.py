import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Callable, Dict, List, Optional, Tuple

from settings import Config

logger = logging.getLogger("voice_assistant")

_VALID_SCOPES = ("auto", "chat", "pdf")


class MemoryService:
    """
    Owns everything about long-term memory: connecting to Milvus, warming up
    the embedding stack, executing vector searches (with a timeout so a slow
    lookup can never stall a live translation) and formatting hits as text.

    Memory is split across TWO Milvus collections:
      - ``wechat_messages`` : past chat/conversation messages
      - ``pdf_documents``   : chunks of ingested PDF files, each carrying its
                              file_name, page_number and chunk_index

    The ``search_memory`` tool accepts a ``scope`` argument so the agent can
    route a query to the right database ("chat" / "pdf") — or leave it on
    "auto", which searches both collections in parallel and returns the best
    matches from each, clearly labelled by source. A ``file_name`` argument
    can further restrict a PDF search to one specific document.

    The service never raises out of ``search()`` — memory must not be able to
    break a live translation.
    """

    TOOL_SCHEMA: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search the user's long-term memory, which is split into two vector "
                "databases: past chat messages and ingested PDF documents. Supports "
                "multi-step searching: start with a broad query to find relevant "
                "context, then issue a refined follow-up query with specific keywords "
                "or entities if the first results need more precision."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A concise search query. Can be broad initially, and "
                            "refined into specific terms during subsequent tool rounds."
                        ),
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["auto", "chat", "pdf"],
                        "description": (
                            "Which vector database to search. 'chat' = past "
                            "conversations only, 'pdf' = ingested documents only, "
                            "'auto' (default) = search both in parallel and return "
                            "the best matches from each, labelled by source."
                        ),
                    },
                    "file_name": {
                        "type": "string",
                        "description": (
                            "Optional. Restrict a document search to a single PDF by "
                            "its file name, e.g. 'annual_report.pdf'. Only affects "
                            "the document database (scope 'pdf' or 'auto')."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    }

    def __init__(
        self,
        config: Config,
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._cfg = config
        self._status_cb = status_cb or (lambda _msg: None)
        # Two workers so "auto" scope can query both collections in parallel.
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory")
        self._client: Any = None
        self._embedder: Any = None
        # Instance-level copy: the file list discovered at init is baked into
        # this schema's file_name description without mutating the class attr.
        self._tool_schema = copy.deepcopy(self.TOOL_SCHEMA)

    @property
    def tool_schema(self) -> Dict[str, Any]:
        """The (possibly document-annotated) tool schema to hand to the LLM."""
        return self._tool_schema

    @property
    def available(self) -> bool:
        return self._client is not None and self._embedder is not None

    def initialize(self) -> None:
        """
        Connect to Milvus, load BOTH collections, and initialize the embedder.

        Heavy imports remain inside this method so the application can start
        without the memory stack when memory_enabled is False.
        """
        try:
            self._status_cb("🧠 Connecting to Milvus memory...\n")

            from settings import WECHAT_COLLECTION_NAME, PDF_COLLECTION_NAME
            from storage.milvus_schema import (
                create_or_load_wechat_collection,
                create_or_load_pdf_collection,
                get_milvus_client,
            )

            client = get_milvus_client(
                self._cfg.milvus_host,
                self._cfg.milvus_port,
            )
            create_or_load_wechat_collection(client)
            create_or_load_pdf_collection(client)
            client.load_collection(WECHAT_COLLECTION_NAME)
            client.load_collection(PDF_COLLECTION_NAME)

            from nlp.embeddings import TextEmbedder

            self._embedder = TextEmbedder()
            self._client = client

            self._raw_chat_search("warm up")
            self._raw_pdf_search("warm up")

            self._advertise_ingested_files()

            self._status_cb("🧠 Memory ready (chats + documents).\n")
            logger.info(
                "Memory service initialized (host=%s:%s)",
                self._cfg.milvus_host,
                self._cfg.milvus_port,
            )
        except Exception as exc:
            logger.exception("Memory initialization failed")
            self._client = None
            self._embedder = None
            self._status_cb(
                f"⚠️ Memory unavailable ({type(exc).__name__}: {str(exc)[:150]}) "
                "- translating without it.\n"
            )

    def _advertise_ingested_files(self) -> None:
        """Bake the list of ingested PDF file names into the tool schema, so
        the agent knows which documents exist and can target them with the
        ``file_name`` argument."""
        try:
            from storage.conversation_retrieval import list_ingested_files

            files = list_ingested_files(self._client)
        except Exception:
            logger.exception("Could not list ingested PDF files")
            return
        if not files:
            return
        listing = ", ".join(f"'{f}'" for f in files[:25])
        file_param = self._tool_schema["function"]["parameters"]["properties"]["file_name"]
        file_param["description"] += f" Currently ingested documents: {listing}."
        self._status_cb(f"🧠 Document memory: {len(files)} file(s) indexed: {listing}\n")

    def search(
        self,
        query: str,
        scope: str = "auto",
        file_name: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Run ``search_memory`` tool logic: vector-search memory for *query*.

        ``scope`` picks the database(s): "chat", "pdf", or "auto" (both, in
        parallel). ``file_name`` optionally restricts the PDF search to one
        document. Returns (formatted_text_block, number_of_hits). The text
        block is empty when memory is off, nothing relevant was found, or the
        lookup failed. Never raises.
        """
        if not self.available:
            return "", 0
        if scope not in _VALID_SCOPES:
            logger.warning("Unknown memory scope %r — falling back to 'auto'", scope)
            scope = "auto"

        started = time.monotonic()
        try:
            futures = {}
            if scope in ("auto", "chat"):
                futures["chat"] = self._pool.submit(self._raw_chat_search, query)
            if scope in ("auto", "pdf"):
                futures["pdf"] = self._pool.submit(self._raw_pdf_search, query, file_name)

            results: Dict[str, List[Dict[str, Any]]] = {}
            for key, future in futures.items():
                remaining = self._cfg.memory_timeout_sec - (time.monotonic() - started)
                results[key] = future.result(timeout=max(remaining, 0.05))
            return self._format_hits(results)
        except FuturesTimeout:
            logger.warning("Memory lookup timed out after %ss", self._cfg.memory_timeout_sec)
            self._status_cb(
                f"⚠️ Memory lookup timed out ({self._cfg.memory_timeout_sec:g}s) - skipping it.\n"
            )
        except Exception:
            logger.exception("Memory lookup failed")
            self._status_cb("⚠️ Memory lookup failed - skipping it.\n")
        return "", 0

    def _raw_chat_search(self, query: str) -> List[Dict[str, Any]]:
        """Vector-search the chat collection."""
        if not self.available:
            return []

        from storage.conversation_retrieval import retrieve_conversation_memory

        return retrieve_conversation_memory(
            query=query,
            client=self._client,
            embedder=self._embedder,
            contact_name=self._cfg.memory_contact,
            top_k=self._cfg.memory_top_k,
            include_context_window=True,
        )

    def _raw_pdf_search(
        self, query: str, file_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Vector-search the PDF collection, optionally within one file."""
        if not self.available:
            return []

        from storage.conversation_retrieval import retrieve_pdf_memory

        return retrieve_pdf_memory(
            query=query,
            client=self._client,
            embedder=self._embedder,
            file_name=file_name,
            top_k=self._cfg.memory_pdf_top_k,
            include_neighbor_chunks=True,
        )

    def _within_distance(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self._cfg.memory_max_distance is None:
            return hits
        return [h for h in hits if h["distance"] <= self._cfg.memory_max_distance]

    def _format_hits(self, results: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, int]:
        """Combine chat and PDF hits into one source-labelled text block."""
        chat_hits = self._within_distance(results.get("chat", []))
        pdf_hits = self._within_distance(results.get("pdf", []))

        for h in chat_hits:
            logger.info(
                "[memory:chat] dist=%.3f | %s: %s",
                h["distance"], h["sender"], h["text"][:60],
            )
        for h in pdf_hits:
            logger.info(
                "[memory:pdf] dist=%.3f | %s p.%s: %s",
                h["distance"], h["file_name"], h["page_number"], h["text"][:60],
            )

        sections = []
        chat_block = self._format_chat_hits(chat_hits)
        if chat_block:
            sections.append("Past conversations:\n" + chat_block)
        pdf_block = self._format_pdf_hits(pdf_hits)
        if pdf_block:
            sections.append("Ingested documents:\n" + pdf_block)

        if not sections:
            return "", 0
        return (
            "\n\n".join(sections)[: self._cfg.memory_max_chars],
            len(chat_hits) + len(pdf_hits),
        )

    def _format_chat_hits(self, hits: List[Dict[str, Any]]) -> str:
        """Turn raw chat hits into a compact, de-duplicated text block."""
        seen, blocks = set(), []
        for h in hits:
            day = h["datetime"][:10]
            window = h.get("surrounding_chat") or [
                {"time": h["datetime"][11:], "sender": h["sender"], "text": h["text"]}
            ]
            lines = []
            for m in window:
                key = (h["chat_id"], day, m["time"], m["sender"], m["text"])
                if key not in seen:
                    seen.add(key)
                    lines.append(f"  [{m['time']}] {m['sender']}: {m['text']}")
            if lines:
                blocks.append(f"Chat '{h['chat_id']}' on {day}:\n" + "\n".join(lines))
        return "\n\n".join(blocks)

    def _format_pdf_hits(self, hits: List[Dict[str, Any]]) -> str:
        """Turn raw PDF hits into a de-duplicated block labelled by file/page."""
        seen, blocks = set(), []
        for h in hits:
            day = (h.get("datetime") or "")[:10]
            chunks = h.get("neighboring_chunks") or [
                {
                    "page_number": h["page_number"],
                    "chunk_index": h["chunk_index"],
                    "text": h["text"],
                }
            ]
            lines = []
            for c in chunks:
                key = (h["file_name"], c["page_number"], c["chunk_index"], c["text"])
                if key not in seen:
                    seen.add(key)
                    lines.append(f"  [p.{c['page_number']}] {c['text']}")
            if lines:
                header = f"Document '{h['file_name']}', page {h['page_number']}"
                if day:
                    header += f" (file last modified {day})"
                blocks.append(header + ":\n" + "\n".join(lines))
        return "\n\n".join(blocks)