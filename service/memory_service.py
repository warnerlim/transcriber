# memory_service.py
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Callable, Dict, List, Optional, Tuple

from settings import Config

logger = logging.getLogger("voice_assistant")


class MemoryService:
    """
    Owns everything about long-term memory: connecting to Milvus, warming up
    the embedding stack, executing vector searches (with a timeout so a slow
    lookup can never stall a live translation) and formatting hits as text.

    The service never raises out of ``search()`` — memory must not be able to
    break a live translation.
    """

    TOOL_SCHEMA: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search the user's long-term chat memory. Support multi-step searching: "
                "start with a broad query to find relevant context, then issue a refined "
                "follow-up query with specific keywords or entities if the first results "
                "need more precision."
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
                    }
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
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory")
        self._client: Any = None
        self._embedder: Any = None

    @property
    def available(self) -> bool:
        return self._client is not None and self._embedder is not None

    def initialize(self) -> None:
        """
        Connect to Milvus, load the collection, and initialize the query embedder.

        Heavy imports remain inside this method so the application can start
        without the memory stack when memory_enabled is False.
        """
        try:
            self._status_cb("🧠 Connecting to Milvus memory...\n")

            from settings import WECHAT_COLLECTION_NAME
            from storage.milvus_schema import (
                create_or_load_wechat_collection,
                get_milvus_client,
            )

            client = get_milvus_client(
                self._cfg.milvus_host,
                self._cfg.milvus_port,
            )
            create_or_load_wechat_collection(client)
            client.load_collection(WECHAT_COLLECTION_NAME)

            from nlp.embeddings import TextEmbedder

            self._embedder = TextEmbedder()
            self._client = client

            self._raw_search("warm up")

            self._status_cb("🧠 Memory ready.\n")
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

    def search(self, query: str) -> Tuple[str, int]:
        """
        Run ``search_memory`` tool logic: vector-search past chats for *query*.

        Returns (formatted_text_block, number_of_hits). The text block is empty
        when memory is off, nothing relevant was found, or the lookup failed.
        Never raises.
        """
        if not self.available:
            return "", 0
        try:
            hits = self._pool.submit(self._raw_search, query).result(
                timeout=self._cfg.memory_timeout_sec
            )
            return self._format_hits(hits)
        except FuturesTimeout:
            logger.warning("Memory lookup timed out after %ss", self._cfg.memory_timeout_sec)
            self._status_cb(
                f"⚠️ Memory lookup timed out ({self._cfg.memory_timeout_sec:g}s) - skipping it.\n"
            )
        except Exception:
            logger.exception("Memory lookup failed")
            self._status_cb("⚠️ Memory lookup failed - skipping it.\n")
        return "", 0

    def _raw_search(self, query: str) -> List[Dict[str, Any]]:
        """Run a synchronous memory search inside the memory worker thread."""
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

    def _format_hits(self, hits: List[Dict[str, Any]]) -> Tuple[str, int]:
        """Turn raw Milvus hits into a compact, de-duplicated text block."""
        for h in hits:
            logger.info(
                "[memory] dist=%.3f | %s: %s",
                h["distance"],
                h["sender"],
                h["text"][:60],
            )
        if self._cfg.memory_max_distance is not None:
            hits = [h for h in hits if h["distance"] <= self._cfg.memory_max_distance]

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

        if not blocks:
            return "", 0
        return "\n\n".join(blocks)[: self._cfg.memory_max_chars], len(hits)