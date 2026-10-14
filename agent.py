import json
import logging
import queue
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from settings import Config
from service.memory_service import MemoryService

logger = logging.getLogger("voice_assistant")

_SCOPE_LABELS = {
    "auto": "chats + documents",
    "chat": "past chats",
    "pdf": "documents",
}

class AssistantAgent:
    """
    Manages the conversation with the LLM: message history, interruption, and
    an agentic tool-calling loop.

    Memory retrieval is NOT automatic: the model receives the ``search_memory``
    tool schema and decides per-turn whether it needs past-chat or document
    context, and which vector database to search (scope). When it emits a tool
    call, this class executes it against MemoryService, feeds the result back
    as a ``tool`` message, and asks the model to continue — up to
    ``config.max_tool_rounds`` times per turn.
    """

    def __init__(
        self,
        config: Config,
        memory: MemoryService,
        write_cb: Callable[[str], None],
    ) -> None:
        self._cfg = config
        self._memory = memory
        self._write = write_cb

        self._client: Optional[OpenAI] = None
        self._events: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._history: List[Dict[str, Any]] = []

        self.interrupt_flag = threading.Event()
        self.is_running = False

    def connect(self) -> None:
        """Create the HTTP/OpenAI client (cheap, but kept off the UI thread)."""
        http_client = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=10)
        )
        self._client = OpenAI(
            base_url=self._cfg.lm_base_url,
            api_key=self._cfg.lm_api_key,
            http_client=http_client,
            timeout=self._cfg.lm_timeout_sec,
        )
        logger.info("LLM client ready (%s)", self._cfg.lm_base_url)

    def start(self) -> None:
        """Start the conversation worker thread."""
        threading.Thread(
            target=self._worker, name="assistant-agent", daemon=True
        ).start()
        logger.info("Assistant agent started")

    def enqueue(self, event_type: str, text: str) -> None:
        """Queue a new turn: event_type is 'AUDIO' or 'KEYBOARD'."""
        self._events.put((event_type, text))

    def interrupt_if_running(self) -> None:
        """Abort the in-flight LLM stream/lookup when newer input arrives."""
        if self.is_running:
            self.interrupt_flag.set()

    def _worker(self) -> None:
        while True:
            try:
                event_type, text = self._events.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._run_turn(event_type, text)
            except Exception as exc:
                logger.exception("LLM turn failed")
                self._write(f"\n❌ Error: {exc}")
            finally:
                self.is_running = False
                self._write("\n" + "-" * 40)

    def _run_turn(self, event_type: str, text: str) -> None:
        if event_type == "AUDIO":
            formatted_text = f"[AUDIO INPUT]: {text}"
            self._write(f"\n🗣️ Heard: {text}\n")
        else:
            formatted_text = f"[TEXT INPUT]: {text}"
            self._write(f"\n⌨️ You: {text}\n")

        self._history.append({"role": "user", "content": formatted_text})
        self.is_running = True
        self.interrupt_flag.clear()

        messages: List[Dict[str, Any]] = (
            [{"role": "system", "content": self._cfg.system_prompt}] + self._history
        )
        tools = [self._memory.tool_schema] if self._memory.available else None

        assistant_reply = ""
        translation_header_written = False

        for _round in range(self._cfg.max_tool_rounds + 1):
            if self.interrupt_flag.is_set():
                break

            reply_text, tool_calls, translation_header_written = self._stream_completion(
                messages, tools, translation_header_written
            )

            if self.interrupt_flag.is_set():
                break

            if not tool_calls:
                assistant_reply = reply_text
                break

            messages.append(self._tool_call_message(reply_text, tool_calls))
            for call in tool_calls:
                messages.append(self._execute_tool_call(call))
        else:
            logger.warning("Tool-calling loop hit max rounds (%s)", self._cfg.max_tool_rounds)

        if self.interrupt_flag.is_set():
            self._write(" [⏸️ Interrupted by new input]\n")
            return

        if assistant_reply:
            self._history.append(
                {"role": "assistant", "content": assistant_reply.strip()}
            )
        if len(self._history) > self._cfg.history_limit:
            self._history = self._history[-self._cfg.history_limit :]

    def _stream_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        header_written: bool,
    ) -> Tuple[str, List[Dict[str, Any]], bool]:
        kwargs: Dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools

        stream = self._client.chat.completions.create(
            model=self._cfg.lm_model,
            messages=messages,
            max_tokens=self._cfg.lm_max_tokens,
            stream=True,
            **kwargs,
        )

        reply_parts: List[str] = []
        tool_calls: Dict[int, Dict[str, str]] = {}

        for chunk in stream:
            if self.interrupt_flag.is_set():
                break
            delta = chunk.choices[0].delta

            if delta.content:
                if not header_written:
                    self._write("🤖 Assistant: ")
                    header_written = True
                reply_parts.append(delta.content)
                self._write(delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    entry = tool_calls.setdefault(
                        tc.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function and tc.function.name:
                        entry["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        entry["arguments"] += tc.function.arguments

        ordered = [tool_calls[i] for i in sorted(tool_calls)]
        return "".join(reply_parts), ordered, header_written

    @staticmethod
    def _tool_call_message(
        reply_text: str, tool_calls: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": reply_text or None,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for call in tool_calls
            ],
        }

    def _execute_tool_call(self, call: Dict[str, str]) -> Dict[str, Any]:
        name, raw_args = call["name"], call["arguments"]
        logger.info("Tool call requested: %s(%s)", name, raw_args)

        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            logger.warning("Malformed tool arguments: %r", raw_args)
            args = {}

        if name != "search_memory":
            content = f"Unknown tool: {name}"
        else:
            query = str(args.get("query", "")).strip()
            scope = str(args.get("scope", "auto")).strip().lower()
            if scope not in _SCOPE_LABELS:
                scope = "auto"
            file_name = str(args.get("file_name", "")).strip() or None

            target = f"document '{file_name}'" if file_name else _SCOPE_LABELS[scope]
            self._write(f'📚 Memory: searching {target} for "{query}"...\n')

            block, n_hits = self._memory.search(query, scope=scope, file_name=file_name)
            if block:
                self._write(f"📚 Memory: {n_hits} relevant snippet(s) found\n")
                content = f"<memory>\n{block}\n</memory>"
            else:
                self._write("📚 Memory: nothing relevant found\n")
                content = "No relevant memories found."

        return {"role": "tool", "tool_call_id": call["id"], "content": content}