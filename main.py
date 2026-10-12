# main.py
import os
import threading

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "2")

import tkinter as tk
from typing import Optional

from agent import AssistantAgent
from service.audio_recorder import AudioRecorder
from settings import Config
from service.memory_service import MemoryService
from service.pdf_ingestion_service import PdfIngestionService
from service.transcriber_service import Transcriber
from ui import AppUI


class VoiceAssistantApp:
    """Wires UI, audio, transcription, memory, PDF ingestion and the agent together."""

    def __init__(self, root: tk.Tk, config: Optional[Config] = None) -> None:
        self.config = config or Config()

        self.ui = AppUI(
            root,
            on_user_input=self._on_user_input,
            on_reingest=self._on_reingest_pdfs,
        )
        self.memory = MemoryService(self.config, status_cb=self.ui.write)
        self.pdf_ingestion = PdfIngestionService(
            self.config, status_cb=self.ui.write
        )
        self.agent = AssistantAgent(
            self.config, self.memory, write_cb=self.ui.write
        )
        self.transcriber = Transcriber(
            self.config,
            on_transcription=self._on_transcription,
            status_cb=self.ui.write,
        )
        self.recorder = AudioRecorder(
            self.config,
            on_utterance=self.transcriber.submit,
            on_speech_start=self._on_speech_start,
            status_cb=self.ui.write,
        )

        self.ui.write(
            "⏳ Loading AI models in the background...\n"
            "(You can type, but translations will begin once models load)\n"
        )
        threading.Thread(
            target=self._bootstrap, name="bootstrap", daemon=True
        ).start()

    def _bootstrap(self) -> None:
        self.transcriber.load()
        self.agent.connect()

        if self.config.memory_enabled:
            self.memory.initialize()

        if self.config.pdf_ingest_on_startup and self.memory.available:
            self.pdf_ingestion.ingest(
                client=self.memory._client,
                embedder=self.memory._embedder,
            )
        elif self.config.pdf_ingest_on_startup and not self.memory.available:
            self.ui.write(
                "⚠️ Skipping PDF ingestion — memory service is unavailable.\n"
            )

        self.ui.write(
            "\n✅ Models loaded! \n"
            "🎧 Listening for interviewer's audio...\n"
            "⌨️ Type your response below and hit Enter.\n"
            "🔄 Click 'Re-ingest PDFs' anytime to add new documents.\n"
            + "-" * 40
            + "\n"
        )

        self.recorder.start()
        self.transcriber.start()
        self.agent.start()

    def _on_user_input(self, text: str) -> None:
        self.agent.interrupt_if_running()
        self.agent.enqueue("KEYBOARD", text)

    def _on_transcription(self, text: str) -> None:
        self.agent.enqueue("AUDIO", text)

    def _on_speech_start(self) -> None:
        self.agent.interrupt_if_running()

    def _on_reingest_pdfs(self) -> None:
        if not self.memory.available:
            self.ui.write("⚠️ Cannot ingest PDFs — memory service is unavailable.\n")
            return

        self.ui.write("🔄 Re-ingesting PDFs from files_to_database_here...\n")
        threading.Thread(
            target=self.pdf_ingestion.ingest,
            args=(self.memory._client, self.memory._embedder),
            name="pdf-reingest",
            daemon=True,
        ).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAssistantApp(root)
    root.mainloop()