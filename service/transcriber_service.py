# transcriber_service.py
import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np
from faster_whisper import WhisperModel

from settings import Config

logger = logging.getLogger("voice_assistant")


class Transcriber:
    """
    Pulls finished utterances off an internal queue and runs the slow
    whisper.transcribe() call on its own thread, completely off the capture
    loop. If transcription can't keep up with speech, the queue backs up and
    translations lag - but capture itself never stalls or dies for it.
    """

    def __init__(
        self,
        config: Config,
        on_transcription: Callable[[str], None],
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._cfg = config
        self._on_transcription = on_transcription
        self._status_cb = status_cb or (lambda _msg: None)
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._model: Optional[WhisperModel] = None

    def load(self) -> None:
        """Load the Whisper model. CPU-heavy — call from the init thread."""
        self._model = WhisperModel(
            self._cfg.whisper_model,
            device=self._cfg.whisper_device,
            compute_type=self._cfg.whisper_compute_type,
            cpu_threads=self._cfg.whisper_cpu_threads,
        )
        logger.info("Whisper model '%s' loaded", self._cfg.whisper_model)

    def start(self) -> None:
        """Start the transcription worker thread."""
        threading.Thread(
            target=self._worker, name="transcriber", daemon=True
        ).start()
        logger.info("Transcriber started")

    def submit(self, audio_data: np.ndarray) -> None:
        """Enqueue a finished utterance (called by the AudioRecorder thread)."""
        self._queue.put(audio_data)

    def _worker(self) -> None:
        while True:
            audio_data = self._queue.get()
            try:
                segments, _ = self._model.transcribe(
                    audio_data,
                    beam_size=5,
                    vad_filter=True,
                )
                transcription = " ".join(s.text for s in segments).strip()
                if transcription:
                    self._on_transcription(transcription)
            except Exception:
                logger.exception("Transcription error (skipped one audio segment)")
                self._status_cb("\n⚠️ Transcription error (skipped one audio segment)\n")