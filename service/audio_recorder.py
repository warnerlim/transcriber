# audio_recorder.py
import logging
import threading
import time
from typing import Callable, Optional

import numpy as np
import soundcard as sc

from settings import Config

logger = logging.getLogger("voice_assistant")


class AudioRecorder:
    """
    Capture-thread only job: pull frames from the loopback mic as fast as
    possible and hand finished utterances to ``on_utterance`` as raw audio.

    Nothing slow (e.g. whisper.transcribe) may run in this loop. The WASAPI
    loopback buffer is small; if record() isn't called for a stretch, SoundCard
    silently drops/corrupts audio (SoundCard GitHub issue #165) or the backend
    underruns. Wrapped in try/except + auto-reconnect so a device hiccup pauses
    capture instead of killing it forever.
    """

    def __init__(
        self,
        config: Config,
        on_utterance: Callable[[np.ndarray], None],
        on_speech_start: Callable[[], None],
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._cfg = config
        self._on_utterance = on_utterance
        self._on_speech_start = on_speech_start
        self._status_cb = status_cb or (lambda _msg: None)
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the capture thread. Call this only AFTER heavy model warmup."""
        self._thread = threading.Thread(
            target=self._listen_loop, name="audio-recorder", daemon=True
        )
        self._thread.start()
        logger.info("Audio recorder started")

    def _listen_loop(self) -> None:
        cfg = self._cfg
        while True:
            try:
                speaker = sc.default_speaker()
                loopback_mic = sc.get_microphone(
                    id=str(speaker.id), include_loopback=True
                )
                recording, silence_chunks, audio_buffer = False, 0, []

                with loopback_mic.recorder(samplerate=cfg.sample_rate) as recorder:
                    while True:
                        data = recorder.record(
                            numframes=int(cfg.sample_rate * cfg.chunk_duration)
                        )
                        data_mono = (
                            np.mean(data, axis=1) if len(data.shape) > 1 else data
                        )

                        if np.sqrt(np.mean(data_mono**2)) > cfg.volume_threshold:
                            if not recording:
                                recording = True
                                self._on_speech_start()
                            silence_chunks = 0
                            audio_buffer.append(data_mono)
                        elif recording:
                            audio_buffer.append(data_mono)
                            silence_chunks += 1

                            if (
                                silence_chunks * cfg.chunk_duration
                                >= cfg.silence_limit_sec
                            ):
                                recording = False
                                audio_data = (
                                    np.concatenate(audio_buffer, axis=0)
                                    .astype(np.float32)
                                )
                                audio_buffer = []
                                self._on_utterance(audio_data)

            except Exception:
                logger.exception("Audio capture error - reconnecting")
                self._status_cb(
                    "\n⚠️ Audio capture hiccup - reconnecting to the audio device...\n"
                )
                time.sleep(1.0)