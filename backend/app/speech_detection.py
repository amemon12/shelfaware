"""Bounded local endpointing for 24 kHz PCM16; cloud audio stays unchanged."""
from __future__ import annotations

import math
import struct
from collections import deque
from dataclasses import dataclass
from typing import Callable

FRAME_BYTES = 960  # 20 ms × 24,000 samples/s × signed16 mono


def vad_pcm16_16khz(pcm: bytes) -> bytes:
    """Linear 24k->16k interpolation ONLY for WebRTC VAD, never for transcription."""
    if len(pcm) != FRAME_BYTES:
        raise ValueError('VAD needs one 20 ms frame')
    source = struct.unpack('<480h', pcm)
    target = []
    for index in range(320):
        position = index * 1.5
        left = int(position)
        fraction = position - left
        value = source[left] * (1 - fraction) + source[min(left + 1, 479)] * fraction
        target.append(round(value))
    return struct.pack('<320h', *target)


@dataclass(frozen=True)
class SpeechEvent:
    kind: str
    pcm: bytes = b''
    reason: str = ''


class SpeechDetector:
    def __init__(self, *, classify: Callable[[bytes], bool] | None = None,
                 onset_ms: int = 100, silence_ms: int = 650, preroll_ms: int = 300,
                 max_speech_ms: int = 15000):
        if classify is None:
            import webrtcvad
            vad = webrtcvad.Vad(2)
            classify = lambda audio: vad.is_speech(audio, 16000)
        self.classify = classify
        self.onset_frames = math.ceil(onset_ms / 20)
        self.silence_frames = math.ceil(silence_ms / 20)
        self.max_frames = math.ceil(max_speech_ms / 20)
        self.preroll: deque[bytes] = deque(maxlen=max(self.onset_frames, math.ceil(preroll_ms / 20)))
        self.pending = bytearray()
        self.reset()

    def reset(self) -> None:
        self.pending.clear()
        self.preroll.clear()
        self.voiced = self.silent = self.total = 0
        self.active = False

    def feed(self, pcm: bytes) -> list[SpeechEvent]:
        if not pcm or len(pcm) % 2 or len(pcm) > 2880:
            raise ValueError('Expected at most 60 ms of even-length PCM16')
        self.pending.extend(pcm)
        events: list[SpeechEvent] = []
        while len(self.pending) >= FRAME_BYTES:
            frame = bytes(self.pending[:FRAME_BYTES])
            del self.pending[:FRAME_BYTES]
            speech = self.classify(vad_pcm16_16khz(frame))
            if not self.active:
                self.preroll.append(frame)
                self.voiced = self.voiced + 1 if speech else 0
                if self.voiced >= self.onset_frames:
                    self.active = True
                    self.silent = 0
                    self.total = len(self.preroll)
                    events.append(SpeechEvent('start', b''.join(self.preroll)))
                    self.preroll.clear()
            else:
                self.total += 1
                self.silent = 0 if speech else self.silent + 1
                events.append(SpeechEvent('audio', frame))
                if self.silent >= self.silence_frames or self.total >= self.max_frames:
                    reason = 'max_duration' if self.total >= self.max_frames else 'silence'
                    events.append(SpeechEvent('end', reason=reason))
                    self.active = False
                    self.voiced = self.silent = self.total = 0
                    self.preroll.clear()
        return events
