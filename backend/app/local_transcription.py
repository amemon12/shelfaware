"""Optional local speech recognition: one CPU worker, one pending job, no audio logs.

Preload explicitly before a demo: python -m app.local_transcription --download
Normal startup uses cached files only; it never downloads a model silently.
"""
from __future__ import annotations

import argparse
import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import Any

from .openai_transcription import Transcript, TranscriptionError


@dataclass
class _Job:
    loop: Any
    future: Any
    pcm: bytes | None


class LocalWhisperEngine:
    """One bounded daemon worker shared across connections for a model."""
    def __init__(self, model: str = 'tiny.en', *, model_factory=None):
        if model not in ('tiny.en', 'base.en', 'small.en'):
            raise ValueError('Choose tiny.en, base.en, or small.en for local transcription')
        self.model_name = model
        self.model_factory = model_factory
        self.model = None
        self.jobs: queue.Queue[_Job] = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, name='shelf-local-transcription', daemon=True)
        self.thread.start()

    def _load(self):
        if self.model is None:
            if self.model_factory:
                self.model = self.model_factory()
            else:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(self.model_name, device='cpu', compute_type='int8',
                    cpu_threads=4, num_workers=1, local_files_only=True)
        return self.model

    @staticmethod
    def _deliver(future, result, error):
        if not future.done():
            if error:
                future.set_exception(TranscriptionError(error))
            else:
                future.set_result(result)

    def _run(self):
        while True:
            job = self.jobs.get()
            if job.future.cancelled():
                del job
                continue
            result, error = '', None
            try:
                result = self._recognize(job.pcm)
            except Exception:
                error = 'Local speech recognition is unavailable. Install and preload the configured model, or use text.'
            try:
                job.loop.call_soon_threadsafe(self._deliver, job.future, result, error)
            except RuntimeError:
                pass  # Event loop ended; never retain session audio or callback.
            del job
            result = error = None

    def _recognize(self, pcm):
        model = self._load()
        if pcm is None:
            return ''
        import numpy as np
        source = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
        # faster-whisper's raw array contract is 16 kHz mono floats.
        target = np.arange(0, source.size, 1.5)
        audio = np.interp(target, np.arange(source.size), source).astype(np.float32)
        segments, _ = model.transcribe(audio, language='en', beam_size=1, best_of=1,
            temperature=0, condition_on_previous_text=False, vad_filter=False,
            word_timestamps=False, no_speech_threshold=.6)
        words = []
        length = 0
        for segment in segments:
            value = segment.text.strip()
            length += len(value) + 1
            if length > 4000:
                raise ValueError('Transcript too long')
            words.append(value)
        return ' '.join(words)

    async def process(self, pcm: bytes | None):
        future = asyncio.get_running_loop().create_future()
        try:
            self.jobs.put_nowait(_Job(asyncio.get_running_loop(), future, pcm))
        except queue.Full:
            raise TranscriptionError('Local speech recognizer is busy. Please wait and repeat the question.') from None
        return await future


_engines: dict[str, LocalWhisperEngine] = {}


def engine_for(model: str):
    # Created on the backend's one asyncio event-loop thread.
    if model not in _engines:
        _engines[model] = LocalWhisperEngine(model)
    return _engines[model]


class LocalTranscription:
    def __init__(self, on_event, on_error, *, model='tiny.en', engine=None):
        self.on_event, self.on_error = on_event, on_error
        self.engine = engine or engine_for(model)
        self.ready = False
        self.closed = False
        self.active_turn = None
        self.pcm = bytearray()
        self.generation = 0
        self.tasks: set[asyncio.Task] = set()

    async def start(self):
        try:
            async with asyncio.timeout(20):
                await self.engine.process(None)
            self.ready = True
        except Exception:
            await self.close()
            raise TranscriptionError('Preload local speech recognition before starting voice.') from None

    def begin_turn(self, turn_id):
        self.discard_turn()
        self.active_turn = turn_id

    def append(self, pcm):
        if not self.ready or self.closed:
            raise TranscriptionError('Local speech recognition is unavailable.')
        if self.active_turn is not None:
            if len(self.pcm) + len(pcm) > 720000:
                self.discard_turn()
                raise TranscriptionError('Speech exceeded 15 seconds. Please ask a shorter question.')
            self.pcm.extend(pcm)

    def commit(self, turn_id):
        if turn_id != self.active_turn:
            return
        pcm = bytes(self.pcm)
        self.pcm.clear()
        self.active_turn = None
        task = asyncio.create_task(self._complete(turn_id, pcm, self.generation))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _complete(self, turn_id, pcm, generation):
        try:
            async with asyncio.timeout(25):
                text = await self.engine.process(pcm)
            if not self.closed and generation == self.generation:
                await self.on_event(Transcript('final', turn_id, text, turn_id))
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self.closed and generation == self.generation:
                await self.on_error('Local speech recognition could not finish. Please repeat, or use text.')

    def discard_turn(self):
        self.generation += 1
        self.active_turn = None
        self.pcm.clear()
        for task in list(self.tasks):
            if task is not asyncio.current_task():
                task.cancel()

    async def close(self):
        self.closed = True
        self.ready = False
        self.discard_turn()
        tasks = [task for task in self.tasks if task is not asyncio.current_task()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description='Explicitly preload the optional local speech model')
    parser.add_argument('--model', choices=['tiny.en', 'base.en', 'small.en'], default='tiny.en')
    parser.add_argument('--download', action='store_true', help='Allow downloading public model weights into the local model cache')
    args = parser.parse_args()
    try:
        from faster_whisper import WhisperModel
        WhisperModel(args.model, device='cpu', compute_type='int8', local_files_only=not args.download,
                     cpu_threads=4, num_workers=1)
    except Exception:
        parser.exit(1, 'Speech model unavailable. Install requirements-voice-local.txt, then rerun with --download while online.\n')
    print(f'Local speech model {args.model} is cached and loadable on CPU. No microphone was used.')


if __name__ == '__main__':
    main()
