# Shared comparison and speech adapters

These modules are imported by the backend and must remain in `addons/pair_pipeline` relative to the repository root. See the [setup guide](../../README.md) for the active configuration.

- `decision.py`: deterministic decisions for the optional reviewed-catalog workflow.
- `capture_gate.py` and `perceptron_adapter.py`: capture and candidate-detection helpers.
- `voice_adapter.py`: shared speech request/PCM types and the optional Gemini adapter.
- `elevenlabs_voice_adapter.py`: ElevenLabs HTTP speech streaming.
- `elevenlabs_dialogue_adapter.py`: Eleven v4 Turbo dialogue WebSocket streaming.
- `openai_voice_adapter.py`: retained optional speech adapter.
- `audition_voice.py`: explicit adapter audition utility; inspect `--help` before enabling a live provider call.

The repository example selects DeepSeek reasoning/search, Perceptron vision, and ElevenLabs transcription/speech. Alternative adapters remain available in source; providers are not automatically switched after errors. The active product-search conversation route lives in `backend/app/conversation_product_search.py`.

Run all shared and backend tests from the repository root:

```sh
.venv/bin/python -m pytest backend/tests addons/pair_pipeline -q
```

Include the whole add-on directory: its root-level `test_capture_gate.py` is part of the suite. Tests use synthetic inputs and mocked providers. See [speech contract](VOICE.md) and [optional reviewed-catalog setup](../../docs/CONVERSATION-CATALOG.md).
