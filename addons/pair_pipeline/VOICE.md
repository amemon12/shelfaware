# Speech adapter contract

Speech adapters accept validated reply text through the shared `SpeechRequest` type and yield `PCMChunk` audio for the conversation transport. The stream uses 24 kHz mono PCM16LE. Provider API keys remain on the backend.

The [repository example](../../.env.example) selects ElevenLabs Scribe v2 realtime for transcription and Eleven v4 Turbo for output. `elevenlabs_dialogue_adapter.py` implements the v4 WebSocket path; `elevenlabs_voice_adapter.py` retains the HTTP streaming alternative. `voice_adapter.py` and `openai_voice_adapter.py` retain other optional implementations. Consult each adapter's source for supported models and configuration; selecting a provider is explicit.

`backend/app/conversation_transport.py` coordinates turn cancellation, bounded audio queues, speech failures, and playback acknowledgements. The native conversation client pauses microphone input while processing and playing a response. A test or playback acknowledgement does not prove that sound reached the wearer through glasses.

Adapter tests use fake provider responses and run with the [full suite](../../README.md#validation). `audition_voice.py` can perform an explicitly requested live audition; its no-live default does not contact a provider. Never put actual keys in audition arguments, documentation, or tracked files.
