# ShelfAware

ShelfAware connects a native iPhone app and Meta glasses to a Python shopping assistant. Requested product photos feed visual identification and cited product search; the app supports spoken comparisons, follow-up questions, and an optional browser companion.

This repository contains the selected **iOS 6.1.0 build 8** source and the latest backend comparison fix, assembled on October 18, 2026. In product-search conversation mode, **“Compare them” and “Now compare them” request a fresh photo**, clear the previous selection, and cannot fall back to the old pair when capture fails. Questions about the current products can reuse that selection.

## Contents

| Directory | Purpose |
| --- | --- |
| `ios/` | SwiftUI app, glasses integration, phone fallback, and XcodeGen specifications |
| `backend/` | FastAPI sessions, product search, conversation transport, provider adapters, and tests |
| `addons/pair_pipeline/` | Shared comparison, vision, and speech adapters |
| `web/` | React/Vite shopper dashboard and observer views |
| `scripts/` | Local launch, setup, preflight, and packaging helpers |
| `tools/visual_matcher/` | Optional reference-image experiment, separate from the active product-search path |

## Backend

Use Python 3.11 or newer. From the repository root:

```sh
cp .env.example .env
# Edit .env locally to configure provider credentials.
python3 scripts/start_backend.py --setup --port 8001
```

The example selects DeepSeek for text and product search, Perceptron for vision, and ElevenLabs Scribe v2 realtime / Eleven v4 Turbo for speech. It contains no credentials. Live conversations require access to those configured providers; synthetic fixture tests run without keys. Provider errors do not automatically switch to another provider.

For an iPhone on the same trusted network, add `--lan` to the launcher command. The launcher creates or reuses a private bearer token and saves connection details in ignored `.runtime/phone-connection.txt`. Enter the computer's reachable LAN origin and token in the app. The default launcher listens only on loopback. The health endpoint is `/api/health`.

The active mode is `CONVERSATION_PRODUCT_MODE=product_search`. The optional original `reviewed_catalog` mode requires an independently reviewed catalog; no real reviewed package catalog is bundled. See [catalog setup](docs/CONVERSATION-CATALOG.md).

## iPhone app

See [native setup](ios/README.md) for XcodeGen, signing, and the glasses/phone targets. All three specifications identify version 6.1.0, build 8. Meta DAT is pinned to 1.0.1. Signing identities, provisioning profiles, SDK binaries, and installed applications are excluded.

## Browser companion

Use Node 22.12+ within Node 22, Node 24, or Node 26+. To run the backend and dashboard together on their default ports, use:

```sh
bash scripts/start.sh --setup
```

This starts the backend on port 8000 and Vite on port 5173. The browser proxy targets port 8000; it is separate from the native-only port 8001 example. Do not start a second server on an occupied port. See [web setup](web/README.md).

## Validation

After installing dependencies, run from the repository root:

```sh
.venv/bin/python -m pytest backend/tests addons/pair_pipeline -q
npm --prefix web test
npm --prefix web run build
swiftc ios/ShelfAware/SceneStabilityGate.swift ios/Tests/SceneStabilityGateTests.swift -o /tmp/shelfaware-scene-tests
/tmp/shelfaware-scene-tests
```

Include all of `addons/pair_pipeline` so its root-level capture tests run too. Tests use synthetic fixtures and mocked provider responses. They do not establish recognition accuracy, real glasses audio, or live provider availability. Product identity and online source variants can still be uncertain; catalog search results are not reviewed physical labels.

## Local data

Credentials, saved conversations, recordings, photographs, runtime checkpoints, installed dependencies, and build output are excluded. Keep local `.env` and `private-media/` files out of version control. Never put provider credentials or the LAN token in frontend `VITE_` variables. See [source provenance](docs/SOURCE-SNAPSHOT.md), [project boundaries](CONTRACT.md), and [third-party notices](THIRD_PARTY_NOTICES.md).
