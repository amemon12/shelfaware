# ShelfAware backend

FastAPI owns sessions, image processing, provider requests, product evidence, comparisons, and conversation WebSockets. Shared vision/voice/decision adapters live in `../addons/pair_pipeline`; keep that directory alongside the backend.

See the [root setup guide](../README.md) for installation, credentials, launch, and test commands. `scripts/start_backend.py` resolves paths from the repository root and starts only the API. `scripts/start.sh` also starts the browser companion.

The default configuration uses live `product_search`. A fresh comparison request captures new evidence and clears the old selection before capture. A failed or cancelled capture cannot reuse the previous pair. Current-product follow-ups retain bounded conversation context. The optional `reviewed_catalog` flow requires actual reviewed package facts; see [catalog setup](../docs/CONVERSATION-CATALOG.md).

Key modules:

- `app/main.py`: API wiring and request boundaries.
- `app/service.py`: sessions, turns, evidence, cancellation, and confirmation.
- `app/product_guidance.py`: product identification, cited lookup, and response validation.
- `app/conversation_product_search.py`: fresh-view and follow-up routing.
- `app/conversation_transport.py`: voice connection lifecycle and audio flow.
- `app/product_memory.py`: bounded memory scoped to selected product identities.
- `app/fixtures.py`: explicitly fictional fixture inputs.

API health is public; protected routes require the configured bearer token. Keep provider keys in the root `.env`. Session and media data are runtime state and are not part of this repository. [Exact-product lookup rationale](DATABASE_DECISION.md) describes the choices around catalog and GTIN lookup.
