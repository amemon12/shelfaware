# Reviewed catalog for the conversational shopping brain

> Optional mode: this document describes `CONVERSATION_PRODUCT_MODE=reviewed_catalog`. The repository example uses `product_search`; it does not require a reviewed catalog.

The new `backend/app/conversation_brain.py` implements a separate voice-connection shopping state. It imports the isolated deterministic pair decision module and Perceptron candidate detector. The transport/router must instantiate one brain per connection and retain it across turns. This is not a claim of real recognition accuracy on the four packages.

Configuration: `DEEPSEEK_API_KEY`, optional `DEEPSEEK_MODEL=deepseek-flash`, `PERCEPTRON_API_KEY`, and optional `SHELF_CATALOG_PATH`. The default path is the project's `private-media/catalog.json`, independent of the launch directory. The brain never reads an OpenAI key or silently falls back to OpenAI. DeepSeek produces text and visual candidates, not audio. Speech transport/playback is separate.

The explicit alternative is `CONVERSATION_REASONING_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_MODEL=claude-sonnet-5-5`. Only the selected provider receives intent/vision requests; comparison arithmetic and confirmation remain local. Relative catalog paths resolve against the repository root even when the backend launches from `backend/`.

DeepSeek uses the documented `/chat/completions` endpoint with `thinking: {type: disabled}` and JSON-object mode. JSON syntax alone is not full schema enforcement: the application validates every result using strict Pydantic models and requires a complete `stop` response with no tool calls. Local unambiguous commands avoid provider requests. Unknown natural language uses the actual adapter; it is not a scripted-only flow. Current official references: [JSON mode](https://api-docs.deepseek.com/guides/json_mode/), [thinking control](https://api-docs.deepseek.com/guides/thinking_mode/), [vision](https://api-docs.deepseek.com/guides/vision/).

Start with an honest empty catalog: `{"version":1,"products":[]}`. The app can connect, but comparison returns a setup message. No real-package nutrition is bundled.

`catalog-reference.example.json` is an **intentionally invalid, unreviewed worksheet**. Do not use it as a runtime catalog. Fill the values from the exact physical packages and their correct serving column, then set `label_reviewed:true` only after review and provide the actual UTC `reviewed_at`. Missing nutrients remain null; never enter zero to make the form pass. At most four entries are accepted. Use matching categories for genuine alternatives, such as `chips` and `juice_drink`; grams and mL cannot be interchanged. For whole bottles, enter verified `net_quantity` using the same unit as the label basis and set `net_quantity_reviewed:true` after checking the physical bottle.

Optional reference images live inside the catalog directory. Each entry accepts one JPEG/PNG, at most 2 MiB: `{path:"drink-a-front.jpg",width:800,height:600,box_xyxy_px:[100,40,500,580]}`. Numbers in this example describe an illustrative crop, not a real product; use actual decoded dimensions and actual box. The loader decodes images with Pillow and rejects incorrect dimensions, invalid boxes, oversized files and paths escaping the catalog directory. The public catalog snapshot omits reference paths/images. Perceptron Detect proposes candidates from these exemplars. If there is no usable detection, configured DeepSeek vision may propose a catalog identity from the fresh image; the source is explicitly recorded. Every proposed identity needs user confirmation before catalog nutrition is used.

Validate without any provider call, from the backend directory:

```bash
../.venv/bin/python -m app.conversation_brain --validate ../private-media/catalog.json
../.venv/bin/python -m app.conversation_brain --schema
```

Conversation example: “Compare these” → fresh camera → “Are these exact variants correct?” → “Yes” → “Both whole bottles” → confirm proposed shopping goals → deterministic comparison → “Half of each bottle” → recalculation → “Add the second one” → “Yes” → “Shopping list.” At a two-item prompt, “half the bottle” or an unqualified amount is ambiguous unless only one item lacks a portion. Name first/second/both; unknown units and label nutrient amounts cannot silently replace consumption portions.

“Compare these” captures anew and invalidates the old comparison immediately; “compare again” reuses a confirmed pair. Switching to family clears portions, goals, and pending actions. Personal ingredient exclusions are not applied to the household. A declared ingredient match fails the constraint; absence from text remains unknown, never an allergy-safe claim. Unknown constraints or selected nutrients block an overall winner.

Normal speech-start interruption must call `brain.cancel(clear_pending=False)` so that replying “yes” preserves the pending confirmation; an explicit stop/cancel/context change uses `brain.cancel()` to discard it. The transport also cancels its running task/audio. Confirmation is one-use and expires after 120 seconds. The brain's cart is an explicitly confirmed list in its snapshot; it does not execute checkout. The integrated transport synchronizes confirmed entries to the existing session cart by action ID.

The transport now merges confirmed conversation cart entries into the existing session cart by action ID. Earlier discussion of a brain-only cart describes the module in isolation; the integrated transport owns synchronization. The brain stores bounded product/portion/goal/cart state in memory, not raw frames or transcript logs. The transport retains at most 20 recent messages in session memory for the evidence view. A source frame is only sent for an explicit capture request; it must be fresh at request acceptance. Runtime validation and fake-provider tests demonstrate the protocol and arithmetic, not real provider access, package detection, audio routing, clinical correctness, or latency on the friend's Mac.
