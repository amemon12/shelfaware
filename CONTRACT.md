# Shelf Aware implementation contract

Decision locked 2026-10-17 22:36 PDT. Build a local browser vertical slice first, an iOS DAT bridge in parallel, and preserve honest source modes. No paid provider key is required to run the explicitly synthetic fixture mode. Provider and hardware functionality must be separately verified.

## Ownership

- Backend: `backend/**` only. Python FastAPI/Pydantic, deterministic nutrition, bounded Perceptron/Gemini adapters, in-memory sessions, tests.
- Frontend: `web/**` only. React + TypeScript + Vite, polished audio-first shopping and evidence/video view.
- Mobile: `ios/**` only. SwiftUI bridge based on current official DAT APIs; build and real-device verification require a Mac and team hardware.
- Root: repository setup, README/scripts, integration review, execution plan, end-to-end validation.

## HTTP contract

All routes below use JSON. Frontend development proxies `/api` to `http://127.0.0.1:8000`. Backend binds loopback by default. UUID identifiers, ISO UTC timestamps. Errors `{detail: string}`. No keys in browser. Responses always include source mode; fixture mode must be prominently labeled **Synthetic demo**.

`GET /api/health` => `{status:"ok", providers:{perceptron:boolean,gemini:boolean}, version:string}`. Booleans mean configured, not successful connectivity.

`GET /api/fixtures` => `{products: Product[]}`. Deliberately fictional products and values; never imply a real SKU. IDs `oat-a`, `oat-b`, `unknown`. Two oats demonstrate differing label servings. Unknown must clarify, not invent.

`POST /api/sessions` request `{mode:Mode, profile?:{display_name:string, goals:string[], avoid_ingredients:string[], consented:boolean, goal_source?:GoalSource|null}}` => `Session`.

`GoalSource = {kind:"self_selected"|"reviewed_guidance"|"synthetic", source_date:"YYYY-MM-DD"|null, confirmed:true, reviewed_at?:string}`. This metadata is optional and requires profile consent. The client omits `reviewed_at`; the server assigns its UTC timestamp when accepting the confirmed profile and returns it. Source dates must not be future dates. No raw documents, lab values, diagnosis, clinical notes, age/weight, or derived dietary thresholds belong in this object. Metadata is not sent to AI providers or spoken as clinical authority. Existing clients may omit it entirely.

`GET /api/sessions/{id}` => `Session`.

`GET /api/sessions/{id}/latest` => last completed `TurnResponse`, or HTTP404 if no completed turn. This allows a judges' browser to observe a native-phone session. The browser accepts `?session=<uuid>` (or an Attach session control) and polls while explicitly observing. It must not create turns or speak automatically in observer mode. Native app shows/copies its session ID. Empty/unchanged polls are not new events.

`GET /api/sessions/{id}/frame` => `{image_data_url:string,captured_at:string,media_time_s?:number|null}` or404. One latest user-requested capture is kept in session memory, replaced on the next capture, expired after120seconds and cleared on cancel/context/delete/shutdown. Response is `Cache-Control:no-store`. Observers label it a captured still; this is not a live video endpoint. `&stage=1` enables the silent large-text stage layout.

`POST /api/sessions/{id}/turn` request `{text:string, client_turn_id:string, recipient_id?:"self"|"household", portion_grams?:number, fixture_product_id?:string, image_data_url?:string, captured_at?:string, media_time_s?:number}` => `TurnResponse`.

Optional `barcode:string` accepts checksum-valid GTIN lengths and preserves leading zeros. Only an exact reviewed barcode already in this session may skip OCR; a new barcode alone never establishes nutrition. Explicit spoken recipient/grams override stale client snapshot fields. A spoken recipient change without a new spoken portion clears the old portion. Real clients send new images only for a deliberate new capture; follow-up questions reuse reviewed product state.

`POST /api/sessions/{id}/context` request `{recipient_id:"self"|"household", portion_grams?:number}` => `Session`. Context changes invalidate pending actions and stale replies.

`POST /api/sessions/{id}/cancel` => `Session`. Increments generation, invalidates pending actions, clears speaking/request state. Caller also aborts network and stops its TTS.

`POST /api/sessions/{id}/confirm` request `{action_id:string}` => `TurnResponse`. An expired, already used, or mismatched action is HTTP409. Cart additions happen only here. No checkout/payment.

`DELETE /api/sessions/{id}` => `{deleted:true}`.

`Mode = "fixture" | "live_phone" | "live_glasses" | "recorded_glasses"`.

`Product = {id:string,name:string,brand?:string,barcode?:string,market:string,ingredients:string[],nutrition:{basis_grams:number,carbohydrate_g:number|null,fiber_g:number|null,sugars_g:number|null,protein_g:number|null,sodium_mg:number|null},provenance:{kind:"synthetic"|"package_ocr"|"reviewed_label"|"database",source:string,captured_at:string,verified:boolean},uncertain_fields:string[]}`.

`Session = {id:string,mode:Mode,recipient_id:"self"|"household",portion_grams:number|null,generation:number,profile:{display_name:string,goals:string[],avoid_ingredients:string[],consented:boolean,goal_source?:GoalSource|null},products:Product[],current_product_id:string|null,cart:CartItem[],pending_action:Action|null,messages:Message[],created_at:string}`.

`CartItem = {id:string,product_id:string,product_name:string,recipient_id:string,quantity:number,confirmed_at:string}`.

`Action = {id:string,label:string,product_id:string,recipient_id:string,quantity:number,expires_at:string}`.

`Message = {id:string,role:"user"|"assistant",text:string,created_at:string}`.

`TurnResponse = {session:Session,reply:string,status:"ready"|"needs_image"|"needs_portion"|"needs_review"|"needs_confirmation"|"unknown"|"cancelled",evidence:Evidence[],comparison:Comparison|null,events:Event[],pending_action:Action|null}`.

`Evidence = {label:string,value:string,source:string,kind:"synthetic"|"package_ocr"|"reviewed_label"|"database"|"calculation",verified:boolean}`.

`Comparison = {portion_grams:number,recipient_id:string,items:{product_id:string,name:string,carbohydrate_g:number|null,fiber_g:number|null,sugars_g:number|null,protein_g:number|null,sodium_mg:number|null}[],summary:string}`.

`Event = {id:string,type:string,at:string,stage:string,message:string,media_time_s?:number,box?:{x:number,y:number,w:number,h:number}}`. Boxes normalized0–1 to original media, omitted unless actual model output. Events must not fabricate elapsed time, object tracks, or provider success.

## Boundaries

- Fictional fixture values can be used immediately because the whole mode is clearly labeled synthetic. A real image must never silently use a fixture. Live input with no Perceptron key returns an actionable unavailable response.
- OCR produces untrusted candidates; uncertain or unsupported units/columns request review or a new view. No model numeric repairs, missing values as zero, macro inference from food appearance, or glucose prediction.
- A `needs_review` response may present the candidate, but it must not recommend from unconfirmed extracted values. Add `POST /api/sessions/{id}/review` with `{product_id:string,confirmed:boolean,corrections?:{name?:string,nutrition?:Product["nutrition"],ingredients?:string[]}}` => `TurnResponse` if review is required. Backend and frontend coordinate this now.
- Perceptron performs genuine image OCR/localization. Gemini interprets utterances into a bounded intent/recipient/portion schema; no direct state writes. Deterministic functions own amounts and comparisons. Server templates remain a reliable fallback.
- One recipient and explicit portion per turn. Switching recipients clears portion and pending cart action. Preserve compared product facts, never silently carry personal goals into household mode.
- No telemetry/raw transcript logging, real medical data, hidden recording, or cloud keys shipped to browser. Session deletion removes local server state; provider retention is separately governed.
- Request/body/image limits, MIME checks, HTTP timeouts, provider failure handling, expiry and idempotency, and stale-generation checks are required. All network/provider behavior is testable with fakes.
