# Source snapshot — October 18, 2026

This repository combines the selected native **6.1.0 build 8** source with the latest backend used for the product-search conversation flow. Native build 8 was reconstructed from the clean v4 handoff plus its saved build 8 patch. Newer local native build 9/10 changes are not part of this snapshot.

The backend includes the fresh-comparison routing correction: direct “Compare them” / “Now compare them” requests take a new photo even when a pair is already selected. Selection is invalidated before capture, and failed capture or a single newly identified product cannot borrow the previous pair. Follow-up questions retain the current selection as appropriate.

Application, test, and launcher code is copied without behavioral changes. Repository preparation replaces historical deployment documents with portable setup instructions and sets the credential-free example to the selected Eleven v4 Turbo speech model. All native project specifications remain version 6.1.0 (8), and Meta DAT remains exactly 1.0.1.

Private checkpoints, real session evidence, credentials, media, device identifiers, generated projects, signing artifacts, and installed dependencies are omitted. Synthetic test fixtures and the explicitly unreviewed catalog worksheet remain included. No application or backend restart is required to publish this source snapshot.

## Publication checks

The packaged source passed 1,472 backend/add-on tests plus 44 subtests, 22 web tests, the production web build, the native scene-stability checks, and a clean unsigned phone-target build. One existing Starlette/httpx deprecation warning remains. Source fidelity was checked against the selected native and latest backend workspaces. Tracked files were checked for actual configured credential values and common credential patterns; no matches were found. No live provider or glasses trial was performed for publication.
