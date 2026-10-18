# Optional visual reference matcher — isolated experiment

**Decision:** useful as a conservative shortlist for a handful of distinctive snack packages. It is not a general grocery classifier, a trained neural model, or a substitute for reading the barcode. No real product photographs have been supplied; grocery accuracy is entirely unverified. Nothing here changes the backend, web app, iOS app, or their dependencies.

The CLI extracts ORB binary descriptors, keeps reciprocal distinctive Hamming matches, then checks a RANSAC homography and spatial coverage. Those operations follow the official [ORB documentation](https://docs.opencv.org/4.13.0/d1/d89/tutorial_py_orb.html), [feature-matching documentation](https://docs.opencv.org/4.13.0/dc/dc3/tutorial_py_matcher.html), and [homography tutorial](https://docs.opencv.org/4.13.0/d1/de0/tutorial_py_feature_homography.html). The thresholds and conservative decision rules are this project's engineering choices, not calibrated probabilities.

## What to collect at QFC

For 4–6 products, capture 5–8 sharp front-panel photos per SKU, with modest viewpoint and distance changes. Crop around the actual package, not shared shelf scenery. Preserve exact flavor, package size, market and barcode separately. Favor rigid boxes or flat fronts; curved bottles, crumpled bags, glare, tiny distant objects and occluded branding are harder for a planar matcher. Do not use nutrition panels as visual identities; many products share their layout.

Use 3–5 photos as references. Re-hold the item and take the remaining 2–3 in a **separate session**, ideally on the actual glasses, as a holdout. Adjacent video frames and edits of reference photos are not independent validation. Keep at least two unseen snacks and two similar flavor/size variants as negative examples. Include blur, a blank image, hands covering the package, and two products in one frame.

Put similar packaging variants in the same `variant_group`. The matcher deliberately refuses candidate output for any group containing more than one indexed SKU. A visually indistinguishable, unindexed variant may still match: every returned candidate requires exact barcode and label verification.

## Install and run

Python 3.11 or 3.12 recommended. This tool has its own virtual environment. From this directory, Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
.venv\Scripts\python.exe -m unittest -v
.venv\Scripts\python.exe matcher.py index manifest.json snacks.index.json
.venv\Scripts\python.exe matcher.py query snacks.index.json data\holdout\known-snack.jpg
.venv\Scripts\python.exe matcher.py evaluate snacks.index.json holdout.json --output evaluation.json
```

On Mac substitute `python3 -m venv .venv` and `.venv/bin/python`. Mac execution remains unverified. Copy the example manifests and replace every placeholder. Use real photos, not the example path names. Headless OpenCV needs no display window. Package selection/licensing follows the [OpenCV Python package documentation](https://pypi.org/project/opencv-python-headless/4.13.0.92/); keep its notices when redistributing dependencies.

## Fail-closed output

The only decisions are `candidate` and `unknown`. No nutrition fields, health advice, automatic identity confirmation, or shopping-list changes are produced. Output explicitly includes `exact_identity_verified: false` and `nutrition_lookup_allowed: false`. Candidate metadata is copied from the user-created manifest and has not itself been verified by this program.

The current rules require 24 geometric inliers, 65% inliers, distributed reference coverage, a plausible projected quadrilateral, two matching references, and a substantial margin over competing SKUs. Low texture, repetitive matches, a single supporting reference, competing matches, or known near-variant groups cause abstention. Duplicate/conflicting reference images fail index creation. Holdout evaluation rejects reused capture-session IDs and pixel-identical reference images. It cannot automatically detect every edited or adjacent-frame leak; collection discipline is still required.

## Validation gate before any integration

1. Run the procedural-image unit tests. They prove selected algorithm/control boundaries, **not real product accuracy**.
2. Run the held-out QFC manifest once without tuning on it. Inspect every false candidate.
3. Target **zero false candidates** on at least 20 negative/near-variant/unreadable cases and at least 80% candidate recall on 15 independently captured known-product cases. These are small-demo acceptance targets, not achieved results or statistical safety guarantees. Any false candidate keeps this experiment out of the live identity path.
4. Measure `p95_elapsed_ms` on the presentation laptop. Initial target: below 750 ms for 5 SKUs × 3 references; this is an unmeasured target. If slower or unreliable, use barcode/OCR and drop this optional matcher. Do not lower rejection thresholds just to make the demo pass.

Costs: zero model API calls, no GPU, no model training. Each CLI invocation loads its JSON index; this is not yet a persistent low-latency service. Actual memory and product-image latency must be measured. The prototype caps the index at 12 products and 10 photos each, and resizes images to a 1000-pixel longest edge. JPEG/PNG images and local feature indices stay on this machine. Do not commit collected photos or generated indices; this directory ignores them.

Future integration, if validation passes: use a candidate only to ask “Is this the [variant]?” or to choose which barcode record to verify. The current code is deliberately disconnected from the application.
