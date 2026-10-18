# Exact-product lookup decision for the Mac handover

Decision: retain the reviewed-label session cache and OCR fallback. No new database adapter, dependency, credential, or API contract is required for this handover. This is an engineering scope choice, not a claim that OCR is intrinsically more accurate than a correctly matched product database.

## What official sources establish

USDA's API needs a data.gov key. Its normal default is 1,000 requests/hour/IP; the shared example `DEMO_KEY` permits only 30/hour and 50/day. The data is public domain/CC0. Search returns candidates, and details are retrieved by `fdcId`. [USDA API guide](https://fdc.nal.usda.gov/api-guide/)

FDC supports searching a GTIN. An updated record can have a new FDC identifier; monthly branded updates and history may include changed ingredients, nutrition, or serving size. Thus a numeric search hit is not a guarantee that the package in hand matches its record version. [USDA search and dates documentation](https://fdc.nal.usda.gov/help/)

The published OpenAPI `FoodSearchCriteria` has `query`, `dataType`, pagination/sort fields, and `brandOwner`; it does not document a separate exact-GTIN lookup parameter. Returned branded records contain `gtinUpc`. A future adapter should search `query:<scanned digits>, dataType:["Branded"]` and independently require exact normalized GTIN equality before considering a candidate. Never accept the highest similarity score or a close product name. [Official API specification](https://api.nal.usda.gov/fdc/v1/json-spec?api_key=DEMO_KEY)

USDA Branded Foods are manufacturer-supplied food-label data. Foundation Foods have a different analytical provenance. A USDA-hosted branded record is not government approval of a product's healthworthiness or suitability for diabetes. [USDA data-type comparison](https://fdc.nal.usda.gov/data-documentation/)

Open Food Facts currently recommends v3.6. Product reads are limited to 15/minute/IP and searches to 10/minute/IP; reads need an identifying custom User-Agent, not a user API key. Its documentation states that contributed data has no accuracy or completeness assurance. Database reuse is under ODbL, with separate content/image terms. Shared hackathon network limits are a material integration risk. [Official API introduction](https://openfoodfacts.github.io/openfoodfacts-server/api/)

OFF's v3.5 nutrition structure and v3.6 tags structure are breaking schema changes. An adapter must be version-pinned and tested against actual source sets, rather than assuming old flattened `nutriments` fields. [Official schema change log](https://openfoodfacts.github.io/openfoodfacts-server/api/ref-api-and-product-schema-change-log/)

OFF documents that scanners may add or remove leading zeroes, including equivalent UPC-A and EAN-13 forms. Its own lookup normalization can change the returned `code`. Therefore raw string inequality alone can falsely reject the same product. [Official barcode normalization reference](https://openfoodfacts.github.io/openfoodfacts-server/api/ref-barcode-normalization/)

## Narrow cache improvements implemented

- Validate GTIN length/checksum, then compare 14-digit zero-padded identity while preserving original stored barcode digits.
- A nonzero GTIN-14 packaging indicator remains distinct: a case and a single retail unit are not merged.
- Cache only a label already reviewed in the same session. Preserve its original capture timestamp and source. A cache hit makes no new OCR/provider success claim.
- Refuse cache reuse if reviewed records for that GTIN conflict in product identity, market, ingredients, or nutrition. A fresh panel capture is needed.
- Treat two reviewed observations of the same normalized GTIN as revisions/observations, not two alternative products in a comparison.
- OCR and scanner codes may differ only by equivalent zero padding. A genuinely different valid GTIN blocks the candidate.
- Unknown GTINs never resolve to the fictional demo fixtures. Checksum validity proves digit consistency, not package association.

The scanner must expand compressed UPC-E before sending it. This API cannot distinguish an unexpanded eight-digit UPC-E from a declared EAN-8 using digits alone. Exclude unsupported store-internal/coupon codes and use a barcode from the same explicitly captured package. The client must not attach a previously scanned barcode to a new package frame.

## If database lookup is added after handover

Choose one optional USDA adapter first for US packaged foods, provided a real key is available. Keep it off the reviewed-cache fast path. Search by barcode, strictly post-filter `gtinUpc`, then fetch details and preserve `fdcId`, source/date, market, serving unit/basis, and all missing values. Multiple conflicting exact-GTIN revisions or a physical package mismatch must produce a review request. Manufacturer names and descriptions do not break ties safely. Do not substitute generic Foundation food values for an unmatched branded package.

A database record should enter as an unreviewed `database` candidate and never authorize arithmetic or a cart action until the wearer confirms the exact variant and basis. The current `/review` contract only reviews OCR candidates; adding a database adapter therefore needs a coordinated review-state change, source attribution, request deadline/rate-limit behavior, and tests. This extra UI and provenance work is why it was deferred rather than silently inserted into the Mac build.

## Handover environment

No `USDA_FDC_API_KEY` or Open Food Facts configuration is required or read by this build. Existing optional Perceptron/Gemini keys and the local API token are unchanged. Fixture mode runs without keys. Real label OCR still requires Perceptron access, and all hardware/provider verification remains a separate live test.
