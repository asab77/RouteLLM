# Cost-aware routing policy

RouteLLM keeps quality prediction and routing policy as separate concepts.

The **quality predictor** estimates
`P(candidate produces an acceptable response | request)` for every eligible model.
It receives governed, request-visible features and an optional validated category
signal. The predictor interface does not prescribe how category is obtained.

The **cost-aware policy** receives those probabilities, exact pre-generation
projected costs, and a caller-supplied quality threshold. It selects the cheapest
eligible candidate predicted to meet the threshold. If none meets it, it falls
back deterministically to the highest predicted probability, then lower projected
cost, then stable model ID.

Eligibility remains an upstream registry/capability concern. Projected cost uses
the request's approximate input tokens, the effective maximum output allowance,
and configured Decimal token prices. Selection never uses realized output tokens,
latency, quality, or provider cost.

Phase 7 validated the category/candidate interaction logistic-regression
formulation offline. Phase 8A does not load or serialize that trained model, call a
provider, or connect routing to the inference endpoint. The production threshold
and production category acquisition method remain unresolved. Post-generation
validation and escalation belong to Phase 9.

`RoutingDecision` exposes privacy-safe metadata for future telemetry: selected
model, predicted acceptability, configured threshold, threshold state, fallback
state, projected selected cost, eligible and qualifying counts, and a typed reason.
It does not contain candidate probability vectors, prompts, responses, or
reasoning text. Phase 8A does not persist this metadata.

## Production request features

Phase 8B adds a pure `ProductionRequestFeatureExtractor`:

```text
Raw Request
    |
    v
Feature Extractor
    |
    v
RoutingRequestFeatures
    |
    v
    SklearnQualityPredictor [Phase 8C, explicitly loaded]
    |
    v
Candidate predictions
    |
    v
Cost-Aware Policy [Phase 8A]
```

It deterministically derives user/system character lengths, message count, system
prompt presence, a whitespace-based approximate input-token count, the frozen
Phase 7 code/reasoning/constraint indicators, explicit structured-output state,
and the requested output allowance. The estimate is provider-independent and is
compatible with the approximate-input value consumed by projected pricing.

Structured output is true only when the internal caller explicitly supplies that
requirement. Prompt text mentioning JSON does not activate it. Extraction performs
no network access, provider lookup, embedding, ML inference, or model execution.

## Category contract

The current `phase7-v1` routing taxonomy is `classification`, `coding`,
`extraction`, `structured_json`, `qa`, `reasoning`, and `summarization`. It is a
versioned current taxonomy inherited from the controlled benchmark, not a permanent
domain ontology.

A category hint is optional advisory metadata. If supplied, it must be a taxonomy
value and provenance is `CLIENT_HINT`; it is never treated as ground truth. If it
is omitted, category remains `None` with `ABSENT` provenance. `INFERRED` is reserved
for a future component and the Phase 8B extractor never emits it. There is no
default category, category classifier, production ML artifact, or public API field.

## Feature governance and leakage

`PRODUCTION_FEATURE_GOVERNANCE` machine-documents every field's type, source,
semantics, required state, client control, derivation, and leakage status.
`FORBIDDEN_ROUTING_FEATURES` prevents response quality, actual usage, realized
latency/cost, evaluator outcomes, benchmark labels, ground truth, Foundation
difficulty, errors, and other post-generation outcomes from entering the contract.

## Training/serving compatibility

The historical Phase 8B `TRAINING_SERVING_SKEW_AUDIT` recorded every incompatibility
in the accepted Phase 7 category/candidate interaction formulation before the
Phase 8C correction:

| Classification | Features |
| --- | --- |
| `EXACT_MATCH` | candidate ID; prompt characters; approximate input tokens; requested output allowance; constraint/reasoning counts; configured prices; context window; code indicator; temperature capability |
| `COMPATIBLE_WITH_DOCUMENTED_DIFFERENCE` | category; reasoning effort; structured-output requirement; category/candidate interaction |
| `NOT_AVAILABLE_IN_PRODUCTION` | candidate-specific effective output allowance |
| `EXPERIMENT_ONLY` | frozen upstream provider pin |

That audit prevented direct loading of the historical Phase 7 formulation. The
following Phase 8C contract resolves every predictive incompatibility while
preserving the historical audit for traceability.

## Phase 8C canonical predictor

Phase 8C resolves that skew with the approved `INTERACTION_NO_PROVIDER_PIN`
formulation. The historical Phase 7 formulation remains unchanged. Provider pin is
absent from the production schema, registry, preprocessing, artifact, and
prediction inputs following the controlled Phase 8C-0 ablation.

The version `1.0.0` `CanonicalQualityFeatures` contract contains canonical category,
candidate identity, normalized reasoning effort, governed request measurements,
requested and effective output allowances, configured prices, context window,
code/structured-output indicators, and temperature capability. The shared matrix
layer derives the category/candidate interaction. It excludes category provenance,
Foundation difficulty, targets, evaluator data, response data, actual usage,
latency, realized cost, errors, reasoning tokens, provider payloads, and ground
truth.

Training maps `json` to `structured_json`; production already uses
`structured_json`. All other categories map directly. Production category absence
raises `MissingRoutingCategoryError`. `CLIENT_HINT` and future `INFERRED` provenance
values do not enter the vector, and Phase 8C does not infer categories.

Structured output normalizes explicit benchmark configuration and the explicit
internal production requirement to one boolean. Reasoning effort normalizes enum,
string, and omitted values without model-slug rules. Effective output allowance is
resolved from the request plus typed `ModelConfig.output_token_policy` category
overrides. The resolver has no candidate-ID cases.

The deterministic skew audit covers all 16 predictive fields: 11 are
`EXACT_MATCH`, five are `CANONICALIZED_MATCH`, and zero are `UNRESOLVED`. Training
and production adapters produce equal canonical objects and equal transformed
vectors for all 224 frozen request/candidate rows.

## Artifact and compatibility boundary

Run the local, provider-free build:

```sh
python -m adaptive_llm_gateway.routing.train_predictor \
  --root benchmark-results \
  --output artifacts/routing-quality/interaction-no-provider-pin-v1
```

The build verifies frozen hashes and compatibility, excludes eight missing labels,
fits the unchanged logistic-regression configuration on 216 valid labels, writes a
versioned metadata document and checksummed sklearn pipeline, then reloads it for a
prediction smoke test. Binary output is ignored by Git and is not downloaded or
rebuilt automatically at startup.

`SklearnQualityPredictor` validates trusted local metadata and checksum before
unpickling the application-owned pipeline. It rejects missing, corrupt, or
incompatible artifacts with typed failures. Python pickle is unsafe for untrusted
input; the loader must never receive user uploads or arbitrary downloaded files.

Artifact metadata records the four trained candidate identities. An arbitrary
model may still be registered, but it cannot be predicted by this artifact.
Unsupported identities raise `UnsupportedPredictorCandidateError`; collecting
labels, grouped offline validation, and retraining are required before a new model
becomes ML-routable.

The predictor returns candidate-associated acceptability probabilities in input
order. It does not choose a model, contain a quality threshold, call providers,
write telemetry, or load Foundation data after artifact construction. The Phase 8A
policy remains the only component that can combine probabilities, projected cost,
and a caller-supplied threshold. No production threshold has been selected, and
automatic routing remains disconnected from `/v1/inference`.

The final full-data fit is deployment construction, not new generalization
evidence. Scientific evidence remains the grouped OOF Phase 7 and Phase 8C-0
experiments; training-set scores are neither generated nor reported as evidence.

## Offline production-path composition

Phase 8D adds `RoutingDecisionService`, which composes the existing extractor,
predictor, canonical projected-cost calculation, and Phase 8A policy. The caller
provides the candidate set and quality threshold; an optional upstream eligibility
filter may narrow the set. Registry eligibility remains distinct from artifact
compatibility, so an eligible model unknown to the loaded artifact fails explicitly.

The service performs no provider request, response validation, escalation, category
inference, threshold selection, HTTP integration, or telemetry write. It preserves
the public requirement for an explicit `model_id` and is loaded only when explicitly
constructed, keeping normal API startup independent of sklearn and local artifacts.

The offline validator uses Foundation V3 only as a fixture source and compares the
composed path with direct invocation of the same full-fit pipeline and Phase 8A
policy. It checks probability and decision parity, repeatability, input-order
independence, and threshold-set invariants. Its generated report is ignored. The
reported selection and cost distributions describe full-fit integration behavior;
they must not be interpreted as OOF quality or generalization evidence.
