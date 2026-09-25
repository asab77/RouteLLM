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
QualityPredictor [future Phase 8C]
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

`TRAINING_SERVING_SKEW_AUDIT` covers every feature in the accepted Phase 7
category/candidate interaction formulation:

| Classification | Features |
| --- | --- |
| `EXACT_MATCH` | candidate ID; prompt characters; approximate input tokens; requested output allowance; constraint/reasoning counts; configured prices; context window; code indicator; temperature capability |
| `COMPATIBLE_WITH_DOCUMENTED_DIFFERENCE` | category; reasoning effort; structured-output requirement; category/candidate interaction |
| `NOT_AVAILABLE_IN_PRODUCTION` | candidate-specific effective output allowance |
| `EXPERIMENT_ONLY` | frozen upstream provider pin |

The category difference is material: Phase 7 always had a category and called its
structured category `json`; production uses `structured_json` and permits explicit
absence. A future artifact adapter must map the legacy value or retrain with the
production vocabulary. Phase 8C must also resolve absent-category behavior,
candidate-specific output budgets, and the experimental provider-pin feature.
Until then, directly loading the Phase 7 formulation would create unsafe skew.
