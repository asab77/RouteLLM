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
