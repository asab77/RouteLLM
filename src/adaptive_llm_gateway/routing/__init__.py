"""Routing boundaries with lazy compatibility exports for offline experiments."""

from importlib import import_module

_POLICY_EXPORTS = {
    "CandidateEligibilityFilter",
    "CandidatePrediction",
    "CostAwareRoutingPolicy",
    "ModelAcceptabilityPrediction",
    "QualityPredictor",
    "RoutingDecision",
    "RoutingDecisionReason",
    "RoutingRequestFeatures",
}
_ANALYSIS_EXPORTS = {
    "ALWAYS_CHEAPEST",
    "ALWAYS_STRONGEST",
    "ORACLE_CHEAPEST_ACCEPTABLE",
    "RANDOM_SEED",
    "RANDOM_SEEDED",
    "RULE_BASED_V1",
    "AlwaysCheapestPolicy",
    "FixedCandidatePolicy",
    "RandomSeededPolicy",
    "RuleBasedV1Policy",
    "analyze_foundation_v3",
    "load_frozen_foundation_v3",
}

__all__ = sorted(_POLICY_EXPORTS | _ANALYSIS_EXPORTS)


def __getattr__(name: str):
    """Keep experiment dependencies out of production imports until requested."""
    if name in _POLICY_EXPORTS:
        return getattr(import_module(".policy", __name__), name)
    if name in _ANALYSIS_EXPORTS:
        return getattr(import_module(".analysis", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
