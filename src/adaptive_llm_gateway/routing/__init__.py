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
_FEATURE_EXPORTS = {
    "CategoryProvenance",
    "FeatureCompatibilityStatus",
    "FORBIDDEN_ROUTING_FEATURES",
    "PRODUCTION_FEATURE_GOVERNANCE",
    "ProductionRequestFeatureExtractor",
    "ROUTING_CATEGORY_TAXONOMY_VERSION",
    "RoutingCategory",
    "TRAINING_SERVING_SKEW_AUDIT",
}
_QUALITY_FEATURE_EXPORTS = {
    "CANONICAL_PREDICTIVE_FEATURES",
    "CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION",
    "CANONICAL_TRAINING_SERVING_AUDIT",
    "CanonicalQualityFeatures",
}
_PREDICTOR_EXPORTS = {
    "QualityPredictorArtifactMetadata",
    "SklearnQualityPredictor",
}
_SERVICE_EXPORTS = {"RoutingDecisionService"}
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

__all__ = sorted(
    _POLICY_EXPORTS | _FEATURE_EXPORTS | _QUALITY_FEATURE_EXPORTS
    | _PREDICTOR_EXPORTS | _SERVICE_EXPORTS | _ANALYSIS_EXPORTS
)


def __getattr__(name: str):
    """Keep experiment dependencies out of production imports until requested."""
    if name in _POLICY_EXPORTS:
        return getattr(import_module(".policy", __name__), name)
    if name in _FEATURE_EXPORTS:
        return getattr(import_module(".features", __name__), name)
    if name in _QUALITY_FEATURE_EXPORTS:
        return getattr(import_module(".quality_features", __name__), name)
    if name in _PREDICTOR_EXPORTS:
        return getattr(import_module(".predictor", __name__), name)
    if name in _SERVICE_EXPORTS:
        return getattr(import_module(".service", __name__), name)
    if name in _ANALYSIS_EXPORTS:
        return getattr(import_module(".analysis", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
