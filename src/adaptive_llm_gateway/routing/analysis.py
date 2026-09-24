"""Deterministic, offline routing baselines over a frozen routing export."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Iterable, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from adaptive_llm_gateway.benchmarks.features import RequestFeatures
from adaptive_llm_gateway.benchmarks.routing_export import RoutingDatasetRow
from adaptive_llm_gateway.models.schemas import DomainModel

FOUNDATION_V3_RUN_ID = UUID("61707aba-5ab2-4c16-8ec3-eed74555d69c")
FOUNDATION_V2_SHA256 = "dcab5e9b6f1b8bc063555637e06db0d32ec51056959a7586f2cc411949adc964"
FOUNDATION_V3_SHA256 = "64eda8e7388233f645906412a2524982ebfb9c848c42ee24a468ba04c31d16e6"
FOUNDATION_V3_PROTOCOL_SHA256 = "973c46e2dfd7c5739f623ff1c7f2378dc3e77c80e5815e23c975816700d4b861"

ALWAYS_CHEAPEST = "ALWAYS_CHEAPEST"
ALWAYS_STRONGEST = "ALWAYS_STRONGEST"
RANDOM_SEEDED = "RANDOM_SEEDED"
RULE_BASED_V1 = "RULE_BASED_V1"
ORACLE_CHEAPEST_ACCEPTABLE = "ORACLE_CHEAPEST_ACCEPTABLE"
RANDOM_SEED = 20260924

NEMOTRON = "candidate-nemotron-3.5-lightning"
LUNA = "candidate-gpt-6-luna"
GEMINI = "candidate-gemini-3-flash"
SONNET = "candidate-claude-sonnet-5"
EXPECTED_CANDIDATES = frozenset({NEMOTRON, LUNA, GEMINI, SONNET})

RULE_BASED_V1_RULES = (
    "If category is coding or the prompt contains code, choose Sonnet.",
    "Else if category is reasoning or reasoning_indicator_count >= 3, choose Sonnet.",
    "Else if structured output is requested, choose Luna.",
    "Else if category is QA, choose Nemotron.",
    "Else choose Luna.",
)


class CandidateOption(DomainModel):
    """Candidate information knowable before generation."""

    candidate_id: str
    input_cost_per_1m_tokens: Decimal = Field(ge=0)
    output_cost_per_1m_tokens: Decimal = Field(ge=0)
    effective_max_output_tokens: int = Field(gt=0)

    def projected_max_cost(self, approximate_input_tokens: int) -> Decimal:
        with localcontext() as context:
            context.prec = 50
            return (
                Decimal(approximate_input_tokens) * self.input_cost_per_1m_tokens
                + Decimal(self.effective_max_output_tokens) * self.output_cost_per_1m_tokens
            ) / Decimal(1_000_000)

    @property
    def configured_cost_score(self) -> Decimal:
        """Canonical one-input-plus-one-output-token configured price."""
        return self.input_cost_per_1m_tokens + self.output_cost_per_1m_tokens


class PolicyRequest(DomainModel):
    """The complete selection boundary; no result/evaluation fields can enter it."""

    task_id: str
    features: RequestFeatures
    candidates: tuple[CandidateOption, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def candidate_ids_are_unique(self):
        if len({candidate.candidate_id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("Policy request candidate IDs must be unique")
        return self


class RoutingDecision(DomainModel):
    policy: str
    task_id: str
    candidate_id: str | None
    reason: str


class RowObservation(DomainModel):
    """Post-generation fields available only to retrospective evaluation."""

    task_id: str
    candidate_id: str
    features: RequestFeatures
    analysis_difficulty: Literal["easy", "medium", "hard"] | None
    option: CandidateOption
    label_status: Literal["valid", "missing"]
    acceptable: bool | None
    quality_score: float | None = Field(default=None, ge=0, le=1)
    realized_cost_usd: Decimal | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def label_state_is_consistent(self):
        if self.label_status == "valid" and (self.acceptable is None or self.quality_score is None):
            raise ValueError("Valid observations require acceptable and quality")
        if self.label_status == "missing" and (self.acceptable is not None or self.quality_score is not None):
            raise ValueError("Missing observations cannot contain quality labels")
        return self


class FrozenRoutingData(DomainModel):
    run_id: UUID
    dataset_sha256: str
    protocol_sha256: str
    observations: tuple[RowObservation, ...]
    requests: tuple[PolicyRequest, ...]


class RoutingPolicy(Protocol):
    name: str

    def select(self, request: PolicyRequest) -> RoutingDecision: ...


class AlwaysCheapestPolicy:
    name = ALWAYS_CHEAPEST

    def select(self, request: PolicyRequest) -> RoutingDecision:
        candidate = min(
            request.candidates,
            key=lambda item: (
                item.projected_max_cost(request.features.approximate_input_tokens),
                item.candidate_id,
            ),
        )
        return RoutingDecision(
            policy=self.name,
            task_id=request.task_id,
            candidate_id=candidate.candidate_id,
            reason="lowest pre-generation projected maximum cost from configured prices",
        )


class FixedCandidatePolicy:
    name = ALWAYS_STRONGEST

    def __init__(self, candidate_id: str) -> None:
        self.candidate_id = candidate_id

    def select(self, request: PolicyRequest) -> RoutingDecision:
        if self.candidate_id not in {item.candidate_id for item in request.candidates}:
            raise ValueError("Resolved strongest candidate is not eligible")
        return RoutingDecision(
            policy=self.name,
            task_id=request.task_id,
            candidate_id=self.candidate_id,
            reason="fixed candidate resolved from aggregate valid-label performance",
        )


class RandomSeededPolicy:
    name = RANDOM_SEEDED

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.seed = seed

    def select(self, request: PolicyRequest) -> RoutingDecision:
        candidates = sorted(item.candidate_id for item in request.candidates)
        digest = hashlib.sha256(f"{self.seed}:{request.task_id}".encode()).digest()
        candidate_id = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
        return RoutingDecision(
            policy=self.name,
            task_id=request.task_id,
            candidate_id=candidate_id,
            reason=f"deterministic uniform hash draw with seed {self.seed}",
        )


class RuleBasedV1Policy:
    """Benchmark-derived heuristic using only fields in PolicyRequest."""

    name = RULE_BASED_V1

    def select(self, request: PolicyRequest) -> RoutingDecision:
        feature = request.features
        if feature.category == "coding" or feature.contains_code:
            candidate_id, rule = SONNET, RULE_BASED_V1_RULES[0]
        elif feature.category == "reasoning" or feature.reasoning_indicator_count >= 3:
            candidate_id, rule = SONNET, RULE_BASED_V1_RULES[1]
        elif feature.requests_structured_output:
            candidate_id, rule = LUNA, RULE_BASED_V1_RULES[2]
        elif feature.category == "qa":
            candidate_id, rule = NEMOTRON, RULE_BASED_V1_RULES[3]
        else:
            candidate_id, rule = LUNA, RULE_BASED_V1_RULES[4]
        if candidate_id not in {item.candidate_id for item in request.candidates}:
            raise ValueError("Rule-selected candidate is not eligible")
        return RoutingDecision(
            policy=self.name, task_id=request.task_id,
            candidate_id=candidate_id, reason=rule,
        )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_export(path: Path) -> tuple[RoutingDatasetRow, ...]:
    rows = tuple(RoutingDatasetRow.model_validate_json(line) for line in path.read_bytes().splitlines() if line)
    if not rows:
        raise ValueError("Frozen routing dataset is empty")
    return rows


def load_frozen_foundation_v3(root: Path, run_id: UUID = FOUNDATION_V3_RUN_ID) -> FrozenRoutingData:
    directory = root / str(run_id)
    export_path = directory / "routing-dataset.jsonl"
    manifest_path = directory / "manifest.json"
    if not export_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Frozen Foundation V3 routing artifacts are unavailable")
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("run_id") != str(run_id):
        raise ValueError("Foundation V3 run ID mismatch")
    if manifest.get("dataset_sha256") != FOUNDATION_V3_SHA256:
        raise ValueError("Foundation V3 dataset hash mismatch")
    protocol_sha256 = manifest.get("configuration", {}).get("execution_protocol_sha256")
    if protocol_sha256 != FOUNDATION_V3_PROTOCOL_SHA256:
        raise ValueError("Foundation V3 protocol hash mismatch")
    rows = _read_export(export_path)
    if len(rows) != 224 or len({(row.task_id, row.candidate.internal_id) for row in rows}) != 224:
        raise ValueError("Foundation V3 must have 224 unique task/candidate rows")
    task_ids = {row.task_id for row in rows}
    candidates = {row.candidate.internal_id for row in rows}
    if len(task_ids) != 56 or candidates != EXPECTED_CANDIDATES:
        raise ValueError("Foundation V3 must have 56 tasks and the four frozen candidates")
    if Counter(row.label_status for row in rows) != Counter({"valid": 216, "missing": 8}):
        raise ValueError("Foundation V3 label counts differ from the frozen review")
    if any(str(row.run_id) != str(run_id) or row.dataset_sha256 != FOUNDATION_V3_SHA256 for row in rows):
        raise ValueError("Routing row source identity mismatch")
    if any(row.candidate_cost_usd is None for row in rows):
        raise ValueError("Every frozen row must retain authoritative realized cost")

    observations = tuple(RowObservation(
        task_id=row.task_id,
        candidate_id=row.candidate.internal_id,
        features=row.request_features,
        analysis_difficulty=row.analysis_difficulty,
        option=CandidateOption(
            candidate_id=row.candidate.internal_id,
            input_cost_per_1m_tokens=row.candidate.input_cost_per_1m_tokens,
            output_cost_per_1m_tokens=row.candidate.output_cost_per_1m_tokens,
            effective_max_output_tokens=row.effective_max_output_tokens,
        ),
        label_status=row.label_status,
        acceptable=row.acceptable,
        quality_score=row.quality_score,
        realized_cost_usd=row.candidate_cost_usd,
        latency_ms=row.latency_ms,
    ) for row in rows)

    grouped: dict[str, list[RowObservation]] = defaultdict(list)
    for row in observations:
        grouped[row.task_id].append(row)
    requests = []
    for task_id, task_rows in sorted(grouped.items()):
        feature_snapshots = {row.features.model_dump_json() for row in task_rows}
        if len(feature_snapshots) != 1 or len(task_rows) != 4:
            raise ValueError("Every task must have one feature snapshot and four candidates")
        requests.append(PolicyRequest(
            task_id=task_id,
            features=task_rows[0].features,
            candidates=tuple(sorted((row.option for row in task_rows), key=lambda item: item.candidate_id)),
        ))
    return FrozenRoutingData(
        run_id=run_id,
        dataset_sha256=FOUNDATION_V3_SHA256,
        protocol_sha256=protocol_sha256,
        observations=tuple(sorted(observations, key=lambda row: (row.task_id, row.candidate_id))),
        requests=tuple(requests),
    )


def resolve_strongest(observations: Iterable[RowObservation]) -> tuple[str, dict[str, dict[str, object]]]:
    grouped: dict[str, list[RowObservation]] = defaultdict(list)
    for row in observations:
        grouped[row.candidate_id].append(row)
    ranking = []
    details: dict[str, dict[str, object]] = {}
    for candidate_id, rows in sorted(grouped.items()):
        valid = [row for row in rows if row.label_status == "valid"]
        if not valid:
            raise ValueError("Strongest resolution requires valid labels per candidate")
        acceptable_rate = sum(bool(row.acceptable) for row in valid) / len(valid)
        mean_quality = statistics.fmean(row.quality_score for row in valid if row.quality_score is not None)
        cost_score = rows[0].option.configured_cost_score
        details[candidate_id] = {
            "valid_labels": len(valid),
            "acceptable": sum(bool(row.acceptable) for row in valid),
            "acceptable_rate": acceptable_rate,
            "mean_quality": mean_quality,
            "configured_cost_score": str(cost_score),
        }
        ranking.append((-acceptable_rate, -mean_quality, cost_score, candidate_id))
    return min(ranking)[3], details


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _rates_by(rows: list[RowObservation], attribute: str) -> dict[str, dict[str, object]]:
    groups: dict[str, list[RowObservation]] = defaultdict(list)
    for row in rows:
        value = getattr(row, attribute) if attribute != "category" else row.features.category
        groups[str(value)].append(row)
    return {
        key: {
            "valid_labels": len(items),
            "acceptable": sum(bool(item.acceptable) for item in items),
            "acceptable_rate": sum(bool(item.acceptable) for item in items) / len(items),
        }
        for key, items in sorted(groups.items())
    }


def evaluate_decisions(
    observations: Iterable[RowObservation],
    decisions: Iterable[RoutingDecision],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = tuple(observations)
    index = {(row.task_id, row.candidate_id): row for row in rows}
    decisions = tuple(sorted(decisions, key=lambda item: item.task_id))
    total_requests = len({row.task_id for row in rows})
    if len(decisions) != total_requests or len({item.task_id for item in decisions}) != total_requests:
        raise ValueError("Policy must return exactly one decision per request")
    selected: list[RowObservation] = []
    records: list[dict[str, object]] = []
    distribution: Counter[str] = Counter()
    for decision in decisions:
        if decision.candidate_id is None:
            distribution["no_acceptable_candidate"] += 1
            records.append({"decision": decision.model_dump(mode="json"), "retrospective": None})
            continue
        row = index.get((decision.task_id, decision.candidate_id))
        if row is None:
            raise ValueError("Policy selected an unavailable task/candidate row")
        selected.append(row)
        distribution[row.candidate_id] += 1
        records.append({
            "decision": decision.model_dump(mode="json"),
            "retrospective": {
                "label_status": row.label_status,
                "acceptable": row.acceptable,
                "quality_score": row.quality_score,
                "realized_cost_usd": str(row.realized_cost_usd) if row.realized_cost_usd is not None else None,
                "latency_ms": row.latency_ms,
                "analysis_difficulty": row.analysis_difficulty,
                "category": row.features.category,
            },
        })
    valid = [row for row in selected if row.label_status == "valid"]
    costs = [row.realized_cost_usd for row in selected if row.realized_cost_usd is not None]
    cost_per_request = [float(row.realized_cost_usd or 0) for row in selected]
    cost_per_request.extend([0.0] * (total_requests - len(selected)))
    latencies = [row.latency_ms for row in selected]
    qualities = [row.quality_score for row in valid if row.quality_score is not None]
    total_cost = sum(costs, Decimal(0))
    acceptable = sum(bool(row.acceptable) for row in valid)
    summary: dict[str, object] = {
        "policy": decisions[0].policy if decisions else "",
        "total_requests": total_requests,
        "selected_rows": len(selected),
        "valid_selected_labels": len(valid),
        "missing_selected_labels": len(selected) - len(valid),
        "valid_label_coverage": len(valid) / total_requests,
        "acceptable": acceptable,
        "unacceptable": len(valid) - acceptable,
        "acceptable_rate_among_valid": acceptable / len(valid) if valid else None,
        "mean_quality_among_valid": statistics.fmean(qualities) if qualities else None,
        "median_quality_among_valid": statistics.median(qualities) if qualities else None,
        "total_realized_cost_usd": str(total_cost),
        "average_realized_cost_per_request_usd": str(total_cost / total_requests),
        "average_realized_cost_per_selected_request_usd": str(total_cost / len(selected)) if selected else None,
        "median_realized_cost_per_request_usd": str(Decimal(str(statistics.median(cost_per_request)))),
        "p95_realized_cost_per_request_usd": str(Decimal(str(_percentile(cost_per_request, 0.95)))),
        "missing_realized_costs": len(selected) - len(costs),
        "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "model_distribution": {
            candidate_id: {
                "count": count,
                "percentage_of_requests": count / total_requests,
            }
            for candidate_id, count in sorted(distribution.items())
        },
        "category_acceptable_rates": _rates_by(valid, "category"),
        "difficulty_acceptable_rates_analysis_only": _rates_by(valid, "analysis_difficulty"),
    }
    return summary, records


def oracle_cheapest_acceptable(data: FrozenRoutingData) -> tuple[list[RoutingDecision], dict[str, int]]:
    grouped: dict[str, list[RowObservation]] = defaultdict(list)
    for row in data.observations:
        grouped[row.task_id].append(row)
    decisions = []
    observation_counts = Counter()
    for task_id, rows in sorted(grouped.items()):
        fully_observed = all(row.label_status == "valid" for row in rows)
        acceptable = [row for row in rows if row.label_status == "valid" and row.acceptable]
        if not acceptable:
            observation_counts["no_acceptable_candidate"] += 1
            decisions.append(RoutingDecision(
                policy=ORACLE_CHEAPEST_ACCEPTABLE, task_id=task_id, candidate_id=None,
                reason="no_acceptable_candidate",
            ))
            continue
        selected = min(acceptable, key=lambda row: (row.realized_cost_usd, row.candidate_id))
        state = "fully_observed" if fully_observed else "partially_observed"
        observation_counts[state] += 1
        decisions.append(RoutingDecision(
            policy=ORACLE_CHEAPEST_ACCEPTABLE, task_id=task_id,
            candidate_id=selected.candidate_id,
            reason=f"lowest realized cost among valid acceptable rows; {state}",
        ))
    return decisions, dict(observation_counts)


def _comparison(summaries: dict[str, dict[str, object]]) -> dict[str, dict[str, float]]:
    reference = summaries[ALWAYS_STRONGEST]
    reference_cost = Decimal(str(reference["total_realized_cost_usd"]))
    reference_rate = float(reference["acceptable_rate_among_valid"])
    compared = {}
    for policy, summary in summaries.items():
        cost = Decimal(str(summary["total_realized_cost_usd"]))
        rate = float(summary["acceptable_rate_among_valid"])
        compared[policy] = {
            "cost_reduction_vs_always_strongest": float((reference_cost - cost) / reference_cost),
            "acceptable_rate_difference_vs_always_strongest": rate - reference_rate,
            "acceptable_rate_retention_vs_always_strongest": rate / reference_rate,
        }
    return compared


def _report(summaries: dict[str, dict[str, object]], comparison: dict[str, dict[str, float]],
            strongest: str, oracle_state: dict[str, int], opportunity: dict[str, int]) -> str:
    lines = [
        "# Foundation V3 Phase 6 Routing Baselines",
        "",
        "Offline retrospective analysis only. Deployable policies never receive labels, difficulty,",
        "realized cost, latency, output usage, evaluator metadata, or ground truth.",
        "",
        f"ALWAYS_STRONGEST resolves mechanically to `{strongest}` from aggregate valid-label acceptable rate. It is an analysis baseline, not a leakage-free production policy.",
        "RULE_BASED_V1 is benchmark-derived and is not evaluated on unseen data.",
        "ORACLE_CHEAPEST_ACCEPTABLE uses post-generation labels and realized costs and is not deployable.",
        "",
        "| Policy | Coverage | Acceptable rate | Mean quality | Total cost | Average cost/request | P50 latency | P95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in (ALWAYS_CHEAPEST, ALWAYS_STRONGEST, RANDOM_SEEDED, RULE_BASED_V1, ORACLE_CHEAPEST_ACCEPTABLE):
        item = summaries[policy]
        lines.append(
            f"| {policy} | {item['valid_label_coverage']:.3f} | {item['acceptable_rate_among_valid']:.3f} | "
            f"{item['mean_quality_among_valid']:.3f} | ${Decimal(str(item['total_realized_cost_usd'])):.8f} | "
            f"${Decimal(str(item['average_realized_cost_per_request_usd'])):.8f} | "
            f"{item['p50_latency_ms']:.2f} ms | {item['p95_latency_ms']:.2f} ms |"
        )
    lines.extend([
        "",
        "## Rule-based policy",
        "",
        *[f"{index}. {rule}" for index, rule in enumerate(RULE_BASED_V1_RULES, 1)],
        "",
        "## Oracle observation state",
        "",
        f"- Fully observed decisions: {oracle_state.get('fully_observed', 0)}",
        f"- Partially observed decisions: {oracle_state.get('partially_observed', 0)}",
        f"- No acceptable candidate: {oracle_state.get('no_acceptable_candidate', 0)}",
        f"- Tasks with valid-label disagreement: {opportunity['label_disagreement_tasks']}",
        f"- Tasks where the observed oracle selects a model other than ALWAYS_STRONGEST: {opportunity['oracle_non_strongest_selections']}",
        "",
        "## Reference comparisons",
        "",
    ])
    for policy, values in comparison.items():
        lines.append(
            f"- {policy}: cost reduction {values['cost_reduction_vs_always_strongest']:.3%}; "
            f"acceptable-rate difference {values['acceptable_rate_difference_vs_always_strongest']:+.3f}."
        )
    lines.extend([
        "",
        "Foundation V3 contains only 56 unique requests. These results demonstrate observed routing",
        "variation in this frozen dataset and do not establish generalization to production traffic.",
        "",
    ])
    return "\n".join(lines)


def analyze_foundation_v3(root: Path, run_id: UUID = FOUNDATION_V3_RUN_ID) -> dict[str, Path]:
    data = load_frozen_foundation_v3(root, run_id)
    strongest, strongest_details = resolve_strongest(data.observations)
    policies: tuple[RoutingPolicy, ...] = (
        AlwaysCheapestPolicy(),
        FixedCandidatePolicy(strongest),
        RandomSeededPolicy(),
        RuleBasedV1Policy(),
    )
    summaries: dict[str, dict[str, object]] = {}
    all_records: list[dict[str, object]] = []
    for policy in policies:
        decisions = [policy.select(request) for request in data.requests]
        summary, records = evaluate_decisions(data.observations, decisions)
        summaries[policy.name] = summary
        all_records.extend(records)
    oracle_decisions, oracle_state = oracle_cheapest_acceptable(data)
    oracle_summary, oracle_records = evaluate_decisions(data.observations, oracle_decisions)
    oracle_summary["observation_state"] = oracle_state
    summaries[ORACLE_CHEAPEST_ACCEPTABLE] = oracle_summary
    all_records.extend(oracle_records)

    by_task: dict[str, list[RowObservation]] = defaultdict(list)
    for row in data.observations:
        by_task[row.task_id].append(row)
    oracle_selected = {item.task_id: item.candidate_id for item in oracle_decisions}
    opportunity = {
        "label_disagreement_tasks": sum(
            len({row.acceptable for row in rows if row.label_status == "valid"}) > 1
            for rows in by_task.values()
        ),
        "oracle_non_strongest_selections": sum(
            candidate_id is not None and candidate_id != strongest
            for candidate_id in oracle_selected.values()
        ),
    }
    comparison = _comparison(summaries)
    always_cheapest_cost = Decimal(str(summaries[ALWAYS_CHEAPEST]["total_realized_cost_usd"]))
    oracle_cost = Decimal(str(oracle_summary["total_realized_cost_usd"]))
    oracle_opportunity = {
        **opportunity,
        "cost_reduction_vs_always_cheapest": float(
            (always_cheapest_cost - oracle_cost) / always_cheapest_cost),
        "acceptable_request_coverage": oracle_summary["acceptable"] / len(data.requests),
    }
    summary_artifact = {
        "artifact_version": "1.0.0",
        "source": {
            "run_id": str(data.run_id),
            "foundation_v2_sha256": FOUNDATION_V2_SHA256,
            "foundation_v3_sha256": data.dataset_sha256,
            "foundation_v3_protocol_sha256": data.protocol_sha256,
            "rows": len(data.observations),
            "requests": len(data.requests),
            "valid_labels": sum(row.label_status == "valid" for row in data.observations),
            "missing_labels": sum(row.label_status == "missing" for row in data.observations),
            "effective_output_limits_available": all(
                row.option.effective_max_output_tokens > 0 for row in data.observations),
            "realized_costs_available": sum(
                row.realized_cost_usd is not None for row in data.observations),
            "valid_quality_values_available": sum(
                row.quality_score is not None for row in data.observations),
        },
        "selection_boundary": {
            "pre_generation": "PolicyRequest contains request-visible features, candidate configured prices, and effective output limits only.",
            "retrospective": "Labels, difficulty, realized cost/latency, usage, outcomes, and evaluator data are joined only after selection.",
        },
        "random_seed": RANDOM_SEED,
        "rule_based_v1_rules": RULE_BASED_V1_RULES,
        "rule_based_v1_status": "offline benchmark-derived heuristic; not evaluated on unseen data",
        "always_strongest": {
            "resolved_candidate": strongest,
            "definition": "highest acceptable rate among valid labels; ties use mean quality, configured cost score, then candidate ID",
            "analysis_only": True,
            "candidate_metrics": strongest_details,
        },
        "policies": summaries,
        "comparison_to_always_strongest": comparison,
        "routing_opportunity": oracle_opportunity,
    }
    oracle_artifact = {
        "policy": ORACLE_CHEAPEST_ACCEPTABLE,
        "analysis_only": True,
        "limitation": "Observed oracle over valid labels; eight missing labels prevent a perfect global upper-bound claim.",
        "observation_state": oracle_state,
        "summary": oracle_summary,
        "routing_opportunity": oracle_opportunity,
    }
    output = root / str(run_id) / "phase-6"
    paths = {
        "summary": output / "baseline-summary.json",
        "decisions": output / "policy-decisions.jsonl",
        "oracle": output / "oracle-analysis.json",
        "report": output / "phase-6-report.md",
    }
    _atomic_write(paths["summary"], json.dumps(summary_artifact, indent=2, sort_keys=True) + "\n")
    _atomic_write(paths["decisions"], "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in sorted(all_records, key=lambda item: (
            item["decision"]["policy"], item["decision"]["task_id"]))
    ))
    _atomic_write(paths["oracle"], json.dumps(oracle_artifact, indent=2, sort_keys=True) + "\n")
    _atomic_write(paths["report"], _report(summaries, comparison, strongest, oracle_state, opportunity))
    return paths
