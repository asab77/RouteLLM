import ast
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

from adaptive_llm_gateway.benchmarks.models import BenchmarkResult, BenchmarkTask

from .models import EvaluationResult
from .normalization import (
    count_sentences, final_answer, flatten_json, normalize_answer, parse_json_response,
    strip_code_fence,
)

EVALUATION_VERSION = "1.1.0"
DEFAULT_THRESHOLDS = {"qa": 1.0, "classification": 1.0, "extraction": 1.0,
                      "json": 1.0, "reasoning": 1.0, "coding": 1.0,
                      "summarization": 0.8}


def _f1(matched: int, expected: int, predicted: int) -> float:
    if expected == predicted == 0:
        return 1.0
    precision = matched / predicted if predicted else 0.0
    recall = matched / expected if expected else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, str):
        return normalize_answer(left) == normalize_answer(right)
    return left == right


class Evaluator(ABC):
    category: ClassVar[str]
    name: ClassVar[str]
    version: ClassVar[str] = EVALUATION_VERSION

    def evaluate(self, task: BenchmarkTask, result: BenchmarkResult) -> EvaluationResult:
        threshold = task.acceptable_threshold
        if threshold is None:
            threshold = DEFAULT_THRESHOLDS[task.category]
        if not result.success or result.response is None:
            return EvaluationResult(
                benchmark_result_id=result.result_id, run_id=result.run_id,
                task_id=task.task_id, model_id=result.model_id,
                evaluator_name="call_failure", evaluator_version=self.version,
                evaluation_status="partial", quality_score=None,
                acceptable_threshold=threshold, acceptable=None,
                reason="Provider infrastructure failed; no candidate response was available to score.",
                details={"error_category": result.error_category or "missing_response"},
            )
        score, reason, components, details = self.score(task, result.response.text)
        score = min(1.0, max(0.0, score))
        incomplete_status = self.status_for(task)
        return EvaluationResult(
            benchmark_result_id=result.result_id, run_id=result.run_id,
            task_id=task.task_id, model_id=result.model_id,
            evaluator_name=self.name, evaluator_version=self.version,
            evaluation_status=incomplete_status or "evaluated",
            quality_score=score, acceptable_threshold=threshold,
            acceptable=None if incomplete_status else score >= threshold, reason=reason,
            component_scores=components, details=details,
        )

    def status_for(self, task: BenchmarkTask) -> str | None:
        return None

    @abstractmethod
    def score(self, task: BenchmarkTask, response: str) -> tuple[float, str, dict[str, float], dict[str, Any]]: ...


class AcceptedAnswerEvaluator(Evaluator):
    name = "accepted_answer_exact_match"

    def accepted(self, task: BenchmarkTask) -> list[str]:
        metadata = task.evaluation_metadata
        answers = metadata.get("accepted_answers", metadata.get("reference"))
        if isinstance(answers, (str, int, float)) and not isinstance(answers, bool):
            answers = [str(answers)]
        if not isinstance(answers, list) or not answers or not all(isinstance(answer, (str, int, float)) for answer in answers):
            raise ValueError(f"Task {task.task_id} requires accepted_answers")
        return [str(answer) for answer in answers]

    def candidate(self, response: str) -> str:
        return response

    def score(self, task, response):
        accepted = self.accepted(task)
        predicted = normalize_answer(self.candidate(response))
        matched = any(predicted == normalize_answer(answer) for answer in accepted)
        return float(matched), ("Matched an explicitly accepted answer." if matched else "Did not match an explicitly accepted answer."), {"exact_match": float(matched)}, {
            "predicted": predicted, "accepted_answers": accepted}


class QAEvaluator(AcceptedAnswerEvaluator):
    category = "qa"


class ClassificationEvaluator(AcceptedAnswerEvaluator):
    category = "classification"
    name = "classification_label_match"

    def accepted(self, task):
        expected = task.evaluation_metadata.get("expected_label", task.evaluation_metadata.get("reference"))
        if not isinstance(expected, str):
            raise ValueError(f"Task {task.task_id} requires expected_label")
        return [expected]


class ReasoningEvaluator(AcceptedAnswerEvaluator):
    category = "reasoning"
    name = "reasoning_final_answer_match"

    def candidate(self, response):
        return final_answer(response)


class ExtractionEvaluator(Evaluator):
    category = "extraction"
    name = "structured_extraction_f1"

    def score(self, task, response):
        expected = task.evaluation_metadata.get("expected", task.evaluation_metadata.get("reference"))
        try:
            predicted = parse_json_response(response)
        except (ValueError, TypeError):
            return 0.0, "Response was not valid JSON.", {"syntax": 0.0, "field_f1": 0.0}, {"valid_json": False}
        expected_fields, predicted_fields = flatten_json(expected), flatten_json(predicted)
        matched = sum(path in predicted_fields and _json_equal(value, predicted_fields[path])
                      for path, value in expected_fields.items())
        f1 = _f1(matched, len(expected_fields), len(predicted_fields))
        return f1, f"Matched {matched} of {len(expected_fields)} expected values with {len(predicted_fields)} predicted values.", {"syntax": 1.0, "field_f1": f1}, {
            "matched_fields": matched, "expected_fields": len(expected_fields), "predicted_fields": len(predicted_fields)}


class JSONEvaluator(Evaluator):
    category = "json"
    name = "json_structure_and_value"

    def score(self, task, response):
        expected = task.evaluation_metadata.get("expected", task.evaluation_metadata.get("reference"))
        try:
            predicted = parse_json_response(response)
        except (ValueError, TypeError):
            return 0.0, "Response was not valid JSON.", {"syntax": 0.0, "schema": 0.0, "values": 0.0}, {"valid_json": False}
        expected_fields, predicted_fields = flatten_json(expected), flatten_json(predicted)
        structural = sum(path in predicted_fields and type(predicted_fields[path]) is type(value)
                         for path, value in expected_fields.items())
        values = sum(path in predicted_fields and _json_equal(value, predicted_fields[path])
                     for path, value in expected_fields.items())
        schema_score = _f1(structural, len(expected_fields), len(predicted_fields))
        value_score = values / len(expected_fields) if expected_fields else float(predicted_fields == {})
        score = (1.0 + schema_score + value_score) / 3
        return score, "Combined JSON syntax, required structure, and expected-value scores.", {
            "syntax": 1.0, "schema": schema_score, "values": value_score}, {
            "expected_fields": len(expected_fields), "predicted_fields": len(predicted_fields), "matched_values": values}


class SummarizationEvaluator(Evaluator):
    category = "summarization"
    name = "required_fact_coverage"

    def status_for(self, task):
        return "requires_semantic_judge" if "semantic_requirements" in task.evaluation_metadata else None

    @staticmethod
    def _fact_groups(task: BenchmarkTask) -> list[list[str]]:
        facts = task.evaluation_metadata.get("required_facts", task.evaluation_metadata.get("reference"))
        if not isinstance(facts, list) or not facts:
            raise ValueError(f"Task {task.task_id} requires required_facts")
        groups = []
        for fact in facts:
            group = fact if isinstance(fact, list) else [fact]
            if not group or not all(isinstance(value, str) and value.strip() for value in group):
                raise ValueError(f"Task {task.task_id} contains an invalid required fact")
            groups.append(group)
        return groups

    def score(self, task, response):
        if "semantic_requirements" in task.evaluation_metadata:
            return self._score_constraints(task, response)
        normalized = normalize_answer(response)
        groups = self._fact_groups(task)
        matched = sum(any(normalize_answer(variant) in normalized for variant in group) for group in groups)
        coverage = matched / len(groups)
        one_sentence = float(count_sentences(response) == 1)
        forbidden = task.evaluation_metadata.get("forbidden_claims", [])
        if not isinstance(forbidden, list) or not all(isinstance(item, str) for item in forbidden):
            raise ValueError(f"Task {task.task_id} contains invalid forbidden_claims")
        violations = [claim for claim in forbidden if normalize_answer(claim) in normalized]
        score = 0.8 * coverage + 0.2 * one_sentence
        if violations:
            score = 0.0
        reason = f"Covered {matched} of {len(groups)} required facts; one-sentence constraint {'met' if one_sentence else 'not met'}."
        if violations:
            reason += " An explicitly forbidden claim was detected."
        return score, reason, {"required_fact_coverage": coverage, "format_compliance": one_sentence,
                               "forbidden_claim_compliance": float(not violations)}, {
            "matched_facts": matched, "required_facts": len(groups), "forbidden_violations": violations}

    def _score_constraints(self, task, response):
        constraints = task.evaluation_metadata.get("deterministic_constraints", {})
        words = len(response.split())
        sentence_count = count_sentences(response)
        checks: dict[str, float] = {}
        if "max_sentences" in constraints:
            checks["sentence_limit"] = float(sentence_count <= constraints["max_sentences"])
        if "max_words" in constraints:
            checks["word_limit"] = float(words <= constraints["max_words"])
        if "required_prefix" in constraints:
            checks["required_prefix"] = float(response.strip().casefold().startswith(
                str(constraints["required_prefix"]).casefold()))
        forbidden = constraints.get("forbidden_strings", [])
        checks["forbidden_string_compliance"] = float(not any(
            str(value).casefold() in response.casefold() for value in forbidden))
        score = sum(checks.values()) / len(checks) if checks else 1.0
        return score, (
            "Deterministic format constraints checked; semantic summary quality requires a future judge."
        ), checks, {"word_count": words, "sentence_count": sentence_count,
                    "semantic_requirements_used_for_scoring": False}


class CodingEvaluator(Evaluator):
    category = "coding"
    name = "constrained_python_static_analysis"
    forbidden_calls = {"__import__", "breakpoint", "compile", "eval", "exec", "input", "open"}

    def status_for(self, task):
        return ("requires_functional_execution"
                if task.evaluation_metadata.get("evaluation_mode") == "static_pending_functional" else None)

    def score(self, task, response):
        metadata = task.evaluation_metadata
        legacy = metadata.get("reference", {})
        function_name = metadata.get("function_name", legacy.get("function") if isinstance(legacy, dict) else None)
        parameters = metadata.get("parameters", 1)
        required_ast = metadata.get("required_ast", [])
        if (not isinstance(function_name, str) or type(parameters) is not int or parameters < 0
                or not isinstance(required_ast, list)
                or not all(isinstance(name, str) for name in required_ast)):
            raise ValueError(f"Task {task.task_id} has invalid coding metadata")
        try:
            tree = ast.parse(strip_code_fence(response))
        except (SyntaxError, ValueError, TypeError):
            return 0.0, "Response was not valid Python syntax.", {"syntax": 0.0, "function": 0.0,
                "signature": 0.0, "safety": 0.0, "required_constructs": 0.0}, {"executed": False}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name]
        function_score = float(len(functions) == 1)
        function = functions[0] if len(functions) == 1 else None
        signature_score = float(function is not None and len(function.args.args) == parameters and
                                not function.args.vararg and not function.args.kwarg)
        forbidden_types = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
                           ast.AsyncFunctionDef, ast.With, ast.AsyncWith, ast.Try, ast.Raise)
        forbidden_call_names = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in self.forbidden_calls
        }
        safety_score = float(
            not any(isinstance(node, forbidden_types) for node in ast.walk(tree))
            and not forbidden_call_names
        )
        known_nodes = {name: getattr(ast, name, None) for name in required_ast}
        if any(node_type is None for node_type in known_nodes.values()):
            raise ValueError(f"Task {task.task_id} contains an unknown required_ast node")
        present = {name: any(isinstance(node, node_type) for node in ast.walk(function or tree))
                   for name, node_type in known_nodes.items()}
        required_score = sum(present.values()) / len(present) if present else 1.0
        components = {"syntax": 1.0, "function": function_score, "signature": signature_score,
                      "safety": safety_score, "required_constructs": required_score}
        score = sum(components.values()) / len(components)
        return score, "Static validation only: syntax, named function, signature, restricted AST, and required constructs.", components, {
            "executed": False, "function_name": function_name, "required_ast_present": present,
            "forbidden_calls": sorted(forbidden_call_names),
            "limitation": "Generated code was not executed; behavioral correctness is not proven."}


EVALUATORS = {evaluator.category: evaluator() for evaluator in (
    QAEvaluator, SummarizationEvaluator, ExtractionEvaluator, ClassificationEvaluator,
    JSONEvaluator, ReasoningEvaluator, CodingEvaluator)}


def evaluator_for(task: BenchmarkTask) -> Evaluator:
    try:
        return EVALUATORS[task.category]
    except KeyError:
        raise ValueError(f"No evaluator registered for category {task.category!r}") from None
