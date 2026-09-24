import ast
from collections import Counter
from typing import Any

from .models import BenchmarkDataset, BenchmarkTask

V2_CATEGORIES = {"qa", "classification", "extraction", "json", "reasoning", "coding", "summarization"}
V2_DIFFICULTIES = {"easy": 2, "medium": 3, "hard": 3}


def _nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def validate_foundation_v2(dataset: BenchmarkDataset) -> None:
    if dataset.name != "routellm-foundation-v2" or dataset.version != "2.0.0":
        raise ValueError("Foundation V2 identity/version mismatch")
    if len(dataset.tasks) != 56:
        raise ValueError("Foundation V2 must contain exactly 56 tasks")
    categories = Counter(task.category for task in dataset.tasks)
    if categories != Counter({category: 8 for category in V2_CATEGORIES}):
        raise ValueError("Foundation V2 must contain exactly eight tasks per category")
    for category in V2_CATEGORIES:
        difficulties = Counter(task.difficulty for task in dataset.tasks if task.category == category)
        if difficulties != Counter(V2_DIFFICULTIES):
            raise ValueError(f"{category} has invalid difficulty distribution")
    for task in dataset.tasks:
        _validate_task(task)


def _validate_task(task: BenchmarkTask) -> None:
    expected_prefix = f"{task.category}-{task.difficulty}-"
    if not task.task_id.startswith(expected_prefix):
        raise ValueError(f"{task.task_id} does not encode its category/difficulty")
    if task.acceptable_threshold is None:
        raise ValueError(f"{task.task_id} requires an explicit threshold")
    metadata = task.evaluation_metadata
    if metadata.get("difficulty") != task.difficulty:
        raise ValueError(f"{task.task_id} requires a valid evaluation-only difficulty")
    answers = metadata.get("accepted_answers")
    if task.category in {"qa", "reasoning"} and (
            not _nonempty_list(answers) or not all(
                isinstance(answer, (str, int, float)) and not isinstance(answer, bool)
                for answer in answers)):
        raise ValueError(f"{task.task_id} requires valid accepted answers")
    if task.category == "classification" and (
            not isinstance(metadata.get("expected_label"), str)
            or not metadata["expected_label"].strip()):
        raise ValueError(f"{task.task_id} requires an expected label")
    if task.category in {"extraction", "json"} and "expected" not in metadata:
        raise ValueError(f"{task.task_id} requires structured expected data")
    if task.category == "coding":
        if metadata.get("evaluation_mode") != "static_pending_functional":
            raise ValueError(f"{task.task_id} requires pending-functional mode")
        if (not isinstance(metadata.get("function_name"), str)
                or type(metadata.get("parameters")) is not int or metadata["parameters"] < 0):
            raise ValueError(f"{task.task_id} requires function metadata")
        required_ast = metadata.get("required_ast")
        if (not isinstance(required_ast, list) or not all(
                isinstance(name, str) and getattr(ast, name, None) is not None
                for name in required_ast)):
            raise ValueError(f"{task.task_id} requires valid AST metadata")
        tests = metadata.get("functional_tests")
        if not _nonempty_list(tests) or not all(isinstance(case, dict) and
                isinstance(case.get("args"), list) and "expected" in case for case in tests):
            raise ValueError(f"{task.task_id} requires valid functional tests")
    if task.category == "summarization":
        semantic = metadata.get("semantic_requirements")
        if (not _nonempty_list(semantic)
                or not all(isinstance(item, str) and item.strip() for item in semantic)):
            raise ValueError(f"{task.task_id} requires semantic requirements")
        constraints = metadata.get("deterministic_constraints")
        if not isinstance(constraints, dict) or not any(
                key in constraints for key in ("max_sentences", "max_words", "required_prefix", "forbidden_strings")):
            raise ValueError(f"{task.task_id} requires deterministic constraints")
        for key in ("max_sentences", "max_words"):
            if key in constraints and (type(constraints[key]) is not int or constraints[key] <= 0):
                raise ValueError(f"{task.task_id} has an invalid {key}")
        forbidden = constraints.get("forbidden_strings", [])
        if not isinstance(forbidden, list) or not all(isinstance(item, str) for item in forbidden):
            raise ValueError(f"{task.task_id} has invalid forbidden strings")


FOUNDATION_V3_CHANGED_TASK_IDS = frozenset({
    "classification-hard-02", "classification-hard-03",
    "classification-medium-01", "classification-medium-02",
    "extraction-hard-03", "extraction-medium-01", "extraction-medium-03",
    "json-easy-01",
    "qa-hard-01", "qa-hard-02", "qa-hard-03",
    "qa-medium-01", "qa-medium-02", "qa-medium-03",
    "reasoning-easy-01", "reasoning-easy-02",
    "reasoning-hard-01", "reasoning-hard-02", "reasoning-hard-03",
    "reasoning-medium-01", "reasoning-medium-02", "reasoning-medium-03",
    "summarization-easy-01", "summarization-hard-02",
    "summarization-medium-01", "summarization-medium-02",
})


def validate_foundation_v3(dataset: BenchmarkDataset, foundation_v2: BenchmarkDataset) -> None:
    """Validate V3 as the approved token-policy revision of immutable V2."""
    if dataset.name != "routellm-foundation-v3" or dataset.version != "3.0.0":
        raise ValueError("Foundation V3 identity/version mismatch")
    validate_foundation_v2(foundation_v2)
    if len(dataset.tasks) != 56:
        raise ValueError("Foundation V3 must contain exactly 56 tasks")
    categories = Counter(task.category for task in dataset.tasks)
    if categories != Counter({category: 8 for category in V2_CATEGORIES}):
        raise ValueError("Foundation V3 must contain exactly eight tasks per category")
    for category in V2_CATEGORIES:
        difficulties = Counter(task.difficulty for task in dataset.tasks if task.category == category)
        if difficulties != Counter(V2_DIFFICULTIES):
            raise ValueError(f"{category} has invalid Foundation V3 difficulty distribution")
    v2_by_id = {task.task_id: task for task in foundation_v2.tasks}
    v3_by_id = {task.task_id: task for task in dataset.tasks}
    if set(v3_by_id) != set(v2_by_id):
        raise ValueError("Foundation V3 task IDs must exactly match Foundation V2")
    changed = set()
    for task_id, task in v3_by_id.items():
        _validate_task(task)
        reference = v2_by_id[task_id]
        current_content = task.model_dump(exclude={"max_output_tokens"})
        reference_content = reference.model_dump(exclude={"max_output_tokens"})
        if current_content != reference_content:
            raise ValueError(f"{task_id} changes Foundation V2 task/evaluation content")
        if task.max_output_tokens != reference.max_output_tokens:
            changed.add(task_id)
            if task_id not in FOUNDATION_V3_CHANGED_TASK_IDS or task.max_output_tokens != 160:
                raise ValueError(f"{task_id} has an unapproved Foundation V3 output limit")
    if changed != set(FOUNDATION_V3_CHANGED_TASK_IDS):
        raise ValueError("Foundation V3 must contain exactly the 26 approved output-limit changes")
