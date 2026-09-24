import asyncio
import subprocess
from pathlib import Path

import pytest

from adaptive_llm_gateway.evaluation.sandbox import (
    DEFAULT_SANDBOX_IMAGE, DockerPythonSandbox, FunctionalEvaluationRequest,
    FunctionalTestCase,
)
from adaptive_llm_gateway.benchmarks.models import load_dataset


def request(source, *, preserve=False):
    return FunctionalEvaluationRequest(source=source, function_name="clamp", parameter_count=3,
        tests=(FunctionalTestCase(args=(5, 0, 10), expected=5),
               FunctionalTestCase(args=(-2, 0, 10), expected=0),
               FunctionalTestCase(args=(20, 0, 10), expected=10)),
        preserve_inputs=preserve)


def test_docker_command_has_required_security_controls_and_no_mounts_or_secrets(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "must-not-enter-command")
    command = DockerPythonSandbox().command("controlled-name")
    joined = " ".join(command)
    assert "--network=none" in command
    assert "--read-only" in command and "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges:true" in command
    assert "--user=65534:65534" in command
    assert any(value.startswith("--memory=") for value in command)
    assert any(value.startswith("--cpus=") for value in command)
    assert any(value.startswith("--pids-limit=") for value in command)
    assert any(value.startswith("--tmpfs=/tmp:") for value in command)
    assert "--rm" in command and "--pull=never" in command
    assert "--mount" not in command and "-v" not in command
    assert "docker.sock" not in joined and "/Users/" not in joined
    assert "must-not-enter-command" not in joined and "AI_GATEWAY_API_KEY" not in joined


@pytest.mark.asyncio
async def test_docker_unavailable_is_unscored_infrastructure_failure(monkeypatch):
    async def unavailable(*args, **kwargs):
        raise FileNotFoundError
    monkeypatch.setattr(asyncio, "create_subprocess_exec", unavailable)
    outcome = await DockerPythonSandbox().evaluate(request("def clamp(value, low, high): return value"))
    assert outcome.execution_status == "infrastructure_failure"
    assert outcome.error_category == "docker_unavailable"
    assert outcome.quality_score is outcome.acceptable is None
    assert outcome.tests_failed == 0


def docker_ready():
    try:
        return subprocess.run(["docker", "image", "inspect", DEFAULT_SANDBOX_IMAGE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5).returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


@pytest.fixture
def docker_sandbox():
    if not docker_ready():
        pytest.skip("Pinned Docker sandbox image is unavailable")
    return DockerPythonSandbox()


@pytest.mark.asyncio
@pytest.mark.docker_sandbox
@pytest.mark.parametrize(("source", "status", "category", "passed"), [
    ("def clamp(value, low, high):\n    if value < low: return low\n    if value > high: return high\n    return value",
     "passed", None, 3),
    ("def clamp(value, low, high):\n    return min(max(value, low), high - 1)",
     "failed", "wrong_output", 2),
    ("def clamp(value, low, high", "failed", "syntax_error", 0),
    ("def clamp(value, low, high):\n    return 1 / 0", "failed", "runtime_exception", 0),
    ("def clamp(value, low, high):\n    return (value,)", "failed", "malformed_return", 0),
    ("def other(value, low, high):\n    return value", "failed", "missing_function", 0),
    ("def clamp(value):\n    return value", "failed", "wrong_signature", 0),
    ("def clamp(value, low, high):\n    open('/tmp/x', 'w')\n    return value", "failed", "restricted_behavior", 0),
    ("import socket\ndef clamp(value, low, high):\n    return value", "failed", "restricted_behavior", 0),
    ("import subprocess\ndef clamp(value, low, high):\n    return value", "failed", "restricted_behavior", 0),
])
async def test_docker_sandbox_categories(docker_sandbox, source, status, category, passed):
    outcome = await docker_sandbox.evaluate(request(source))
    assert outcome.execution_status == status and outcome.error_category == category
    assert outcome.tests_passed == passed
    assert outcome.quality_score == passed / 3
    assert outcome.acceptable is (status == "passed")


@pytest.mark.asyncio
@pytest.mark.docker_sandbox
async def test_docker_sandbox_enforces_wall_timeout(docker_sandbox):
    outcome = await DockerPythonSandbox(timeout_seconds=1).evaluate(
        request("def clamp(value, low, high):\n    while True: pass"))
    assert outcome.execution_status == "failed" and outcome.error_category == "timeout"
    assert outcome.timed_out and outcome.quality_score == 0


@pytest.mark.asyncio
@pytest.mark.docker_sandbox
async def test_docker_sandbox_detects_required_input_immutability(docker_sandbox):
    item = FunctionalEvaluationRequest(source="def sort_values(items):\n    items.sort()\n    return items",
        function_name="sort_values", parameter_count=1,
        tests=(FunctionalTestCase(args=([3, 1, 2],), expected=[1, 2, 3]),),
        preserve_inputs=True)
    outcome = await docker_sandbox.evaluate(item)
    assert outcome.execution_status == "failed" and outcome.error_category == "wrong_output"
    assert outcome.tests_passed == 0


CORRECT_V2_SOLUTIONS = {
    "coding-easy-01": "def clamp(value, low, high):\n    if value < low: return low\n    if value > high: return high\n    return value",
    "coding-easy-02": "def count_nonempty(values):\n    return sum(1 for value in values if value.strip())",
    "coding-medium-01": "def group_totals(rows):\n    totals = {}\n    for category, value in rows:\n        totals[category] = totals.get(category, 0) + value\n    return totals",
    "coding-medium-02": "def parse_ranges(text):\n    values = set()\n    for part in text.split(','):\n        if not part: continue\n        if '-' in part:\n            start, end = part.split('-')\n            values.update(range(int(start), int(end) + 1))\n        else:\n            values.add(int(part))\n    return sorted(values)",
    "coding-medium-03": "def merge_counts(left, right):\n    merged = {}\n    for key in set(left) | set(right):\n        value = left.get(key, 0) + right.get(key, 0)\n        if value != 0: merged[key] = value\n    return merged",
    "coding-hard-01": "def merge_intervals(intervals):\n    merged = []\n    for start, end in sorted(intervals):\n        if merged and start <= merged[-1][1] + 1:\n            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged",
    "coding-hard-02": "def shortest_hops(graph, start, end):\n    if start == end: return 0\n    queue = [(start, 0)]\n    seen = {start}\n    while queue:\n        node, distance = queue.pop(0)\n        for neighbor in graph.get(node, []):\n            if neighbor == end: return distance + 1\n            if neighbor not in seen:\n                seen.add(neighbor)\n                queue.append((neighbor, distance + 1))\n    return None",
    "coding-hard-03": "def max_nonadjacent(values):\n    previous = current = 0\n    for value in values:\n        previous, current = current, max(current, previous + value)\n    return current",
}


@pytest.mark.asyncio
@pytest.mark.docker_sandbox
async def test_all_locked_v2_hidden_functional_fixtures_pass_correct_solutions(docker_sandbox):
    tasks = [task for task in load_dataset(Path("benchmarks/datasets/foundation-v2.json")).tasks
             if task.category == "coding"]
    assert {task.task_id for task in tasks} == set(CORRECT_V2_SOLUTIONS)
    for task in tasks:
        metadata = task.evaluation_metadata
        item = FunctionalEvaluationRequest(source=CORRECT_V2_SOLUTIONS[task.task_id],
            function_name=metadata["function_name"], parameter_count=metadata["parameters"],
            tests=tuple(FunctionalTestCase.model_validate(case)
                        for case in metadata["functional_tests"]),
            preserve_inputs="mutat" in task.prompt.casefold())
        outcome = await docker_sandbox.evaluate(item)
        assert outcome.execution_status == "passed", (task.task_id, outcome)
