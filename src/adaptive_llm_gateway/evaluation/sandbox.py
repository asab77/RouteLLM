"""Ephemeral Docker execution for untrusted benchmark code.

Generated source is only parsed and executed by the trusted harness inside the
container. The host process treats source as data and never imports or executes it.
"""
from __future__ import annotations

import asyncio
import json
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import Field, JsonValue, model_validator

from adaptive_llm_gateway.models.schemas import DomainModel

FUNCTIONAL_EVALUATOR_VERSION = "1.0.0"
DEFAULT_SANDBOX_IMAGE = ("python:3.12.12-alpine3.22@sha256:"
                         "848ba4413eb897e225159b8fc1b02094576cbae4aa73fc13142608ae2c8c0e32")


class FunctionalTestCase(DomainModel):
    args: tuple[JsonValue, ...]
    expected: JsonValue


class FunctionalEvaluationRequest(DomainModel):
    source: str = Field(min_length=1)
    function_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    parameter_count: int = Field(ge=0)
    tests: tuple[FunctionalTestCase, ...] = Field(min_length=1)
    preserve_inputs: bool = False


FunctionalErrorCategory = Literal[
    "syntax_error", "missing_function", "wrong_signature", "runtime_exception",
    "timeout", "wrong_output", "malformed_return", "restricted_behavior",
    "output_limit", "container_start_failure", "docker_unavailable",
    "container_failure",
]


class FunctionalEvaluationResult(DomainModel):
    execution_status: Literal["passed", "failed", "infrastructure_failure"]
    tests_attempted: int = Field(ge=0)
    tests_passed: int = Field(ge=0)
    tests_failed: int = Field(ge=0)
    total_tests: int = Field(gt=0)
    quality_score: float | None = Field(default=None, ge=0, le=1)
    acceptable: bool | None
    timed_out: bool = False
    error_category: FunctionalErrorCategory | None = None
    evaluator_version: str = FUNCTIONAL_EVALUATOR_VERSION

    @model_validator(mode="after")
    def consistent_result(self):
        if self.execution_status == "infrastructure_failure":
            if (self.quality_score is not None or self.acceptable is not None
                    or self.tests_attempted or self.tests_passed or self.tests_failed
                    or self.error_category is None):
                raise ValueError("Infrastructure failures must remain unscored")
            return self
        expected_score = self.tests_passed / self.total_tests
        if (self.quality_score != expected_score
                or self.tests_failed != self.total_tests - self.tests_passed
                or self.tests_attempted > self.total_tests
                or self.acceptable != (self.tests_passed == self.total_tests)
                or (self.execution_status == "passed") != bool(self.acceptable)):
            raise ValueError("Functional result counts and score are inconsistent")
        return self


class FunctionalSandbox(Protocol):
    async def evaluate(self, request: FunctionalEvaluationRequest) -> FunctionalEvaluationResult: ...


_HARNESS = r'''
import ast
import contextlib
import copy
import io
import json
import sys

MAX_CAPTURE = 8192
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
                   ast.AsyncFunctionDef, ast.With, ast.AsyncWith, ast.Try, ast.Raise)
FORBIDDEN_CALLS = {"__import__", "breakpoint", "compile", "eval", "exec", "input", "open"}
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "range": range,
    "reversed": reversed, "set": set, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "zip": zip,
}

class BoundedWriter(io.StringIO):
    def write(self, value):
        if self.tell() + len(value) > MAX_CAPTURE:
            raise OverflowError("captured output limit")
        return super().write(value)

def emit(status, attempted, passed, failed, total, category=None, timed_out=False):
    score = passed / total
    print(json.dumps({"execution_status": status, "tests_attempted": attempted,
        "tests_passed": passed, "tests_failed": failed, "total_tests": total,
        "quality_score": score, "acceptable": passed == total,
        "timed_out": timed_out, "error_category": category,
        "evaluator_version": "1.0.0"}, separators=(",", ":")))

def reject_category(tree, function_name, parameter_count):
    if any(isinstance(node, FORBIDDEN_NODES) for node in ast.walk(tree)):
        return "restricted_behavior"
    if any(isinstance(node, ast.Attribute) and node.attr.startswith("_") for node in ast.walk(tree)):
        return "restricted_behavior"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            return "restricted_behavior"
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if not isinstance(node, ast.FunctionDef):
            return "restricted_behavior"
        if node.decorator_list or node.returns is not None or any(arg.annotation for arg in node.args.args):
            return "restricted_behavior"
        if node.args.defaults or node.args.kw_defaults:
            return "wrong_signature"
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name]
    if len(matches) != 1:
        return "missing_function"
    args = matches[0].args
    if (len(args.args) != parameter_count or args.posonlyargs or args.kwonlyargs
            or args.vararg or args.kwarg):
        return "wrong_signature"
    return None

def main():
    payload = json.load(sys.stdin)
    total = len(payload["tests"])
    try:
        tree = ast.parse(payload["source"], mode="exec")
    except SyntaxError:
        emit("failed", 0, 0, total, total, "syntax_error")
        return
    category = reject_category(tree, payload["function_name"], payload["parameter_count"])
    if category:
        emit("failed", 0, 0, total, total, category)
        return
    namespace = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(compile(tree, "<candidate>", "exec"), namespace, namespace)
        function = namespace[payload["function_name"]]
    except Exception:
        emit("failed", 0, 0, total, total, "runtime_exception")
        return
    passed = 0
    attempted = 0
    first_error = None
    for case in payload["tests"]:
        attempted += 1
        args = copy.deepcopy(case["args"])
        before = copy.deepcopy(args)
        output = BoundedWriter()
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                actual = function(*args)
        except OverflowError:
            first_error = first_error or "output_limit"
            continue
        except BaseException:
            first_error = first_error or "runtime_exception"
            continue
        if payload["preserve_inputs"] and args != before:
            first_error = first_error or "wrong_output"
            continue
        expected = case["expected"]
        try:
            json.dumps(actual, allow_nan=False)
        except (TypeError, ValueError):
            first_error = first_error or "malformed_return"
            continue
        if type(actual) is not type(expected):
            first_error = first_error or "malformed_return"
            continue
        if actual == expected:
            passed += 1
        else:
            first_error = first_error or "wrong_output"
    failed = total - passed
    emit("passed" if failed == 0 else "failed", attempted, passed, failed, total,
         None if failed == 0 else first_error)

main()
'''


class DockerPythonSandbox:
    """Runs one candidate and its fixtures in one disposable, restricted container."""

    def __init__(self, *, image: str = DEFAULT_SANDBOX_IMAGE, timeout_seconds: float = 3,
                 memory: str = "64m", cpus: float = 0.5, pids_limit: int = 32) -> None:
        if timeout_seconds <= 0 or cpus <= 0 or pids_limit <= 0:
            raise ValueError("Sandbox limits must be positive")
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit

    @property
    def configuration(self) -> dict[str, JsonValue]:
        return {"type": "docker-python", "image": self.image,
                "evaluator_version": FUNCTIONAL_EVALUATOR_VERSION,
                "network": "none", "memory": self.memory, "cpus": self.cpus,
                "pids_limit": self.pids_limit, "timeout_seconds": self.timeout_seconds}

    def command(self, name: str) -> tuple[str, ...]:
        return (
            "docker", "run", "--rm", "--pull=never", "--name", name,
            "--network=none", f"--memory={self.memory}", f"--memory-swap={self.memory}",
            f"--cpus={self.cpus}", f"--pids-limit={self.pids_limit}",
            "--user=65534:65534", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m",
            "--env=PYTHONDONTWRITEBYTECODE=1", "--env=PYTHONHASHSEED=0",
            "-i", self.image, "python", "-I", "-S", "-c", _HARNESS,
        )

    async def _remove(self, name: str) -> None:
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", name,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            if process is not None:
                process.kill()
                await process.wait()
        except (FileNotFoundError, OSError):
            pass

    async def evaluate(self, request: FunctionalEvaluationRequest) -> FunctionalEvaluationResult:
        name = "routellm-eval-" + uuid4().hex
        total = len(request.tests)
        payload = request.model_dump_json().encode()
        try:
            process = await asyncio.create_subprocess_exec(
                *self.command(name), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return self._infrastructure(total, "docker_unavailable")
        except OSError:
            return self._infrastructure(total, "container_start_failure")
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(payload), timeout=self.timeout_seconds)
        except TimeoutError:
            process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=1)
            except TimeoutError:
                pass
            await self._remove(name)
            return FunctionalEvaluationResult(
                execution_status="failed", tests_attempted=total, tests_passed=0,
                tests_failed=total, total_tests=total, quality_score=0,
                acceptable=False, timed_out=True, error_category="timeout")
        except asyncio.CancelledError:
            process.kill()
            await self._remove(name)
            raise
        except OSError:
            process.kill()
            await self._remove(name)
            return self._infrastructure(total, "container_failure")
        finally:
            if process.returncode is not None:
                await self._remove(name)
        if len(stdout) > 65536 or len(stderr) > 65536:
            return FunctionalEvaluationResult(
                execution_status="failed", tests_attempted=total, tests_passed=0,
                tests_failed=total, total_tests=total, quality_score=0,
                acceptable=False, error_category="output_limit")
        if process.returncode != 0:
            lowered = stderr.decode("utf-8", "replace").casefold()
            if process.returncode in (137, 139):
                return FunctionalEvaluationResult(
                    execution_status="failed", tests_attempted=total, tests_passed=0,
                    tests_failed=total, total_tests=total, quality_score=0,
                    acceptable=False, error_category="runtime_exception")
            category: FunctionalErrorCategory = (
                "docker_unavailable" if "cannot connect to the docker daemon" in lowered
                or "permission denied" in lowered else "container_start_failure"
                if "unable to find image" in lowered or "no such image" in lowered
                else "container_failure")
            return self._infrastructure(total, category)
        try:
            return FunctionalEvaluationResult.model_validate_json(stdout)
        except ValueError:
            return self._infrastructure(total, "container_failure")

    @staticmethod
    def _infrastructure(total: int, category: FunctionalErrorCategory) -> FunctionalEvaluationResult:
        return FunctionalEvaluationResult(
            execution_status="infrastructure_failure", tests_attempted=0,
            tests_passed=0, tests_failed=0, total_tests=total, quality_score=None,
            acceptable=None, error_category=category)
