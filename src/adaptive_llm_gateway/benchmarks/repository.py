"""Local experiment store, separate from production PostgreSQL telemetry.

Manifest snapshots controlled prompts/configuration before any calls. Individual
result files are atomically published after each call. No append-only partial JSON.
"""
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID, uuid4

from .models import BenchmarkResult, BenchmarkRun


class BenchmarkRepository(Protocol):
    async def start(self, run: BenchmarkRun) -> None: ...
    async def record(self, result: BenchmarkResult) -> None: ...
    async def finish(self, run_id: UUID, status: Literal["completed", "aborted"]) -> None: ...


class FileBenchmarkRepository:
    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def _write(path: Path, data: str) -> None:
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _start(self, run: BenchmarkRun) -> None:
        directory = self.root / str(run.run_id)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "results").mkdir()
        self._write(directory / "manifest.json", run.model_dump_json(indent=2))
        self._write(directory / "status.json", json.dumps({"status": "running"}))

    async def start(self, run: BenchmarkRun) -> None:
        await asyncio.to_thread(self._start, run)

    async def record(self, result: BenchmarkResult) -> None:
        path = self.root / str(result.run_id) / "results" / f"{result.result_id}.json"
        await asyncio.to_thread(self._write, path, result.model_dump_json(indent=2))

    async def finish(self, run_id: UUID, status: Literal["completed", "aborted"]) -> None:
        await asyncio.to_thread(self._write, self.root / str(run_id) / "status.json",
            json.dumps({"status": status, "finished_at": datetime.now(timezone.utc).isoformat()}))
