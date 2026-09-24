"""Explicit opt-in benchmark CLI; defaults to a single task on fake-small."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from adaptive_llm_gateway.bootstrap import configure_gateway, create_development_service
from adaptive_llm_gateway.providers.gateway_config import GatewaySettings
from .models import load_dataset
from .repository import FileBenchmarkRepository
from .runner import BenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("benchmarks/datasets/foundation-v1.json"))
    parser.add_argument("--models", nargs="+", default=["fake-small"])
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--protocol", type=Path,
                        help="Frozen execution protocol containing candidate/task output overrides")
    parser.add_argument("--allow-paid", action="store_true", help="Explicitly allow gateway calls that may incur charges")
    args = parser.parse_args()
    settings = GatewaySettings.from_environment()
    service = create_development_service()
    configure_gateway(service, settings)
    try:
        models = [service.registry.get(model_id) for model_id in args.models]
        if any(model.provider != "fake" for model in models) and not args.allow_paid:
            parser.error("Real model benchmarks require --allow-paid")
        dataset = load_dataset(args.dataset)
        if args.limit <= 0:
            parser.error("--limit must be positive")
        overrides = {}
        configuration = {"gateway_timeout_seconds": settings.timeout_seconds}
        if args.protocol is not None:
            protocol_bytes = args.protocol.read_bytes()
            protocol = json.loads(protocol_bytes)
            if protocol.get("dataset_sha256") != dataset.sha256:
                parser.error("Protocol dataset hash does not match the selected dataset")
            overrides = protocol.get("candidate_task_max_output_tokens", {})
            if not isinstance(overrides, dict):
                parser.error("Protocol output-token overrides are invalid")
            configuration["execution_protocol_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
        runner = BenchmarkRunner(service, FileBenchmarkRepository(args.output),
                                 configuration=configuration,
                                 output_token_overrides=overrides)
        run = asyncio.run(runner.run(dataset, args.models, limit=args.limit))
    except Exception:
        # Avoid emitting connection/provider exception details in a CLI traceback.
        parser.exit(1, "Benchmark failed. Check model IDs, credentials, dataset, and output permissions.\n")
    print(f"Benchmark run: {run.run_id}\nArtifacts: {args.output / str(run.run_id)}")


if __name__ == "__main__":
    main()
