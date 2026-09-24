import argparse
import asyncio
from pathlib import Path
from uuid import UUID

from adaptive_llm_gateway.errors import EvaluationArtifactError, EvaluationNotFoundError
from adaptive_llm_gateway.providers.gateway_config import GatewaySettings, SEMANTIC_JUDGE_MODELS
from adaptive_llm_gateway.providers.vercel import VercelGatewayProvider

from .judge import VercelSemanticJudge
from .sandbox import DEFAULT_SANDBOX_IMAGE, DockerPythonSandbox
from .service import EvaluationService


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an existing benchmark run offline.")
    parser.add_argument("--run-id", type=UUID, required=True)
    parser.add_argument("--root", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--summary-only", action="store_true",
                        help="Read an existing summary without reevaluating raw results")
    parser.add_argument("--functional-docker", action="store_true",
                        help="Run coding fixtures in the restricted Docker sandbox")
    parser.add_argument("--sandbox-image", default=DEFAULT_SANDBOX_IMAGE)
    parser.add_argument("--semantic-judge-model", choices=[model.model_id for model in SEMANTIC_JUDGE_MODELS],
                        help="Explicit Vercel judge model (may incur provider charges)")
    parser.add_argument("--allow-paid-judge", action="store_true",
                        help="Required acknowledgement before any real semantic judge calls")
    args = parser.parse_args()
    if args.summary_only and (args.functional_docker or args.semantic_judge_model or args.allow_paid_judge):
        parser.error("--summary-only cannot invoke evaluators")
    if bool(args.semantic_judge_model) != bool(args.allow_paid_judge):
        parser.error("real semantic judging requires both --semantic-judge-model and --allow-paid-judge")
    judge = None
    if args.semantic_judge_model:
        settings = GatewaySettings.from_environment()
        if settings.api_key is None:
            parser.error("AI_GATEWAY_API_KEY is required for explicit real semantic judging")
        model = next(model for model in SEMANTIC_JUDGE_MODELS if model.model_id == args.semantic_judge_model)
        judge = VercelSemanticJudge(VercelGatewayProvider(model, settings),
                                    provider_name="vercel", model_id=model.provider_model_name)
    service = EvaluationService(args.root,
        functional_sandbox=(DockerPythonSandbox(image=args.sandbox_image)
                            if args.functional_docker else None),
        semantic_judge=judge)
    try:
        summary = asyncio.run(service.summary(args.run_id) if args.summary_only
                              else service.evaluate(args.run_id))
    except (EvaluationArtifactError, EvaluationNotFoundError, ValueError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
