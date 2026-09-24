"""Run offline Foundation V3 baseline/oracle analysis."""
import argparse
from pathlib import Path
from uuid import UUID

from .analysis import FOUNDATION_V3_RUN_ID, analyze_foundation_v3


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen routing data without provider calls")
    parser.add_argument("--root", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--run-id", type=UUID, default=FOUNDATION_V3_RUN_ID)
    args = parser.parse_args()
    for name, path in analyze_foundation_v3(args.root, args.run_id).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
