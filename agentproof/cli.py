"""Command line interface entrypoint for agentproof."""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="agentproof CLI")
    subparsers = parser.add_subparsers(dest="subcommand")

    # dev
    subparsers.add_parser("dev", help="Start development services")

    # sim
    sim_parser = subparsers.add_parser("sim", help="Run simulation benchmarks")
    sim_parser.add_argument("--agent", default="react_baseline", help="Agent under test")
    sim_parser.add_argument("--mode", default="dual", choices=["dual", "no_user", "oracle_plan"])
    sim_parser.add_argument("--tasks", default="all", help="Tasks to run (all, or category/list)")
    sim_parser.add_argument("--trials", type=int, default=1, help="Number of trials per task")
    sim_parser.add_argument("--concurrency", type=int, default=4, help="Concurrency worker count")
    sim_parser.add_argument("--output-dir", default="runs", help="Output directory for runs")

    # report
    report_parser = subparsers.add_parser("report", help="Generate benchmark reports")
    report_parser.add_argument("--run-dir", default=None, help="Specific run directory")

    # validate-judges
    judge_parser = subparsers.add_parser("validate-judges", help="Validate LLM judges")
    judge_parser.add_argument("--data-file", default="data/labels.json", help="Path to labels file")

    # gate
    gate_parser = subparsers.add_parser("gate", help="Run regression gate")
    gate_parser.add_argument(
        "--baseline", default="reports/baseline.json", help="Baseline report file"
    )
    gate_parser.add_argument("--margin", type=float, default=0.05, help="Allowed margin drop")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    print(f"agentproof running subcommand: {args.subcommand}")


if __name__ == "__main__":
    main()
