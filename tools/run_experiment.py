#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.registry import get_experiment, list_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run StereACL experiments by numeric id."
    )
    parser.add_argument("--list", action="store_true", help="List available experiments and exit.")
    parser.add_argument("--id", help="Experiment id, e.g. 01 or 1.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for running experiment scripts.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the experiment script. Use '--' before forwarded args.",
    )
    return parser.parse_args()


def print_experiments() -> None:
    for spec in list_experiments():
        print(f"{spec.id}  {spec.slug:<24} {spec.title}")


def main() -> None:
    args = parse_args()
    if args.list:
        print_experiments()
        return
    if not args.id:
        raise SystemExit("Provide --id <experiment_id> or use --list.")

    spec = get_experiment(args.id)
    script_path = PROJECT_ROOT / spec.script
    if not script_path.exists():
        raise FileNotFoundError(f"Experiment script not found: {script_path}")

    forwarded = args.script_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    cmd = [args.python, str(script_path), *forwarded]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()

