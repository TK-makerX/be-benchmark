from __future__ import annotations

import argparse

from be_predictor.service import analyze, format_text_report, to_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark-informed base-editor prediction router MVP")
    parser.add_argument("--sequence", required=True, help="DNA sequence, e.g. 20nt protospacer or 20nt+PAM")
    parser.add_argument("--editor", default="ABE", help="Editor or editor family, e.g. ABE, ABE8e, CBE, BE4")
    parser.add_argument(
        "--task",
        default="auto",
        choices=[
            "auto",
            "scalar_efficiency",
            "outcome_profile",
            "position_profile",
            "high_low_screening",
            "design_recommendation",
        ],
        help="Prediction task to route",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--run-models", action="store_true", help="Invoke implemented real model backends")
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help="Backend id to run when --run-models is used. Can be repeated. Default: all implemented feasible connectors",
    )
    parser.add_argument("--cell", default="", help="Optional cell/context label for models such as BE-Hive")
    parser.add_argument("--cas", default="", help="Optional Cas/PAM variant label, e.g. SpRY, NG, SpCas9")
    parser.add_argument("--outcome", default="", help="Optional edited outcome sequence for outcome-specific models such as EPCNNBE/Yachie")
    args = parser.parse_args()

    result = analyze(
        sequence=args.sequence,
        editor=args.editor,
        task=args.task,
        run_models=args.run_models,
        backend_ids=args.backends,
        cell_context=args.cell,
        cas_variant=args.cas,
        reference_outcome=args.outcome,
    )
    if args.json:
        print(to_json(result))
    else:
        print(format_text_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
