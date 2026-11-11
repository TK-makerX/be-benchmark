from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from be_predictor.candidates import generate_candidates, score_candidates, write_candidate_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate candidate base-editing guides/windows from a DNA sequence")
    parser.add_argument("--sequence", required=True, help="Long DNA sequence or model-ready sequence")
    parser.add_argument("--editor", default="ABE", help="Editor/editor family, e.g. ABE8e, ABE, CBE, BE4")
    parser.add_argument("--strands", default="both", choices=["both", "plus", "minus", "+", "-"])
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument(
        "--score-backend",
        action="append",
        default=[],
        help="Optional scoring backend. Supported in candidate MVP: forecast_be, deepbaseeditor. Can be repeated.",
    )
    parser.add_argument("--cas", default="", help="Optional Cas/PAM variant label for scoring panels")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("BE_RESULTS_DIR", Path.cwd() / "candidate_runs")) / "latest",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON to stdout")
    args = parser.parse_args()

    report = generate_candidates(
        sequence=args.sequence,
        editor=args.editor,
        strands=args.strands,
        max_candidates=args.max_candidates,
    )
    if args.score_backend:
        report = score_candidates(report, editor=args.editor, score_backends=args.score_backend, cas_variant=args.cas)
    paths = write_candidate_outputs(report, args.output_dir)
    report["output_paths"] = paths
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({
            "returned_candidates": report["returned_candidates"],
            "total_candidates_before_limit": report["total_candidates_before_limit"],
            "scoring_backend_status": report.get("scoring_backend_status", {}),
            "output_paths": paths,
        }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

