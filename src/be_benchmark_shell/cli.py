from __future__ import annotations

import argparse
import json

from .service import evaluate_request


def main() -> int:
    parser = argparse.ArgumentParser(prog="be-benchmark-shell")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--editor-family", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    try:
        payload = evaluate_request(args.sequence, args.editor_family, args.task)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "invalid_request", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
