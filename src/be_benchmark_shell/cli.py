from __future__ import annotations

import argparse
import json

from .registry import RegistryError
from .service import evaluate_request


def main() -> int:
    parser = argparse.ArgumentParser(prog="be-benchmark-shell")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--editor-family", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    try:
        payload = evaluate_request(args.sequence, args.editor_family, args.task)
    except RegistryError as exc:
        print(json.dumps({"status": "invalid_registry", "error": str(exc)}, indent=2))
        return 2
    except (ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "invalid_request", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
