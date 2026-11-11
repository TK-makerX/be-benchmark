from __future__ import annotations
import argparse
import json
from .registry import load_models

def main() -> int:
    parser = argparse.ArgumentParser(prog="be-benchmark")
    parser.add_argument("command", choices=["list-models"])
    args = parser.parse_args()
    payload = load_models()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
