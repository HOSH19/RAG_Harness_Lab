from __future__ import annotations

import argparse
import json

from .backend import backend_status, resolve_backend
from .config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose LLM backend compatibility")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--backend", default=None, help="Override LLM_BACKEND for this check")
    args = parser.parse_args()

    if args.backend:
        import os

        os.environ["LLM_BACKEND"] = args.backend
        from .backend import reload_backend
        from .config import reload_configs

        reload_configs()
        reload_backend()

    status = backend_status()
    if args.json:
        print(json.dumps(status, indent=2))
        return

    settings = get_settings()
    print(f"LLM_BACKEND requested : {status['requested']}")
    print(f"LLM_BACKEND resolved  : {status['resolved']}")
    print(f"platform              : {status['platform']['system']} / {status['platform']['machine']}")
    print(f"effective model       : {status['effective_model_id']}")
    print(f"VLLM_BASE_URL         : {settings.vllm_base_url}")
    print(f"serve via             : {status['profile']['serve_via']}")
    print(f"notes                 : {status['profile']['notes']}")
    print(f"install               : {status['profile']['install_hint']}")
    if status["warnings"]:
        print("warnings:")
        for w in status["warnings"]:
            print(f"  - {w}")
    else:
        print("warnings              : none")

    # Exit non-zero if requested backend is clearly wrong for this machine.
    requested = status["requested"]
    resolved = resolve_backend(requested)
    plat = status["platform"]
    if requested == "metal" and not plat["is_apple_silicon"]:
        raise SystemExit(2)
    if requested == "cuda" and plat["is_apple_silicon"]:
        raise SystemExit(2)
    _ = resolved


if __name__ == "__main__":
    main()