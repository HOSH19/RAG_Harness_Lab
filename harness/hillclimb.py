from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a_cases = {c["id"]: c for c in a.get("cases", [])}
    b_cases = {c["id"]: c for c in b.get("cases", [])}
    ids = sorted(set(a_cases) | set(b_cases))

    rows: list[dict[str, Any]] = []
    for case_id in ids:
        left = a_cases.get(case_id)
        right = b_cases.get(case_id)
        if not left or not right:
            rows.append(
                {
                    "id": case_id,
                    "status": "missing_in_a" if not left else "missing_in_b",
                }
            )
            continue
        delta = right["scores"].get("overall", 0.0) - left["scores"].get("overall", 0.0)
        rows.append(
            {
                "id": case_id,
                "a_overall": left["scores"].get("overall", 0.0),
                "b_overall": right["scores"].get("overall", 0.0),
                "delta": delta,
                "a_hit@5": left["scores"].get("hit@5"),
                "b_hit@5": right["scores"].get("hit@5"),
            }
        )

    agg_keys = sorted(set(a.get("aggregates", {})) | set(b.get("aggregates", {})))
    aggregate_deltas = {
        key: b.get("aggregates", {}).get(key, 0.0) - a.get("aggregates", {}).get(key, 0.0)
        for key in agg_keys
    }
    return {
        "a_suite": a.get("suite"),
        "b_suite": b.get("suite"),
        "a_timestamp": a.get("timestamp"),
        "b_timestamp": b.get("timestamp"),
        "aggregate_deltas": aggregate_deltas,
        "cases": rows,
        "improved": sum(1 for r in rows if r.get("delta", 0) > 1e-9),
        "regressed": sum(1 for r in rows if r.get("delta", 0) < -1e-9),
        "unchanged": sum(1 for r in rows if abs(r.get("delta", 0)) <= 1e-9 and "delta" in r),
    }


def _print_table(comparison: dict[str, Any]) -> None:
    print("Aggregate deltas (B - A):")
    for key, value in comparison["aggregate_deltas"].items():
        print(f"  {key:18s} {value:+.3f}")
    print()
    print(
        f"Cases improved={comparison['improved']} "
        f"regressed={comparison['regressed']} "
        f"unchanged={comparison['unchanged']}"
    )
    print()
    print(f"{'id':28s} {'a':>6} {'b':>6} {'delta':>7}")
    for row in comparison["cases"]:
        if "delta" not in row:
            print(f"{row['id']:28s} {row.get('status')}")
            continue
        print(
            f"{row['id']:28s} {row['a_overall']:6.3f} {row['b_overall']:6.3f} {row['delta']:+7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two harness result JSON files")
    parser.add_argument("a", type=Path, help="Baseline results JSON")
    parser.add_argument("b", type=Path, help="Candidate results JSON")
    parser.add_argument("--json", action="store_true", help="Print raw JSON comparison")
    args = parser.parse_args()

    comparison = compare(load_report(args.a), load_report(args.b))
    if args.json:
        print(json.dumps(comparison, indent=2))
    else:
        _print_table(comparison)


if __name__ == "__main__":
    main()