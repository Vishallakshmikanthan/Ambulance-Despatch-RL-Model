from __future__ import annotations

from typing import Any, Dict


def generate_report(results: Dict[str, Any]) -> None:
    """Print a formatted performance comparison report."""
    base = results["baseline"]
    adv = results["advanced"]

    print("\n=== SYSTEM PERFORMANCE REPORT ===")

    print("\nBaseline:")
    print(f"  Reward: {base['reward']:.2f}")
    print(f"  Served: {base['served']}")

    print("\nAdvanced System:")
    print(f"  Reward: {adv['reward']:.2f}")
    print(f"  Served: {adv['served']}")

    print("\nImprovement:")
    print(f"  Reward Gain: {results['improvement']:.2f}")
    pct = (results["improvement"] / abs(base["reward"])) * 100 if base["reward"] != 0 else 0.0
    print(f"  Reward Gain (%): {pct:.1f}%")
