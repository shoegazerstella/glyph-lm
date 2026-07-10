"""Run training + eval across multiple seeds for statistical significance."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SEEDS = [42, 123, 456, 789, 1024]


def run_seed(seed: int) -> dict:
    """Train both models with given seed, return eval metrics."""
    print(f"\n{'='*60}")
    print(f"Running seed {seed}")
    print(f"{'='*60}\n")

    # Train
    subprocess.run(
        ["python", "-m", "glyph.train", str(seed)],
        check=True,
    )

    # Eval
    subprocess.run(["python", "-m", "glyph.eval"], check=True)

    # Load results
    with open("results.json") as f:
        results = json.load(f)

    # Archive this run
    archive_dir = Path(f"results_archive/seed_{seed}")
    archive_dir.mkdir(parents=True, exist_ok=True)
    with open(archive_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def aggregate_results(all_results: list[dict]) -> None:
    """Compute mean ± std across seeds."""
    metrics = [
        "final_train_loss",
        "perplexity",
        "bits_per_char",
        "tokens_per_second",
        "chars_per_second",
        "compression_ratio",
    ]

    print(f"\n{'='*60}")
    print("Aggregate Results (mean ± std across seeds)")
    print(f"{'='*60}\n")

    print(f"{'Metric':30}{'raw':>20}{'glyph':>20}")
    print("-" * 70)

    for metric in metrics:
        raw_vals = [r["raw"][metric] for r in all_results if r["raw"][metric] is not None]
        glyph_vals = [r["glyph"][metric] for r in all_results if r["glyph"][metric] is not None]

        if raw_vals and glyph_vals:
            raw_mean, raw_std = np.mean(raw_vals), np.std(raw_vals)
            glyph_mean, glyph_std = np.mean(glyph_vals), np.std(glyph_vals)

            print(
                f"{metric:30}{raw_mean:>8.3f} ± {raw_std:<7.3f}{glyph_mean:>8.3f} ± {glyph_std:<7.3f}"
            )

    # Save aggregate
    def stats(model_key, metric):
        vals = [r[model_key][metric] for r in all_results if r[model_key][metric] is not None]
        if not vals:
            return {"mean": None, "std": None}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    with open("results_aggregate.json", "w") as f:
        json.dump(
            {
                "raw": {m: stats("raw", m) for m in metrics},
                "glyph": {m: stats("glyph", m) for m in metrics},
            },
            f,
            indent=2,
        )

    print("\nSaved results_aggregate.json")


if __name__ == "__main__":
    try:
        seeds = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else SEEDS
    except ValueError as e:
        print(f"Error: All seed arguments must be integers. {e}")
        sys.exit(1)
    print(f"Running {len(seeds)} seeds: {seeds}")

    all_results = []
    for seed in seeds:
        results = run_seed(seed)
        all_results.append(results)

    aggregate_results(all_results)
