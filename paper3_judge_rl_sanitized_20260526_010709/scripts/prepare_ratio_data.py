"""Create position-confound training sets at specified A-preferred ratios.

For every underlying preference pair, the script chooses either the original
orientation (preferred response in slot A) or its physically swapped counterpart
(preferred response in slot B). The number of examples is held fixed.

The intermediate ratios in the paper are single-seed diagnostics. Their results
are non-monotonic and should not be interpreted as a smooth dose-response curve.
"""

import argparse
import json
import random
from pathlib import Path


def create_ratio_data(
    train_path: Path,
    swap_train_path: Path,
    output_path: Path,
    ratio_a: float,
    seed: int = 42,
) -> float:
    """Write a fixed-size set with approximately ``ratio_a`` gold-A items.

    ``train_path`` and ``swap_train_path`` must contain aligned physical pairs in
    opposite orientations. Requiring both files prevents a label-only flip from
    being mistaken for a response swap.
    """
    if not 0.0 <= ratio_a <= 1.0:
        raise ValueError(f"ratio_a must be in [0, 1], got {ratio_a}")

    with train_path.open() as handle:
        original = json.load(handle)
    with swap_train_path.open() as handle:
        swapped = json.load(handle)

    if len(original) != len(swapped):
        raise ValueError(
            "Original and swapped files must contain the same number of "
            f"aligned examples, got {len(original)} and {len(swapped)}"
        )

    rng = random.Random(seed)
    output = [
        orig if rng.random() < ratio_a else swap
        for orig, swap in zip(original, swapped)
    ]

    a_count = sum(item.get("gold_label") == "A" for item in output)
    actual_ratio = a_count / len(output) if output else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    print(
        f"target={ratio_a:.3f} actual={actual_ratio:.3f} "
        f"A={a_count} B={len(output) - a_count} output={output_path}"
    )
    return actual_ratio


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-data",
        type=Path,
        default=project_root / "data/train/rewardbench_train.json",
        help="Original-orientation training JSON (gold A).",
    )
    parser.add_argument(
        "--swap-data",
        type=Path,
        default=project_root / "data/train/rewardbench_train_swap.json",
        help="Aligned physically swapped training JSON (gold B).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data/train",
        help="Directory for generated ratio files.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.60, 0.75, 0.80, 0.90, 0.95],
        help="Preferred-response-in-A ratios to generate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for ratio in args.ratios:
        suffix = int(round(ratio * 100))
        create_ratio_data(
            args.train_data,
            args.swap_data,
            args.output_dir / f"rewardbench_train_ratio{suffix}.json",
            ratio_a=ratio,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
