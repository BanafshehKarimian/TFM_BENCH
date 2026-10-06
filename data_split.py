#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from tfmbench.datasets.talent import load_talent_dataset


def make_regression_bins(
    y: np.ndarray,
    n_bins: int = 20,
) -> np.ndarray:
    """
    Convert continuous regression targets into quantile bins.

    IMPORTANT:
    The bins are ONLY used for stratified sampling.
    The original continuous y values are never modified.
    """

    y = np.asarray(y).reshape(-1)

    if len(y) == 0:
        raise ValueError("Empty regression target.")

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "Regression target contains NaN or Inf values."
        )

    # qcut tries to put approximately the same number
    # of samples in each bin.
    #
    # duplicates='drop' is important for targets where
    # many rows have exactly the same value.
    bins = pd.qcut(
        y,
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    bins = np.asarray(
        bins,
        dtype=np.int64,
    )

    n_actual_bins = len(
        np.unique(bins)
    )

    print(
        f"[Sampling] Regression bins: "
        f"{n_actual_bins}"
    )

    return bins



def make_stratified_order(
    strata: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Create ONE deterministic ordering of all rows such that
    prefixes approximately preserve stratum proportions.

    Example:

        order[:100_000]
            subset of
        order[:500_000]
            subset of
        order[:1_000_000]

    Therefore all requested subsets are nested.
    """

    strata = np.asarray(
        strata
    ).reshape(-1)

    n_samples = len(strata)

    rng = np.random.default_rng(
        seed
    )

    priorities = np.empty(
        n_samples,
        dtype=np.float64,
    )

    unique_strata = np.unique(
        strata
    )

    print(
        f"[Sampling] Number of strata: "
        f"{len(unique_strata)}"
    )

    for stratum in unique_strata:

        indices = np.flatnonzero(
            strata == stratum
        )

        # Random ordering inside this class/bin
        indices = rng.permutation(
            indices
        )

        n = len(indices)

        # Spread each stratum approximately uniformly
        # across [0, 1).
        #
        # This is what lets prefixes retain roughly the
        # same class/bin proportions.
        positions = (
            np.arange(
                n,
                dtype=np.float64,
            )
            + rng.random(n)
        ) / n

        priorities[
            indices
        ] = positions

    order = np.argsort(
        priorities,
        kind="stable",
    )

    return order


def make_nested_indices(
    y: np.ndarray,
    task: str,
    sample_sizes: Sequence[int],
    seed: int = 42,
    regression_bins: int = 20,
) -> dict[int, np.ndarray]:
    """
    Generate nested training subsets.

    classification:
        stratify directly by class label

    regression:
        quantile-bin y, then stratify by those bins

    All samples are prefixes of ONE ordering, therefore:

        smallest subset
            ⊂
        medium subset
            ⊂
        largest subset
    """

    y = np.asarray(
        y
    ).reshape(-1)

    n_train = len(y)

    sample_sizes = sorted(
        set(
            int(x)
            for x in sample_sizes
        )
    )

    if not sample_sizes:
        raise ValueError(
            "No sample sizes provided."
        )

    if min(sample_sizes) <= 0:
        raise ValueError(
            "Sample sizes must all be > 0."
        )

    if max(sample_sizes) > n_train:
        raise ValueError(
            f"Largest requested sample size is "
            f"{max(sample_sizes):,}, but dataset only has "
            f"{n_train:,} training rows."
        )


    if task == "classification":

        strata = y

        print(
            "[Sampling] Strategy: "
            "classification stratification"
        )

    elif task == "regression":

        print(
            "[Sampling] Strategy: "
            "regression quantile-bin stratification"
        )

        strata = make_regression_bins(
            y=y,
            n_bins=regression_bins,
        )

    else:

        raise ValueError(
            f"Unsupported task: {task}"
        )


    order = make_stratified_order(
        strata=strata,
        seed=seed,
    )

    samples = {}

    for size in sample_sizes:

        samples[size] = (
            order[:size]
            .copy()
        )

    return samples


def print_classification_distribution(
    y: np.ndarray,
    indices: np.ndarray,
    size: int,
):
    y_sample = y[
        indices
    ]

    values, counts = np.unique(
        y_sample,
        return_counts=True,
    )

    print(
        f"\n[Sampling] Distribution for "
        f"{size:,} rows:"
    )

    for value, count in zip(
        values,
        counts,
    ):

        fraction = (
            count / len(y_sample)
        )

        print(
            f"    class={value}: "
            f"{count:,} "
            f"({fraction:.4%})"
        )


def print_regression_distribution(
    y: np.ndarray,
    indices: np.ndarray,
    size: int,
):
    y_sample = y[
        indices
    ]

    print(
        f"\n[Sampling] Regression statistics "
        f"for {size:,} rows:"
    )

    print(
        f"    mean:   "
        f"{np.mean(y_sample):.6f}"
    )

    print(
        f"    std:    "
        f"{np.std(y_sample):.6f}"
    )

    print(
        f"    min:    "
        f"{np.min(y_sample):.6f}"
    )

    print(
        f"    median: "
        f"{np.median(y_sample):.6f}"
    )

    print(
        f"    max:    "
        f"{np.max(y_sample):.6f}"
    )



def verify_nested(
    indices_by_size: dict[int, np.ndarray],
):
    """
    Explicitly verify:

        100k ⊂ 500k ⊂ 1M
    """

    sizes = sorted(
        indices_by_size
    )

    for small_size, large_size in zip(
        sizes[:-1],
        sizes[1:],
    ):

        small = indices_by_size[
            small_size
        ]

        large = indices_by_size[
            large_size
        ]

        # Because these are prefixes, this should be exact.
        if not np.array_equal(
            small,
            large[:small_size],
        ):

            raise RuntimeError(
                f"Nested property failed: "
                f"{small_size:,} is not a prefix "
                f"of {large_size:,}"
            )

        print(
            f"[Sampling] Verified: "
            f"{small_size:,} ⊂ {large_size:,}"
        )



def save_samples(
    data,
    indices_by_size: dict[int, np.ndarray],
    output_root: Path,
    seed: int,
    regression_bins: int,
):
    """
    Save the sampled ROW INDICES.

    We intentionally do not duplicate X_train here.
    Every model can load the exact same indices later.
    """

    dataset_dir = (
        output_root
        / data.name
        / f"seed_{seed}"
    )

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    for size, indices in (
        indices_by_size.items()
    ):

        path = (
            dataset_dir
            / f"train_idx_{size}.npy"
        )

        np.save(
            path,
            indices,
        )

        print(
            f"[Sampling] Saved "
            f"{size:,} indices -> "
            f"{path}"
        )


    metadata = {
        "dataset": data.name,
        "task": data.task,
        "seed": seed,
        "original_n_train": int(
            len(data.y_train)
        ),
        "original_n_test": int(
            len(data.y_test)
        ),
        "sample_sizes": [
            int(size)
            for size
            in sorted(indices_by_size)
        ],
        "nested": True,
        "sampling": (
            "stratified"
            if data.task == "classification"
            else "quantile_binned_stratified"
        ),
        "regression_bins": (
            regression_bins
            if data.task == "regression"
            else None
        ),
    }

    # Classification metadata
    if data.task == "classification":

        values, counts = np.unique(
            data.y_train,
            return_counts=True,
        )

        metadata[
            "full_train_class_distribution"
        ] = {
            str(
                value.item()
                if isinstance(
                    value,
                    np.generic,
                )
                else value
            ): int(count)
            for value, count
            in zip(values, counts)
        }

    # Regression metadata
    else:

        y = np.asarray(
            data.y_train
        )

        metadata[
            "full_train_target_statistics"
        ] = {
            "mean": float(
                np.mean(y)
            ),
            "std": float(
                np.std(y)
            ),
            "min": float(
                np.min(y)
            ),
            "median": float(
                np.median(y)
            ),
            "max": float(
                np.max(y)
            ),
        }

    metadata_path = (
        dataset_dir
        / "sampling_info.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    print(
        f"[Sampling] Metadata -> "
        f"{metadata_path}"
    )

    return dataset_dir


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create fixed nested training subsets "
            "for TALENT-format datasets."
        )
    )


    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help=(
            "Root directory containing TALENT datasets."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Dataset name passed to "
            "load_talent_dataset."
        ),
    )


    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        required=True,
        help=(
            "Nested training sample sizes. "
            "Example: "
            "--sample-sizes "
            "1000000 500000 100000"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Sampling random seed. "
            "Default: 42"
        ),
    )

    parser.add_argument(
        "--regression-bins",
        type=int,
        default=20,
        help=(
            "Number of quantile bins used to "
            "stratify regression datasets. "
            "Default: 20"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="./sampling",
        help=(
            "Directory where fixed sampling "
            "indices will be saved."
        ),
    )

    args = parser.parse_args()


    print("=" * 70)
    print("Loading dataset")
    print("=" * 70)

    data = load_talent_dataset(
        root=args.data_root,
        dataset_name=args.dataset,
    )

    print(
        f"Dataset:   {data.name}"
    )

    print(
        f"Task:      {data.task}"
    )

    print(
        f"Train:     {len(data.y_train):,}"
    )

    print(
        f"Test:      {len(data.y_test):,}"
    )

    print(
        f"Features:  {data.X_train.shape[1]}"
    )

    print()
    print("=" * 70)
    print("Creating nested samples")
    print("=" * 70)

    indices_by_size = make_nested_indices(
        y=data.y_train,
        task=data.task,
        sample_sizes=args.sample_sizes,
        seed=args.seed,
        regression_bins=args.regression_bins,
    )

    print()
    print("=" * 70)
    print("Verifying nesting")
    print("=" * 70)

    verify_nested(
        indices_by_size
    )

    print()
    print("=" * 70)
    print("Sample statistics")
    print("=" * 70)

    for size in sorted(
        indices_by_size
    ):

        indices = (
            indices_by_size[size]
        )

        if (
            data.task
            == "classification"
        ):

            print_classification_distribution(
                y=np.asarray(
                    data.y_train
                ),
                indices=indices,
                size=size,
            )

        else:

            print_regression_distribution(
                y=np.asarray(
                    data.y_train
                ),
                indices=indices,
                size=size,
            )


    print()
    print("=" * 70)
    print("Saving")
    print("=" * 70)

    output_dir = save_samples(
        data=data,
        indices_by_size=indices_by_size,
        output_root=Path(
            args.output_root
        ),
        seed=args.seed,
        regression_bins=args.regression_bins,
    )

    print()
    print("=" * 70)
    print("Done")
    print("=" * 70)

    print(
        f"Output directory: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()