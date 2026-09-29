import argparse
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from tfmbench.evaluate import evaluate
from tfmbench.datasets.talent import load_talent_dataset


DATA_ROOT = "./data/datasets-talent-large"
RESULT_ROOT = Path("./results")
SAMPLING_ROOT = Path("./data/sampling_idx")
SAMPLE_SEED = 42

MODEL_KWARGS = {
    "xgboost": {
        "tune": True,
        "n_trials": 30,
        "tune_max_rows": 500_000,
        "validation_fraction": 0.20,
        "n_ensemble": 10,
        "ensemble_top_k": 3,
    },

    "catboost": {
        "tune": True,
        "n_trials": 30,
        "tune_max_rows": 500_000,
        "validation_fraction": 0.20,
        "n_ensemble": 10,
        "ensemble_top_k": 3,
    },

    "tabdpt_v1.3": {
        "context_size": 8192,
        "n_ensembles": 1,
    },

    "tabpfn_v3": {
        "n_estimators": 8,
    },
}

def apply_fixed_train_sample(
    data,
    dataset_name,
    sample_size,
    sampling_root,
    seed,
):
    """
    Apply a precomputed training subset.

    X_test/y_test are left untouched.
    """

    safe_dataset = dataset_name.replace("/", "_")

    index_file = (
        sampling_root
        / safe_dataset
        / f"seed_{seed}"
        / f"train_idx_{sample_size}.npy"
    )

    if not index_file.exists():
        raise FileNotFoundError(
            f"Sampling index file does not exist:\n"
            f"{index_file}"
        )

    print(
        f"Loading fixed training sample: "
        f"{index_file}"
    )

    indices = np.load(
        index_file
    )

    if len(indices) != sample_size:
        raise ValueError(
            f"Expected {sample_size:,} indices, "
            f"but found {len(indices):,}."
        )

    if indices.max() >= len(data.y_train):
        raise ValueError(
            f"Sampling indices are incompatible with dataset. "
            f"Maximum index={indices.max():,}, "
            f"but train size={len(data.y_train):,}."
        )

    # --------------------------------------------------
    # X_train
    # --------------------------------------------------

    if hasattr(data.X_train, "iloc"):

        # pandas DataFrame
        data.X_train = (
            data.X_train
            .iloc[indices]
            .reset_index(drop=True)
        )

    else:

        # NumPy array / similar
        data.X_train = (
            data.X_train[
                indices
            ]
        )

    # --------------------------------------------------
    # y_train
    # --------------------------------------------------

    data.y_train = (
        np.asarray(data.y_train)[
            indices
        ]
    )

    print(
        f"Applied fixed sample: "
        f"{len(data.y_train):,} training rows"
    )

    return data

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--index",
        type=int,
        required=True,
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Number of training rows to use from the "
            "precomputed fixed sample. "
            "Default: None = use full training set."
        ),
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    jobs = pd.read_csv(args.manifest)

    job = jobs.iloc[args.index]

    dataset_name = job["dataset"]
    model_name = job["model"]
    sample_size = args.sample_size

    print("=" * 100)
    print(f"Dataset : {dataset_name}")
    print(f"Model   : {model_name}")
    print(f"N       : {job['total_rows']:,}")
    print(f"N train : {job['train_rows']:,}")
    print(f"F       : {job['n_features']:,}")
    print("=" * 100)

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_dataset = dataset_name.replace("/", "_")

    output_file = (
        RESULT_ROOT
        / f"{safe_dataset}__{model_name}.json"
    )

    if output_file.exists():
        print(
            f"Result already exists: {output_file}"
        )
        return

    output = {
        "dataset": dataset_name,
        "model": model_name,
        "job_id": int(job["job_id"]),
        "status": "failed",
    }

    try:


        print("Loading dataset...")

        data = load_talent_dataset(
            dataset_name,
            root=DATA_ROOT,
            include_val=False,
        )
        
        original_train_size = len(data.y_train)

        if sample_size is not None:

            if sample_size < original_train_size:

                data = apply_fixed_train_sample(
                    data=data,
                    dataset_name=dataset_name,
                    sample_size=sample_size,
                    sampling_root=SAMPLING_ROOT,
                    seed=SAMPLE_SEED,
                )

            else:

                print(
                    f"Requested sample size "
                    f"{sample_size:,} >= "
                    f"available training rows "
                    f"{original_train_size:,}. "
                    f"Using full training set."
                )

        else:

            print(
                "No sample size provided. "
                "Using full training set."
            )
        print(
            f"Loaded: "
            f"{data.X_train.shape} -> "
            f"{data.X_test.shape}"
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print("Device:", device)

        start = time.time()

        result = evaluate(
            model_name=model_name,
            data=data,
            device=device,
            tabpfn_token=os.environ.get(
                "TABPFN_TOKEN"
            ),
            model_kwargs=MODEL_KWARGS.get(
                model_name,
                {},
            ),
            return_predictions=False,
            test_batch_size=(
                args.test_batch_size
            ),
        )

        wall_seconds = time.time() - start
        
        output.update({
            "status": "success",

            "metrics": result.metrics,

            "fit_seconds": result.fit_seconds,
            "predict_seconds": result.predict_seconds,

            "wall_seconds": wall_seconds,

            "peak_gpu_memory_mb":
                result.peak_gpu_memory_mb,
        })

    except Exception as e:

        output.update({
            "status": "failed",

            "error_type":
                type(e).__name__,

            "error":
                str(e),

            "traceback":
                traceback.format_exc(),
        })

        print(traceback.format_exc())

    finally:

        with open(output_file, "w") as f:
            json.dump(
                output,
                f,
                indent=2,
                default=str,
            )

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()