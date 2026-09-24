import argparse
import json
import os
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

from tfmbench.evaluate import evaluate
from tfmbench.datasets.talent import load_talent_dataset


DATA_ROOT = "./data/datasets-talent-large"
RESULT_ROOT = Path("./results")


MODEL_KWARGS = {

    "xgboost": {
        "n_estimators": 500,
        "max_depth": 8,
    },

    "catboost": {
        "iterations": 500,
        "depth": 8,
    },

    "tabdpt_v1.3": {
        "context_size": 8192,
        "n_ensembles": 1,
    },

    "tabpfn_v3": {
        "n_estimators": 8,
    },
}


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

    args = parser.parse_args()

    jobs = pd.read_csv(args.manifest)

    job = jobs.iloc[args.index]

    dataset_name = job["dataset"]
    model_name = job["model"]

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