import json
from pathlib import Path

import numpy as np
import pandas as pd

from tfmbench.data import TabularDataset


def _load_split(dataset_dir, split):
    dataset_dir = Path(dataset_dir)

    parts = []

    n_path = dataset_dir / f"N_{split}.npy"
    c_path = dataset_dir / f"C_{split}.npy"

    if n_path.exists():
        X_num = np.load(n_path, allow_pickle=True)
        parts.append(
            pd.DataFrame(
                X_num,
                columns=[f"num_{i}" for i in range(X_num.shape[1])]
            )
        )

    if c_path.exists():
        X_cat = np.load(c_path, allow_pickle=True)
        parts.append(
            pd.DataFrame(
                X_cat,
                columns=[f"cat_{i}" for i in range(X_cat.shape[1])]
            )
        )

    if not parts:
        raise ValueError(
            f"No numeric or categorical features found in {dataset_dir}"
        )

    X = pd.concat(parts, axis=1)

    y = np.load(
        dataset_dir / f"y_{split}.npy",
        allow_pickle=True,
    )

    return X, y


def load_talent_dataset(
    dataset_name,
    root="./data",
    include_val=False,
):
    dataset_dir = Path(root) / dataset_name

    assert dataset_dir.exists()

    with open(dataset_dir / "info.json") as f:
        info = json.load(f)

    X_train, y_train = _load_split(
        dataset_dir,
        "train",
    )

    X_test, y_test = _load_split(
        dataset_dir,
        "test",
    )

    if include_val:
        X_val, y_val = _load_split(
            dataset_dir,
            "val",
        )

        X_train = pd.concat(
            [X_train, X_val],
            axis=0,
            ignore_index=True,
        )

        y_train = np.concatenate(
            [y_train, y_val]
        )

    task_type = info["task_type"]

    task = (
        "regression"
        if task_type == "regression"
        else "classification"
    )

    return TabularDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task=task,
        name=dataset_name,
    )