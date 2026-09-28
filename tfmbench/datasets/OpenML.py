from typing import Literal
import pandas as pd
import numpy as np
import openml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tfmbench.datasets import BaseTabularDataset


TaskType = Literal["classification", "regression"]


class OpenMLDataset:
    def __init__(
        self,
        name: str,
        test_size: float = 0.2,
        seed: int = 42,
        fold: int = 0,
        repeat: int = 0,
        sample: int = 0,
    ):
        self.name = name
        self.test_size = test_size
        self.seed = seed

        self.fold = fold
        self.repeat = repeat
        self.sample = sample

    def _find_official_task(self, dataset):

        tasks = openml.tasks.list_tasks(
            data_id=dataset.dataset_id,
            status="active",
        )

        print(tasks)
        if len(tasks.keys()) < 1:
            return None

        task_id = tasks[list(tasks.keys())[0]]["tid"]

        return openml.tasks.get_task(
            task_id
        )

    def load(self) -> BaseTabularDataset:
        dataset = openml.datasets.get_dataset(
            self.name
        )


        official_task = (
            self._find_official_task(
                dataset
            )
        )
        self.task = "classification" if "Classification" in official_task.task_type else "regression"

        used_official_split = False
        
        if official_task is not None:

            try:

                (
                    train_indices,
                    test_indices,
                ) = (
                    official_task
                    .get_train_test_split_indices(
                        repeat=0,
                        fold=self.fold,
                        sample=0,
                    )
                )

                used_official_split = True

                print(
                    f"[OpenML] Using official "
                    f"fold {self.fold}"
                )

            except Exception as exc:

                print(
                    "[OpenML] WARNING: "
                    "Official split could not "
                    "be loaded."
                )

                print(
                    f"[OpenML] "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(
                    "[OpenML] Falling back to "
                    f"custom "
                    f"{1-self.test_size:.0%}/"
                    f"{self.test_size:.0%} "
                    f"split "
                    f"(seed={self.seed})."
                )


        if dataset.default_target_attribute is None:
            raise ValueError(
                f"Dataset '{dataset.name}' has no "
                "default target and no matching "
                "OpenML task."
            )
        X, y, categorical_indicator, feature_names = (
            dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format="dataframe",
            )
        )
        
        if not used_official_split:

            print(
                f"[OpenML] No active {self.task} task "
                f"found for '{dataset.name}'."
            )

            print(
                f"[OpenML] Using custom "
                f"{1 - self.test_size:.0%}/"
                f"{self.test_size:.0%} split "
                f"(seed={self.seed})."
            )


            indices = np.arange(
                len(X)
            )

            stratify = (
                y
                if self.task == "classification"
                else None
            )

            (
                train_indices,
                test_indices,
            ) = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=stratify,
            )

        if self.task == "classification":

            encoder = LabelEncoder()

            y = encoder.fit_transform(
                y
            )

        else:

            y = np.asarray(
                y,
                dtype=np.float32,
            )

        X_train = (
            X.iloc[train_indices]
            .reset_index(drop=True)
        )

        X_test = (
            X.iloc[test_indices]
            .reset_index(drop=True)
        )

        y_train = np.asarray(
            y[train_indices]
        )

        y_test = np.asarray(
            y[test_indices]
        )

        print(
            f"[OpenML] Dataset: {dataset.name}"
        )
        print(
            f"[OpenML] Train: {X_train.shape}"
        )
        print(
            f"[OpenML] Test:  {X_test.shape}"
        )
        
        return BaseTabularDataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task=self.task,
            name=dataset.name,
        )