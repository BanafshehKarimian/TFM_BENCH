from pathlib import Path
from typing import Literal, Optional
import json

import numpy as np
import pandas as pd
import openml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tfmbench.datasets import BaseTabularDataset


TaskType = Literal[
    "classification",
    "regression",
]


class OpenMLDataset:
    def __init__(
        self,
        name: str,
        test_size: float = 0.2,
        seed: int = 42,
        fold: int = 0,
        repeat: int = 0,
        sample: int = 0,
        task: Optional[TaskType] = None,
    ):
        self.name = name

        self.test_size = test_size
        self.seed = seed

        self.fold = fold
        self.repeat = repeat
        self.sample = sample

        self.task = task

        self.dataset = None
        self.official_task = None
        self.target_name = None
        self.split_source = None

        self.categorical_indicator = None
        self.feature_names = None

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.label_encoder = None


    def _find_official_task(
        self,
        dataset,
    ):
        tasks = openml.tasks.list_tasks(
            data_id=dataset.dataset_id,
            status="active",
        )

        if not tasks:
            return None

        supervised_tasks = []

        for tid, metadata in tasks.items():

            task_type = str(
                metadata.get(
                    "task_type",
                    ""
                )
            ).lower()

            if (
                "supervised classification"
                in task_type
            ):
                inferred_task = (
                    "classification"
                )

            elif (
                "supervised regression"
                in task_type
            ):
                inferred_task = (
                    "regression"
                )

            else:
                continue

            if (
                self.task is not None
                and inferred_task != self.task
            ):
                continue

            supervised_tasks.append(
                (
                    int(tid),
                    inferred_task,
                    metadata,
                )
            )

        if not supervised_tasks:
            return None


        supervised_tasks.sort(
            key=lambda x: x[0]
        )

        task_id, task_type, metadata = (
            supervised_tasks[0]
        )

        task = openml.tasks.get_task(
            task_id
        )

        print(
            f"[OpenML] Found official "
            f"{task_type} task {task_id}"
        )

        print(
            f"[OpenML] Target: "
            f"{task.target_name}"
        )

        print(
            f"[OpenML] Estimation procedure: "
            f"{task.estimation_procedure}"
        )

        return task


    def _task_from_openml_task(
        self,
        task,
    ) -> TaskType:

        task_type = str(
            task.task_type
        ).lower()

        if "classification" in task_type:
            return "classification"

        if "regression" in task_type:
            return "regression"

        raise ValueError(
            f"Unsupported OpenML task type: "
            f"{task.task_type}"
        )

    def _infer_task_from_target(
        self,
        y,
    ) -> TaskType:

        if self.task is not None:
            return self.task

        y_series = pd.Series(y)

        if (
            pd.api.types.is_object_dtype(
                y_series
            )
            or pd.api.types.is_string_dtype(
                y_series
            )
            or pd.api.types.is_bool_dtype(
                y_series
            )
            or isinstance(
                y_series.dtype,
                pd.CategoricalDtype,
            )
        ):
            return "classification"

        n_unique = (
            y_series
            .nunique(
                dropna=True
            )
        )

        if n_unique <= 50:
            return "classification"

        return "regression"


    def _custom_split(
        self,
        y,
    ):

        indices = np.arange(
            len(y)
        )

        stratify = (
            y
            if self.task == "classification"
            else None
        )

        try:

            (
                train_indices,
                test_indices,
            ) = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=stratify,
            )

        except ValueError:

            # Rare-class fallback
            (
                train_indices,
                test_indices,
            ) = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=None,
            )

        return (
            train_indices,
            test_indices,
        )

    def load(
        self,
    ) -> BaseTabularDataset:

        dataset = (
            openml.datasets.get_dataset(
                self.name
            )
        )

        self.dataset = dataset

        print(
            f"[OpenML] Dataset: "
            f"{dataset.name}"
        )

        print(
            f"[OpenML] Dataset ID: "
            f"{dataset.dataset_id}"
        )


        official_task = (
            self._find_official_task(
                dataset
            )
        )

        self.official_task = (
            official_task
        )

        if official_task is not None:

            self.task = (
                self._task_from_openml_task(
                    official_task
                )
            )

            target_name = (
                official_task.target_name
            )

        else:

            target_name = (
                dataset
                .default_target_attribute
            )

            if target_name is None:

                raise ValueError(
                    f"Dataset '{dataset.name}' "
                    f"has no supervised OpenML task "
                    f"and no default target."
                )

        self.target_name = (
            target_name
        )

        print(
            f"[OpenML] Target: "
            f"{target_name}"
        )
        

        (
            X,
            y,
            categorical_indicator,
            feature_names,
        ) = dataset.get_data(
            target=target_name,
            dataset_format="dataframe",
        )

        self.categorical_indicator = (
            categorical_indicator
        )

        self.feature_names = (
            list(feature_names)
        )

        if official_task is None:

            self.task = (
                self._infer_task_from_target(
                    y
                )
            )

        print(
            f"[OpenML] Task: "
            f"{self.task}"
        )

        if self.task == "classification":

            encoder = LabelEncoder()

            y = encoder.fit_transform(
                y
            )

            self.label_encoder = (
                encoder
            )

        else:

            y = np.asarray(
                y,
                dtype=np.float32,
            )

        used_official_split = False

        if official_task is not None:

            try:

                (
                    train_indices,
                    test_indices,
                ) = (
                    official_task
                    .get_train_test_split_indices(
                        repeat=self.repeat,
                        fold=self.fold,
                        sample=self.sample,
                    )
                )

                used_official_split = True

                self.split_source = (
                    f"openml_task_"
                    f"{official_task.id}"
                )

                print(
                    f"[OpenML] Using official "
                    f"fold {self.fold}"
                )

            except Exception as exc:

                print(
                    "no official split"
                )

                print(
                    f"[OpenML] "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )



        if not used_official_split:

            (
                train_indices,
                test_indices,
            ) = self._custom_split(
                y
            )

            self.split_source = (
                f"custom_seed_"
                f"{self.seed}"
            )

        X_train = (
            X.iloc[
                train_indices
            ]
            .reset_index(
                drop=True
            )
        )

        X_test = (
            X.iloc[
                test_indices
            ]
            .reset_index(
                drop=True
            )
        )

        y_train = np.asarray(
            y[
                train_indices
            ]
        )

        y_test = np.asarray(
            y[
                test_indices
            ]
        )


        self.X_train = X_train
        self.X_test = X_test

        self.y_train = y_train
        self.y_test = y_test

        print(
            f"[OpenML] Train: "
            f"{X_train.shape}"
        )

        print(
            f"[OpenML] Test:  "
            f"{X_test.shape}"
        )

        print(
            f"[OpenML] Split source: "
            f"{self.split_source}"
        )

        return BaseTabularDataset(
            X_train=X_train,
            y_train=y_train,

            X_test=X_test,
            y_test=y_test,

            task=self.task,
            name=dataset.name,
        )

    def _make_validation_split(
        self,
        val_fraction_of_train: float,
    ):

        indices = np.arange(
            len(self.X_train)
        )

        stratify = (
            self.y_train
            if self.task == "classification"
            else None
        )

        try:

            (
                train_idx,
                val_idx,
            ) = train_test_split(
                indices,
                test_size=val_fraction_of_train,
                random_state=self.seed,
                stratify=stratify,
            )

        except ValueError:

            (
                train_idx,
                val_idx,
            ) = train_test_split(
                indices,
                test_size=val_fraction_of_train,
                random_state=self.seed,
                stratify=None,
            )

        return (
            train_idx,
            val_idx,
        )

    def _get_feature_groups(
        self,
    ):
        """
        Use OpenML's categorical indicator to split features
        into TALENT's N_* and C_* arrays.
        """

        if (
            self.categorical_indicator
            is not None
            and len(
                self.categorical_indicator
            )
            == self.X_train.shape[1]
        ):

            categorical_columns = [
                column
                for column, is_categorical
                in zip(
                    self.X_train.columns,
                    self.categorical_indicator,
                )
                if is_categorical
            ]

            numerical_columns = [
                column
                for column, is_categorical
                in zip(
                    self.X_train.columns,
                    self.categorical_indicator,
                )
                if not is_categorical
            ]

        else:

            categorical_columns = []

            numerical_columns = []

            for column in (
                self.X_train.columns
            ):

                dtype = (
                    self.X_train[
                        column
                    ].dtype
                )

                if (
                    isinstance(
                        dtype,
                        pd.CategoricalDtype,
                    )
                    or pd.api.types
                    .is_object_dtype(
                        dtype
                    )
                    or pd.api.types
                    .is_string_dtype(
                        dtype
                    )
                ):

                    categorical_columns.append(
                        column
                    )

                else:

                    numerical_columns.append(
                        column
                    )

        return (
            numerical_columns,
            categorical_columns,
        )


    @staticmethod
    def _numerical_array(
        X,
        columns,
    ):

        if not columns:
            return None

        out = (
            X[columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float32
            )
        )

        return out


    @staticmethod
    def _categorical_array(
        X,
        columns,
    ):

        if not columns:
            return None

        out = (
            X[columns]
            .astype("object")
        )

        out = out.where(
            pd.notna(out),
            None,
        )

        return out.to_numpy(
            dtype=object
        )


    def save(
        self,
        output_root: str,
        val_fraction_of_train: float = 0.20,
        overwrite: bool = False,
    ) -> Path:
        """
        Save dataset in TALENT-style format.

        Example
        -------
        ds = OpenMLDataset("US_Accidents_March23")

        ds.load()

        ds.save(
            "/scratch/TALENT-extension/"
            "very-large-scale"
        )
        """

        # ----------------------------------------------------
        # Make sure load() has run
        # ----------------------------------------------------

        if self.X_train is None:

            print(
                "[OpenML] Dataset not loaded yet. "
                "Loading first..."
            )

            self.load()

        # ----------------------------------------------------
        # Output folder
        # ----------------------------------------------------

        output_dir = (
            Path(output_root)
            / self.dataset.name
        )

        if (
            output_dir.exists()
            and not overwrite
        ):

            # Don't silently overwrite an existing dataset.
            existing = list(
                output_dir.glob("*.npy")
            )

            if existing:

                raise FileExistsError(
                    f"Dataset already exists:\n"
                    f"{output_dir}\n\n"
                    f"Use overwrite=True to replace it."
                )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Create validation split from training data
        # ----------------------------------------------------

        (
            inner_train_idx,
            val_idx,
        ) = self._make_validation_split(
            val_fraction_of_train
        )

        X_train = (
            self.X_train
            .iloc[inner_train_idx]
            .reset_index(drop=True)
        )

        X_val = (
            self.X_train
            .iloc[val_idx]
            .reset_index(drop=True)
        )

        X_test = (
            self.X_test
            .reset_index(drop=True)
        )

        y_train = np.asarray(
            self.y_train[
                inner_train_idx
            ]
        )

        y_val = np.asarray(
            self.y_train[
                val_idx
            ]
        )

        y_test = np.asarray(
            self.y_test
        )

        # ----------------------------------------------------
        # Feature groups
        # ----------------------------------------------------

        (
            numerical_columns,
            categorical_columns,
        ) = self._get_feature_groups()

        print(
            f"[TALENT] Numerical features: "
            f"{len(numerical_columns)}"
        )

        print(
            f"[TALENT] Categorical features: "
            f"{len(categorical_columns)}"
        )

        # ====================================================
        # Numerical features
        # ====================================================

        if numerical_columns:

            N_train = (
                self._numerical_array(
                    X_train,
                    numerical_columns,
                )
            )

            N_val = (
                self._numerical_array(
                    X_val,
                    numerical_columns,
                )
            )

            N_test = (
                self._numerical_array(
                    X_test,
                    numerical_columns,
                )
            )

            np.save(
                output_dir
                / "N_train.npy",
                N_train,
            )

            np.save(
                output_dir
                / "N_val.npy",
                N_val,
            )

            np.save(
                output_dir
                / "N_test.npy",
                N_test,
            )

        # ====================================================
        # Categorical features
        # ====================================================

        if categorical_columns:

            C_train = (
                self._categorical_array(
                    X_train,
                    categorical_columns,
                )
            )

            C_val = (
                self._categorical_array(
                    X_val,
                    categorical_columns,
                )
            )

            C_test = (
                self._categorical_array(
                    X_test,
                    categorical_columns,
                )
            )

            np.save(
                output_dir
                / "C_train.npy",
                C_train,
                allow_pickle=True,
            )

            np.save(
                output_dir
                / "C_val.npy",
                C_val,
                allow_pickle=True,
            )

            np.save(
                output_dir
                / "C_test.npy",
                C_test,
                allow_pickle=True,
            )


        np.save(
            output_dir
            / "y_train.npy",
            y_train,
        )

        np.save(
            output_dir
            / "y_val.npy",
            y_val,
        )

        np.save(
            output_dir
            / "y_test.npy",
            y_test,
        )

        if self.task == "regression":

            talent_task_type = (
                "regression"
            )

            n_classes = None

        else:

            n_classes = int(
                len(
                    np.unique(
                        np.concatenate([
                            y_train,
                            y_val,
                            y_test,
                        ])
                    )
                )
            )

            talent_task_type = (
                "binclass"
                if n_classes == 2
                else "multiclass"
            )


        classes = None

        if self.label_encoder is not None:

            classes = [
                (
                    x.item()
                    if isinstance(
                        x,
                        np.generic,
                    )
                    else x
                )
                for x
                in self.label_encoder.classes_
            ]

            classes = [
                (
                    x
                    if isinstance(
                        x,
                        (
                            str,
                            int,
                            float,
                            bool,
                        ),
                    )
                    else str(x)
                )
                for x in classes
            ]

        total_size = (
            len(y_train)
            + len(y_val)
            + len(y_test)
        )

        info = {
            "name": (
                self.dataset.name
            ),

            "source": "openml",

            "openml_dataset_id": int(
                self.dataset.dataset_id
            ),

            "openml_task_id": (
                int(self.official_task.id)
                if self.official_task
                is not None
                else None
            ),

            "target": (
                self.target_name
            ),

            "task_type": (
                talent_task_type
            ),

            "task": (
                self.task
            ),

            "split_source": (
                self.split_source
            ),

            "seed": (
                self.seed
            ),

            "fold": (
                self.fold
            ),

            "repeat": (
                self.repeat
            ),

            "sample": (
                self.sample
            ),

            "train_size": int(
                len(y_train)
            ),

            "val_size": int(
                len(y_val)
            ),

            "test_size": int(
                len(y_test)
            ),

            "total_size": int(
                total_size
            ),

            "train_fraction": (
                len(y_train)
                / total_size
            ),

            "val_fraction": (
                len(y_val)
                / total_size
            ),

            "test_fraction": (
                len(y_test)
                / total_size
            ),

            "n_num_features": int(
                len(
                    numerical_columns
                )
            ),

            "n_cat_features": int(
                len(
                    categorical_columns
                )
            ),

            "n_features": int(
                len(
                    numerical_columns
                )
                +
                len(
                    categorical_columns
                )
            ),

            "num_feature_names": (
                [
                    str(x)
                    for x
                    in numerical_columns
                ]
            ),

            "cat_feature_names": (
                [
                    str(x)
                    for x
                    in categorical_columns
                ]
            ),

            "n_classes": (
                n_classes
            ),

            "classes": (
                classes
            ),
        }

        with open(
            output_dir / "info.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                info,
                f,
                indent=4,
                ensure_ascii=False,
            )

        return output_dir