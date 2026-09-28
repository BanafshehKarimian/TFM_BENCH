import json
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import (
    ParameterSampler,
    train_test_split,
)

from .base import BaseTFM


# ============================================================
# General helpers
# ============================================================

CLASSIFICATION_TASKS = {
    "classification",
    "binclass",
    "multiclass",
}


def _is_classification(task):
    return str(task).lower() in CLASSIFICATION_TASKS


def _as_dataframe(X):
    if isinstance(X, pd.DataFrame):
        return X.copy()

    return pd.DataFrame(X)


def _as_1d_y(y):
    return np.asarray(y).reshape(-1)


def _to_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype(np.float32)


def _to_string(series):
    return series.astype("string")


def _device_type(device):
    """
    Supports both:
        torch.device("cuda")
        "cuda"
        "cuda:0"
        "cpu"
    """

    if hasattr(device, "type"):
        return device.type

    return str(device).split(":")[0]


# ============================================================
# Tuning split helpers
# ============================================================

def _can_stratify(y):
    """
    True when every class has at least 2 examples.
    """

    y = _as_1d_y(y)

    _, counts = np.unique(
        y,
        return_counts=True,
    )

    return (
        len(counts) > 1
        and counts.min() >= 2
    )


def _make_tuning_indices(
    y,
    classification,
    validation_fraction,
    tune_max_rows,
    seed,
):
    """
    Returns:

        tuning_train_indices
        tuning_validation_indices

    If tune_max_rows is set, HPO is done on at most that many
    rows from the original training set.

    Final ensemble training still uses ALL training rows.
    """

    y = _as_1d_y(y)

    n = len(y)

    indices = np.arange(n)

    # --------------------------------------------------------
    # Optional HPO subsample
    # --------------------------------------------------------

    if (
        tune_max_rows is not None
        and n > tune_max_rows
    ):

        stratify = None

        if (
            classification
            and _can_stratify(y)
        ):
            stratify = y

        try:
            pool_indices, _ = train_test_split(
                indices,
                train_size=int(tune_max_rows),
                random_state=seed,
                stratify=stratify,
            )

        except ValueError:
            # Rare classes can occasionally make stratification
            # impossible for the requested sample size.
            rng = np.random.default_rng(seed)

            pool_indices = rng.choice(
                indices,
                size=int(tune_max_rows),
                replace=False,
            )

    else:
        pool_indices = indices

    # --------------------------------------------------------
    # Tuning train / validation split
    # --------------------------------------------------------

    y_pool = y[pool_indices]

    stratify = None

    if (
        classification
        and _can_stratify(y_pool)
    ):
        stratify = y_pool

    try:
        train_idx, val_idx = train_test_split(
            pool_indices,
            test_size=validation_fraction,
            random_state=seed,
            stratify=stratify,
        )

    except ValueError:
        # Fall back if extremely rare classes make stratification
        # impossible.
        train_idx, val_idx = train_test_split(
            pool_indices,
            test_size=validation_fraction,
            random_state=seed,
            stratify=None,
        )

    return (
        np.asarray(train_idx),
        np.asarray(val_idx),
    )


# ============================================================
# Tuning objective
# ============================================================

def _classification_score(
    model,
    X,
    y,
    metric,
):
    """
    Returns:
        raw_score,
        utility

    utility is ALWAYS higher-is-better.
    """

    y = _as_1d_y(y)

    metric = str(metric).lower()

    if metric == "accuracy":

        pred = model.predict(X)

        value = accuracy_score(
            y,
            pred,
        )

        return value, value

    if metric == "balanced_accuracy":

        pred = model.predict(X)

        value = balanced_accuracy_score(
            y,
            pred,
        )

        return value, value

    if metric == "f1_macro":

        pred = model.predict(X)

        value = f1_score(
            y,
            pred,
            average="macro",
            zero_division=0,
        )

        return value, value

    if metric == "log_loss":

        proba = model.predict_proba(X)

        classes = np.asarray(
            model.classes_
        )

        value = log_loss(
            y,
            proba,
            labels=classes,
        )

        # smaller is better
        return value, -value

    if metric == "roc_auc":

        proba = model.predict_proba(X)

        classes = np.asarray(
            model.classes_
        )

        if len(classes) == 2:

            # Explicit binary target so this also works
            # with non-0/1 class labels.
            y_binary = (
                y == classes[1]
            ).astype(int)

            value = roc_auc_score(
                y_binary,
                proba[:, 1],
            )

        else:

            value = roc_auc_score(
                y,
                proba,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )

        return value, value

    raise ValueError(
        f"Unknown classification tuning metric: {metric}"
    )


def _regression_score(
    model,
    X,
    y,
    metric,
):
    """
    Returns:
        raw_score,
        utility

    utility is ALWAYS higher-is-better.
    """

    y = _as_1d_y(y)

    pred = np.asarray(
        model.predict(X)
    ).reshape(-1)

    metric = str(metric).lower()

    if metric == "rmse":

        value = np.sqrt(
            mean_squared_error(
                y,
                pred,
            )
        )

        return value, -value

    if metric == "mae":

        value = mean_absolute_error(
            y,
            pred,
        )

        return value, -value

    if metric == "r2":

        value = r2_score(
            y,
            pred,
        )

        return value, value

    raise ValueError(
        f"Unknown regression tuning metric: {metric}"
    )


def _score_candidate(
    model,
    X,
    y,
    classification,
    metric,
):

    if classification:

        return _classification_score(
            model,
            X,
            y,
            metric,
        )

    return _regression_score(
        model,
        X,
        y,
        metric,
    )


# ============================================================
# Parameter spaces
# ============================================================

DEFAULT_XGB_SEARCH_SPACE = {

    "n_estimators": [
        300,
        500,
        800,
        1200,
    ],

    "learning_rate": [
        0.02,
        0.03,
        0.05,
        0.1,
    ],

    "max_depth": [
        4,
        6,
        8,
        10,
    ],

    "min_child_weight": [
        1,
        3,
        5,
        10,
    ],

    "subsample": [
        0.6,
        0.8,
        1.0,
    ],

    "colsample_bytree": [
        0.6,
        0.8,
        1.0,
    ],

    "reg_alpha": [
        0.0,
        0.01,
        0.1,
        1.0,
        10.0,
    ],

    "reg_lambda": [
        0.1,
        1.0,
        10.0,
    ],
}


DEFAULT_CATBOOST_SEARCH_SPACE = {

    "iterations": [
        300,
        500,
        800,
        1200,
    ],

    "learning_rate": [
        0.02,
        0.03,
        0.05,
        0.1,
    ],

    "depth": [
        4,
        6,
        8,
        10,
    ],

    "l2_leaf_reg": [
        1.0,
        3.0,
        10.0,
        30.0,
    ],

    "random_strength": [
        0.0,
        0.5,
        1.0,
        2.0,
    ],

    "border_count": [
        64,
        128,
        254,
    ],
}


# ============================================================
# XGBoost
# ============================================================

class XGBoostAdapter(BaseTFM):

    def __init__(
        self,
        task,
        device,
        seed,

        # ----------------------------------------------------
        # Tuning
        # ----------------------------------------------------

        tune=True,
        n_trials=20,
        validation_fraction=0.20,

        # HPO only uses this many rows.
        # Final ensemble is still fit on ALL rows.
        tune_max_rows=200_000,

        tuning_metric=None,

        search_space=None,

        # ----------------------------------------------------
        # Ensemble
        # ----------------------------------------------------

        n_ensemble=5,

        # Number of best HPO configurations allowed into
        # the final ensemble.
        ensemble_top_k=3,

        **kwargs,
    ):

        super().__init__(
            task=task,
            device=device,
            seed=seed,
        )

        self.classification = (
            _is_classification(task)
        )

        self.device_type = (
            _device_type(device)
        )

        self.tune = tune
        self.n_trials = int(n_trials)

        self.validation_fraction = (
            float(validation_fraction)
        )

        self.tune_max_rows = (
            tune_max_rows
        )

        if tuning_metric is None:

            if self.classification:
                tuning_metric = (
                    "balanced_accuracy"
                )
            else:
                tuning_metric = "rmse"

        self.tuning_metric = (
            tuning_metric
        )

        self.search_space = (
            search_space
            if search_space is not None
            else DEFAULT_XGB_SEARCH_SPACE
        )

        self.n_ensemble = int(
            n_ensemble
        )

        self.ensemble_top_k = int(
            ensemble_top_k
        )

        # Fixed params supplied by caller.
        #
        # Search parameters override these during tuning.
        self.base_model_kwargs = dict(
            kwargs
        )

        self._category_levels = {}

        self._models = []

        # Keep compatibility with code that expects _model
        self._model = None

        self._classes = None

        self.best_params_ = None
        self.tuning_results_ = []
        self.ensemble_params_ = []


    # ========================================================
    # Model creation
    # ========================================================

    def _make_model(
        self,
        params,
        seed,
    ):

        from xgboost import (
            XGBClassifier,
            XGBRegressor,
        )

        xgb_device = (
            "cuda"
            if self.device_type == "cuda"
            else "cpu"
        )

        common_kwargs = {
            "random_state": seed,
            "tree_method": "hist",
            "device": xgb_device,
            "n_jobs": -1,
            "enable_categorical": True,
        }

        # Explicit caller kwargs
        common_kwargs.update(
            self.base_model_kwargs
        )

        # Tuned parameters win
        common_kwargs.update(
            params
        )

        if self.classification:

            return XGBClassifier(
                **common_kwargs
            )

        return XGBRegressor(
            **common_kwargs
        )


    # ========================================================
    # XGBoost preprocessing
    # ========================================================

    def _fit_preprocessor(
        self,
        X,
    ):

        X = _as_dataframe(X)

        self._category_levels = {}

        for col in X.columns:

            col_name = str(col)

            # ------------------------------------------------
            # TALENT numeric
            # ------------------------------------------------

            if col_name.startswith(
                "num_"
            ):

                X[col] = _to_numeric(
                    X[col]
                )

            # ------------------------------------------------
            # TALENT categorical
            # ------------------------------------------------

            elif col_name.startswith(
                "cat_"
            ):

                values = _to_string(
                    X[col]
                )

                categories = pd.Index(
                    values
                    .dropna()
                    .unique()
                )

                self._category_levels[
                    col
                ] = categories

                X[col] = pd.Categorical(
                    values,
                    categories=categories,
                )

            # ------------------------------------------------
            # Fallback
            # ------------------------------------------------

            else:

                try:

                    X[col] = pd.to_numeric(
                        X[col],
                        errors="raise",
                    ).astype(
                        np.float32
                    )

                except (
                    ValueError,
                    TypeError,
                ):

                    values = _to_string(
                        X[col]
                    )

                    categories = pd.Index(
                        values
                        .dropna()
                        .unique()
                    )

                    self._category_levels[
                        col
                    ] = categories

                    X[col] = pd.Categorical(
                        values,
                        categories=categories,
                    )

        return X


    def _transform_X(
        self,
        X,
    ):

        X = _as_dataframe(X)

        for col in X.columns:

            if col in self._category_levels:

                values = _to_string(
                    X[col]
                )

                X[col] = pd.Categorical(
                    values,
                    categories=(
                        self._category_levels[
                            col
                        ]
                    ),
                )

                # Unseen categories become NaN.

            else:

                X[col] = _to_numeric(
                    X[col]
                )

        return X


    # ========================================================
    # HPO
    # ========================================================

    def _tune_parameters(
        self,
        X,
        y,
    ):

        y = _as_1d_y(y)

        train_idx, val_idx = (
            _make_tuning_indices(
                y=y,
                classification=(
                    self.classification
                ),
                validation_fraction=(
                    self.validation_fraction
                ),
                tune_max_rows=(
                    self.tune_max_rows
                ),
                seed=self.seed,
            )
        )

        X_raw = _as_dataframe(X)

        X_train_raw = X_raw.iloc[
            train_idx
        ].copy()

        X_val_raw = X_raw.iloc[
            val_idx
        ].copy()

        y_train = y[train_idx]
        y_val = y[val_idx]

        # ----------------------------------------------------
        # IMPORTANT:
        # Fit categorical vocabulary on internal HPO TRAIN,
        # not validation.
        # ----------------------------------------------------

        X_train = self._fit_preprocessor(
            X_train_raw
        )

        X_val = self._transform_X(
            X_val_raw
        )

        sampler = ParameterSampler(
            self.search_space,
            n_iter=self.n_trials,
            random_state=self.seed,
        )

        trial_results = []

        for trial_id, params in enumerate(
            sampler
        ):

            model = self._make_model(
                params=params,
                seed=(
                    self.seed
                    + trial_id
                ),
            )

            try:

                model.fit(
                    X_train,
                    y_train,
                )

                raw_score, utility = (
                    _score_candidate(
                        model=model,
                        X=X_val,
                        y=y_val,
                        classification=(
                            self.classification
                        ),
                        metric=(
                            self.tuning_metric
                        ),
                    )
                )

                trial_results.append({
                    "trial": trial_id,
                    "params": dict(params),
                    "validation_score": (
                        float(raw_score)
                    ),
                    "utility": (
                        float(utility)
                    ),
                    "status": "success",
                })

            except Exception as exc:

                trial_results.append({
                    "trial": trial_id,
                    "params": dict(params),
                    "validation_score": None,
                    "utility": (
                        float("-inf")
                    ),
                    "status": "failed",
                    "error": repr(exc),
                })

            finally:

                del model

        valid_trials = [
            trial
            for trial in trial_results
            if trial["status"] == "success"
        ]

        if not valid_trials:

            raise RuntimeError(
                "All XGBoost tuning trials failed."
            )

        valid_trials.sort(
            key=lambda x: x["utility"],
            reverse=True,
        )

        self.tuning_results_ = (
            trial_results
        )

        self.best_params_ = dict(
            valid_trials[0]["params"]
        )

        # ----------------------------------------------------
        # Keep top unique configurations
        # ----------------------------------------------------

        top_configs = []

        seen = set()

        for trial in valid_trials:

            params = dict(
                trial["params"]
            )

            signature = json.dumps(
                params,
                sort_keys=True,
                default=str,
            )

            if signature in seen:
                continue

            seen.add(signature)

            top_configs.append(
                params
            )

            if (
                len(top_configs)
                >= self.ensemble_top_k
            ):
                break

        return top_configs


    # ========================================================
    # Fit
    # ========================================================

    def fit(
        self,
        X,
        y,
    ):

        y = _as_1d_y(y)

        # ----------------------------------------------------
        # Tune
        # ----------------------------------------------------

        if self.tune:

            top_configs = (
                self._tune_parameters(
                    X,
                    y,
                )
            )

        else:

            top_configs = [
                {}
            ]

            self.best_params_ = {}

        # ----------------------------------------------------
        # IMPORTANT:
        # Refit preprocessing on ALL training data.
        # ----------------------------------------------------

        X_full = self._fit_preprocessor(
            X
        )

        # ----------------------------------------------------
        # Fit final ensemble
        # ----------------------------------------------------

        self._models = []
        self.ensemble_params_ = []

        for ensemble_id in range(
            self.n_ensemble
        ):

            config = dict(
                top_configs[
                    ensemble_id
                    % len(top_configs)
                ]
            )

            ensemble_seed = (
                self.seed
                + 10_000
                + ensemble_id
            )

            model = self._make_model(
                params=config,
                seed=ensemble_seed,
            )

            model.fit(
                X_full,
                y,
            )

            self._models.append(
                model
            )

            self.ensemble_params_.append({
                "seed": ensemble_seed,
                "params": config,
            })

        self._model = (
            self._models[0]
        )

        if self.classification:

            self._classes = np.asarray(
                self._model.classes_
            )

        return self


    # ========================================================
    # Prediction
    # ========================================================

    @property
    def classes_(self):

        if not self.classification:
            return None

        return self._classes


    def predict_proba(
        self,
        X,
    ):

        if not self.classification:
            return None

        X = self._transform_X(
            X
        )

        probabilities = []

        for model in self._models:

            p = np.asarray(
                model.predict_proba(X)
            )

            model_classes = np.asarray(
                model.classes_
            )

            # Usually identical, but align defensively.
            if np.array_equal(
                model_classes,
                self._classes,
            ):

                probabilities.append(
                    p
                )

            else:

                aligned = np.zeros(
                    (
                        len(X),
                        len(self._classes),
                    ),
                    dtype=np.float64,
                )

                for j, cls in enumerate(
                    model_classes
                ):

                    target_idx = np.where(
                        self._classes == cls
                    )[0][0]

                    aligned[
                        :,
                        target_idx,
                    ] = p[:, j]

                probabilities.append(
                    aligned
                )

        return np.mean(
            probabilities,
            axis=0,
        )


    def predict(
        self,
        X,
    ):

        if self.classification:

            probabilities = (
                self.predict_proba(X)
            )

            indices = np.argmax(
                probabilities,
                axis=1,
            )

            return self._classes[
                indices
            ]

        X = self._transform_X(
            X
        )

        predictions = [
            np.asarray(
                model.predict(X)
            ).reshape(-1)

            for model in self._models
        ]

        return np.mean(
            predictions,
            axis=0,
        )


# ============================================================
# CatBoost
# ============================================================

class CatBoostAdapter(BaseTFM):

    def __init__(
        self,
        task,
        device,
        seed,

        # ----------------------------------------------------
        # Tuning
        # ----------------------------------------------------

        tune=True,
        n_trials=20,
        validation_fraction=0.20,
        tune_max_rows=200_000,
        tuning_metric=None,
        search_space=None,

        # ----------------------------------------------------
        # Ensemble
        # ----------------------------------------------------

        n_ensemble=5,
        ensemble_top_k=3,

        **kwargs,
    ):

        super().__init__(
            task=task,
            device=device,
            seed=seed,
        )

        self.classification = (
            _is_classification(task)
        )

        self.device_type = (
            _device_type(device)
        )

        self.tune = tune
        self.n_trials = int(n_trials)

        self.validation_fraction = (
            float(validation_fraction)
        )

        self.tune_max_rows = (
            tune_max_rows
        )

        if tuning_metric is None:

            if self.classification:
                tuning_metric = (
                    "balanced_accuracy"
                )
            else:
                tuning_metric = "rmse"

        self.tuning_metric = (
            tuning_metric
        )

        self.search_space = (
            search_space
            if search_space is not None
            else DEFAULT_CATBOOST_SEARCH_SPACE
        )

        self.n_ensemble = int(
            n_ensemble
        )

        self.ensemble_top_k = int(
            ensemble_top_k
        )

        self.base_model_kwargs = dict(
            kwargs
        )

        self._cat_features = []

        self._models = []
        self._model = None

        self._classes = None

        self.best_params_ = None
        self.tuning_results_ = []
        self.ensemble_params_ = []


    # ========================================================
    # Model creation
    # ========================================================

    def _make_model(
        self,
        params,
        seed,
    ):

        from catboost import (
            CatBoostClassifier,
            CatBoostRegressor,
        )

        common_kwargs = {
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
        }

        if self.device_type == "cuda":

            common_kwargs[
                "task_type"
            ] = "GPU"

            # No devices="cuda".
            #
            # Under Slurm CUDA_VISIBLE_DEVICES normally
            # exposes the allocated GPU to the process.

        else:

            common_kwargs[
                "task_type"
            ] = "CPU"

        common_kwargs.update(
            self.base_model_kwargs
        )

        common_kwargs.update(
            params
        )

        if self.classification:

            return CatBoostClassifier(
                **common_kwargs
            )

        return CatBoostRegressor(
            **common_kwargs
        )


    # ========================================================
    # Preprocessing
    # ========================================================

    def _prepare_X(
        self,
        X,
    ):

        X = _as_dataframe(X)

        cat_features = []

        for col in X.columns:

            col_name = str(col)

            # ------------------------------------------------
            # TALENT numeric
            # ------------------------------------------------

            if col_name.startswith(
                "num_"
            ):

                X[col] = _to_numeric(
                    X[col]
                )

            # ------------------------------------------------
            # TALENT categorical
            # ------------------------------------------------

            elif col_name.startswith(
                "cat_"
            ):

                cat_features.append(
                    col
                )

                X[col] = (
                    _to_string(X[col])
                    .fillna(
                        "__MISSING__"
                    )
                    .astype(str)
                )

            # ------------------------------------------------
            # Fallback
            # ------------------------------------------------

            else:

                # TALENT numeric arrays can occasionally
                # arrive with dtype=object, so first test
                # whether every nonmissing value is numeric.
                numeric = pd.to_numeric(
                    X[col],
                    errors="coerce",
                )

                original_nonmissing = (
                    X[col]
                    .notna()
                    .sum()
                )

                numeric_nonmissing = (
                    numeric
                    .notna()
                    .sum()
                )

                if (
                    numeric_nonmissing
                    == original_nonmissing
                ):

                    X[col] = (
                        numeric.astype(
                            np.float32
                        )
                    )

                else:

                    cat_features.append(
                        col
                    )

                    X[col] = (
                        _to_string(X[col])
                        .fillna(
                            "__MISSING__"
                        )
                        .astype(str)
                    )

        return (
            X,
            cat_features,
        )


    # ========================================================
    # HPO
    # ========================================================

    def _tune_parameters(
        self,
        X,
        y,
    ):

        y = _as_1d_y(y)

        train_idx, val_idx = (
            _make_tuning_indices(
                y=y,
                classification=(
                    self.classification
                ),
                validation_fraction=(
                    self.validation_fraction
                ),
                tune_max_rows=(
                    self.tune_max_rows
                ),
                seed=self.seed,
            )
        )

        X_raw = _as_dataframe(X)

        X_train_raw = X_raw.iloc[
            train_idx
        ].copy()

        X_val_raw = X_raw.iloc[
            val_idx
        ].copy()

        y_train = y[
            train_idx
        ]

        y_val = y[
            val_idx
        ]

        X_train, cat_features = (
            self._prepare_X(
                X_train_raw
            )
        )

        X_val, _ = (
            self._prepare_X(
                X_val_raw
            )
        )

        sampler = ParameterSampler(
            self.search_space,
            n_iter=self.n_trials,
            random_state=self.seed,
        )

        trial_results = []

        for trial_id, params in enumerate(
            sampler
        ):

            model = self._make_model(
                params=params,
                seed=(
                    self.seed
                    + trial_id
                ),
            )

            try:

                model.fit(
                    X_train,
                    y_train,
                    cat_features=(
                        cat_features
                    ),
                )

                raw_score, utility = (
                    _score_candidate(
                        model=model,
                        X=X_val,
                        y=y_val,
                        classification=(
                            self.classification
                        ),
                        metric=(
                            self.tuning_metric
                        ),
                    )
                )

                trial_results.append({
                    "trial": trial_id,
                    "params": dict(params),
                    "validation_score": (
                        float(raw_score)
                    ),
                    "utility": (
                        float(utility)
                    ),
                    "status": "success",
                })

            except Exception as exc:

                trial_results.append({
                    "trial": trial_id,
                    "params": dict(params),
                    "validation_score": None,
                    "utility": (
                        float("-inf")
                    ),
                    "status": "failed",
                    "error": repr(exc),
                })

            finally:

                del model

        valid_trials = [
            trial
            for trial in trial_results
            if trial["status"] == "success"
        ]

        if not valid_trials:

            raise RuntimeError(
                "All CatBoost tuning trials failed."
            )

        valid_trials.sort(
            key=lambda x: x["utility"],
            reverse=True,
        )

        self.tuning_results_ = (
            trial_results
        )

        self.best_params_ = dict(
            valid_trials[0]["params"]
        )

        # ----------------------------------------------------
        # Top unique hyperparameter configs
        # ----------------------------------------------------

        top_configs = []

        seen = set()

        for trial in valid_trials:

            params = dict(
                trial["params"]
            )

            signature = json.dumps(
                params,
                sort_keys=True,
                default=str,
            )

            if signature in seen:
                continue

            seen.add(signature)

            top_configs.append(
                params
            )

            if (
                len(top_configs)
                >= self.ensemble_top_k
            ):
                break

        return top_configs


    # ========================================================
    # Fit
    # ========================================================

    def fit(
        self,
        X,
        y,
    ):

        y = _as_1d_y(y)

        # ----------------------------------------------------
        # Tune
        # ----------------------------------------------------

        if self.tune:

            top_configs = (
                self._tune_parameters(
                    X,
                    y,
                )
            )

        else:

            top_configs = [
                {}
            ]

            self.best_params_ = {}

        # ----------------------------------------------------
        # Full training preprocessing
        # ----------------------------------------------------

        X_full, cat_features = (
            self._prepare_X(X)
        )

        self._cat_features = (
            cat_features
        )

        # ----------------------------------------------------
        # Fit ensemble
        # ----------------------------------------------------

        self._models = []
        self.ensemble_params_ = []

        for ensemble_id in range(
            self.n_ensemble
        ):

            config = dict(
                top_configs[
                    ensemble_id
                    % len(top_configs)
                ]
            )

            ensemble_seed = (
                self.seed
                + 10_000
                + ensemble_id
            )

            model = self._make_model(
                params=config,
                seed=ensemble_seed,
            )

            model.fit(
                X_full,
                y,
                cat_features=(
                    self._cat_features
                ),
            )

            self._models.append(
                model
            )

            self.ensemble_params_.append({
                "seed": ensemble_seed,
                "params": config,
            })

        self._model = (
            self._models[0]
        )

        if self.classification:

            self._classes = np.asarray(
                self._model.classes_
            )

        return self


    # ========================================================
    # Prediction
    # ========================================================

    @property
    def classes_(self):

        if not self.classification:
            return None

        return self._classes


    def predict_proba(
        self,
        X,
    ):

        if not self.classification:
            return None

        X, _ = self._prepare_X(
            X
        )

        probabilities = []

        for model in self._models:

            p = np.asarray(
                model.predict_proba(X)
            )

            model_classes = np.asarray(
                model.classes_
            )

            if np.array_equal(
                model_classes,
                self._classes,
            ):

                probabilities.append(
                    p
                )

            else:

                aligned = np.zeros(
                    (
                        len(X),
                        len(self._classes),
                    ),
                    dtype=np.float64,
                )

                for j, cls in enumerate(
                    model_classes
                ):

                    target_idx = np.where(
                        self._classes == cls
                    )[0][0]

                    aligned[
                        :,
                        target_idx,
                    ] = p[:, j]

                probabilities.append(
                    aligned
                )

        return np.mean(
            probabilities,
            axis=0,
        )


    def predict(
        self,
        X,
    ):

        if self.classification:

            probabilities = (
                self.predict_proba(X)
            )

            indices = np.argmax(
                probabilities,
                axis=1,
            )

            return self._classes[
                indices
            ]

        X, _ = self._prepare_X(
            X
        )

        predictions = [
            np.asarray(
                model.predict(X)
            ).reshape(-1)

            for model in self._models
        ]

        return np.mean(
            predictions,
            axis=0,
        )