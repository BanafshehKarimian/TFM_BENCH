import gc
import optuna
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .base import BaseTFM


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
    values = series.astype("string")
    missing = values.isna()
    values = values.astype(object)
    values.loc[missing] = None
    return values


def _device_type(device):
    if hasattr(device, "type"):
        return device.type
    return str(device).split(":")[0]


def _can_stratify(y):
    """True when every class has at least 2 examples."""
    y = _as_1d_y(y)
    _, counts = np.unique(y, return_counts=True)
    return len(counts) > 1 and counts.min() >= 2


def _make_tuning_indices(
    y,
    classification,
    validation_fraction,
    tune_max_rows,
    seed,
):
    """
    Build the HPO/calibration train-validation split.

    For classification:
      - labels are assumed to already be encoded to 0..K-1
      - every class is guaranteed to remain in TRAIN
      - singleton/very rare classes are therefore safe

    For regression:
      - regular random subsampling/splitting is used
    """

    y = _as_1d_y(y)
    n = len(y)

    if n < 2:
        raise ValueError(
            "Need at least 2 rows."
        )

    if not (
        0.0
        < float(validation_fraction)
        < 1.0
    ):
        raise ValueError(
            "validation_fraction must be in (0, 1)."
        )

    indices = np.arange(n)

    # ========================================================
    # REGRESSION
    # ========================================================

    if not classification:

        if (
            tune_max_rows is not None
            and n > int(tune_max_rows)
        ):

            rng = np.random.default_rng(
                seed
            )

            pool_indices = rng.choice(
                indices,
                size=int(tune_max_rows),
                replace=False,
            )

        else:

            pool_indices = indices

        train_idx, val_idx = (
            train_test_split(
                pool_indices,
                test_size=validation_fraction,
                random_state=seed,
            )
        )

        return (
            np.asarray(train_idx),
            np.asarray(val_idx),
        )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    rng = np.random.default_rng(
        seed
    )

    classes = np.unique(y)
    n_classes = len(classes)

    # --------------------------------------------------------
    # Decide HPO pool size
    # --------------------------------------------------------

    if tune_max_rows is None:
        pool_size = n
    else:
        pool_size = min(
            int(tune_max_rows),
            n,
        )

    if pool_size < n_classes:
        raise ValueError(
            f"HPO pool size {pool_size} "
            f"is smaller than number of "
            f"classes {n_classes}."
        )

    # --------------------------------------------------------
    # First guarantee at least one row per class in pool.
    # --------------------------------------------------------

    mandatory_pool = []

    for cls in classes:

        cls_indices = np.flatnonzero(
            y == cls
        )

        chosen = rng.choice(
            cls_indices
        )

        mandatory_pool.append(
            chosen
        )

    mandatory_pool = np.asarray(
        mandatory_pool,
        dtype=int,
    )

    # --------------------------------------------------------
    # Fill rest of pool
    # --------------------------------------------------------

    if pool_size > n_classes:

        used_mask = np.zeros(
            n,
            dtype=bool,
        )

        used_mask[
            mandatory_pool
        ] = True

        remaining_indices = indices[
            ~used_mask
        ]

        n_extra = (
            pool_size
            - n_classes
        )

        extra = rng.choice(
            remaining_indices,
            size=n_extra,
            replace=False,
        )

        pool_indices = np.concatenate([
            mandatory_pool,
            extra,
        ])

    else:

        pool_indices = (
            mandatory_pool.copy()
        )

    rng.shuffle(
        pool_indices
    )

    # --------------------------------------------------------
    # Now reserve one example of EACH class specifically
    # for the TRAIN side.
    # --------------------------------------------------------

    y_pool = y[
        pool_indices
    ]

    mandatory_train = []

    for cls in classes:

        cls_pool = pool_indices[
            y_pool == cls
        ]

        if len(cls_pool) == 0:
            raise RuntimeError(
                f"Class {cls} disappeared "
                f"from HPO pool."
            )

        mandatory_train.append(
            rng.choice(
                cls_pool
            )
        )

    mandatory_train = np.asarray(
        mandatory_train,
        dtype=int,
    )

    reserved_mask = np.zeros(
        n,
        dtype=bool,
    )

    reserved_mask[
        mandatory_train
    ] = True

    remaining_pool = pool_indices[
        ~reserved_mask[
            pool_indices
        ]
    ]

    # --------------------------------------------------------
    # Determine validation size.
    # --------------------------------------------------------

    target_val_size = int(
        round(
            pool_size
            * validation_fraction
        )
    )

    # train_test_split needs something left for train_extra.
    if len(remaining_pool) <= 1:

        raise ValueError(
            "Not enough rows to create "
            "a validation set."
        )

    target_val_size = min(
        target_val_size,
        len(remaining_pool) - 1,
    )

    target_val_size = max(
        1,
        target_val_size,
    )

    remaining_y = y[
        remaining_pool
    ]

    # Stratify the non-mandatory rows if possible.
    stratify = None

    if _can_stratify(
        remaining_y
    ):
        stratify = remaining_y

    try:

        train_extra, val_idx = (
            train_test_split(
                remaining_pool,
                test_size=target_val_size,
                random_state=seed,
                stratify=stratify,
            )
        )

    except ValueError:

        train_extra, val_idx = (
            train_test_split(
                remaining_pool,
                test_size=target_val_size,
                random_state=seed,
                stratify=None,
            )
        )

    train_idx = np.concatenate([
        mandatory_train,
        train_extra,
    ])

    rng.shuffle(
        train_idx
    )

    # --------------------------------------------------------
    # Safety check:
    # every class MUST exist in training.
    # --------------------------------------------------------

    train_classes = np.unique(
        y[train_idx]
    )

    if not np.array_equal(
        train_classes,
        classes,
    ):
        raise RuntimeError(
            "HPO training split lost one "
            "or more classes."
        )

    return (
        np.asarray(train_idx),
        np.asarray(val_idx),
    )


def _classification_score(model, X, y, metric):
    y = _as_1d_y(y)
    metric = str(metric).lower()

    if metric == "accuracy":
        pred = model.predict(X)
        value = accuracy_score(y, pred)
        return value, value

    if metric == "balanced_accuracy":
        pred = model.predict(X)
        value = balanced_accuracy_score(y, pred)
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
        classes = np.asarray(model.classes_)
        value = log_loss(
            y,
            proba,
            labels=classes,
        )
        return value, -value

    if metric == "roc_auc":
        proba = model.predict_proba(X)
        classes = np.asarray(model.classes_)

        if len(classes) == 2:
            y_binary = (y == classes[1]).astype(int)
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


def _regression_score(model, X, y, metric):
    y = _as_1d_y(y)
    pred = np.asarray(model.predict(X)).reshape(-1)
    metric = str(metric).lower()

    if metric == "rmse":
        value = np.sqrt(mean_squared_error(y, pred))
        return value, -value

    if metric == "mae":
        value = mean_absolute_error(y, pred)
        return value, -value

    if metric == "r2":
        value = r2_score(y, pred)
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


def _resolve_hpo_budget(
    n_rows,
    n_trials=None,
    tune_max_rows=None,
):
    n_rows = int(n_rows)

    if n_rows <= 200_000:
        default_rows = n_rows
        default_trials = 100
    elif n_rows <= 1_000_000:
        default_rows = 500_000
        default_trials = 75
    elif n_rows <= 5_000_000:
        default_rows = 750_000
        default_trials = 60
    else:
        default_rows = 1_000_000
        default_trials = 50

    if tune_max_rows is None:
        tune_max_rows = default_rows

    if n_trials is None:
        n_trials = default_trials

    tune_max_rows = min(
        max(2, int(tune_max_rows)),
        n_rows,
    )

    n_trials = max(1, int(n_trials))

    return tune_max_rows, n_trials


def _default_classification_metric(y):
    n_classes = len(np.unique(_as_1d_y(y)))
    return "roc_auc" if n_classes == 2 else "balanced_accuracy"


def _sample_custom_space(trial, search_space):
    params = {}

    for name, values in search_space.items():
        if callable(values):
            params[name] = values(trial)
        else:
            params[name] = trial.suggest_categorical(
                name,
                list(values),
            )

    return params


def _suggest_xgb_params(trial):
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.20,
            log=True,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            12,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            0.5,
            30.0,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.5,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.5,
            1.0,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-3,
            100.0,
            log=True,
        ),
        "gamma": trial.suggest_float(
            "gamma",
            1e-8,
            10.0,
            log=True,
        ),
        "max_bin": trial.suggest_categorical(
            "max_bin",
            [128, 256, 512],
        ),
    }


def _suggest_catboost_params(trial):
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.20,
            log=True,
        ),
        "depth": trial.suggest_int(
            "depth",
            4,
            10,
        ),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg",
            1e-2,
            100.0,
            log=True,
        ),
        "random_strength": trial.suggest_float(
            "random_strength",
            1e-3,
            10.0,
            log=True,
        ),
        "border_count": trial.suggest_categorical(
            "border_count",
            [64, 128, 254],
        ),
    }


def _xgb_early_stopping_metric(
    classification,
    tuning_metric,
    y_train,
):
    tuning_metric = str(tuning_metric).lower()

    if not classification:
        if tuning_metric == "mae":
            return "mae"
        return "rmse"

    n_classes = len(np.unique(_as_1d_y(y_train)))

    if n_classes == 2:
        if tuning_metric == "roc_auc":
            return "auc"
        return "logloss"
    return "mlogloss"


def _catboost_eval_metric(
    classification,
    tuning_metric,
    y_train,
):
    tuning_metric = str(tuning_metric).lower()

    if not classification:
        if tuning_metric == "mae":
            return "MAE"
        if tuning_metric == "r2":
            return "R2"
        return "RMSE"

    n_classes = len(np.unique(_as_1d_y(y_train)))

    if n_classes == 2:
        if tuning_metric == "roc_auc":
            return "AUC"
        if tuning_metric == "balanced_accuracy":
            return "BalancedAccuracy"
        if tuning_metric == "accuracy":
            return "Accuracy"
        if tuning_metric == "f1_macro":
            return "F1"
        return "Logloss"
    return "MultiClass"

class XGBoostAdapter(BaseTFM):
    def __init__(
        self,
        task,
        device,
        seed,
        # HPO
        tune=True,
        n_trials=None,
        validation_fraction=0.20,
        tune_max_rows=None,
        tuning_metric=None,
        search_space=None,
        # Stage-1 early stopping
        max_estimators=5000,
        early_stopping_rounds=100,

        # Stage-2 round calibration
        calibrate_rounds=True,
        round_calibration_fraction=0.20,

        # None = use ALL available training rows
        round_calibration_max_rows=None,

        # Final ensemble
        n_ensemble=5,
        ensemble_top_k=None,
        **kwargs,
    ):
        super().__init__(
            task=task,
            device=device,
            seed=seed,
        )

        self.classification = _is_classification(task)
        self.device_type = _device_type(device)

        self.tune = bool(tune)
        self.n_trials = n_trials
        self.validation_fraction = float(validation_fraction)
        self.tune_max_rows = tune_max_rows
        self.tuning_metric = tuning_metric
        self.search_space = search_space

        self.max_estimators = int(max_estimators)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.n_ensemble = int(n_ensemble)

        if self.max_estimators < 1:
            raise ValueError("max_estimators must be >= 1.")
        if self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be >= 1.")
        if self.n_ensemble < 1:
            raise ValueError("n_ensemble must be >= 1.")

        self.base_model_kwargs = dict(kwargs)

        self._category_levels = {}
        self._models = []
        self._model = None
        self._label_encoder = None
        self._classes = None

        self.best_params_ = None
        self.best_iteration_ = None
        self.best_n_estimators_ = None
        self.best_validation_score_ = None
        self.tuning_results_ = []
        self.ensemble_params_ = []
        self.study_ = None
        self.hpo_n_rows_ = None
        self.hpo_n_trials_ = None
        self.calibrate_rounds = bool(
            calibrate_rounds
        )

        self.round_calibration_fraction = float(
            round_calibration_fraction
        )

        self.round_calibration_max_rows = (
            round_calibration_max_rows
        )
        self.stage1_best_n_estimators_ = None

        self.round_calibration_n_rows_ = None
        self.round_calibration_score_ = None


    def _make_model(self, params, seed):
        from xgboost import XGBClassifier, XGBRegressor

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

        common_kwargs.update(self.base_model_kwargs)
        common_kwargs.update(params)

        if self.classification:
            return XGBClassifier(**common_kwargs)

        return XGBRegressor(**common_kwargs)

    def _fit_preprocessor(self, X):
        X = _as_dataframe(X)
        self._category_levels = {}

        for col in X.columns:
            col_name = str(col)

            if col_name.startswith("num_"):
                X[col] = _to_numeric(X[col])
                continue

            if col_name.startswith("cat_"):
                values = _to_string(X[col])
                categories = pd.Index(
                    values.dropna().unique(),
                    dtype=object,
                )
                self._category_levels[col] = categories
                X[col] = pd.Categorical(
                    values,
                    categories=categories,
                )
                continue

            try:
                X[col] = pd.to_numeric(
                    X[col],
                    errors="raise",
                ).astype(np.float32)
            except (ValueError, TypeError):
                values = _to_string(X[col])
                categories = pd.Index(
                    values.dropna().unique(),
                    dtype=object,
                )
                self._category_levels[col] = categories
                X[col] = pd.Categorical(
                    values,
                    categories=categories,
                )

        return X

    def _transform_X(self, X):
        X = _as_dataframe(X)

        for col in X.columns:
            if col in self._category_levels:
                values = _to_string(X[col])
                X[col] = pd.Categorical(
                    values,
                    categories=self._category_levels[col],
                )
            else:
                X[col] = _to_numeric(X[col])

        return X


    def _tune_parameters(self, X, y):
        y = _as_1d_y(y)

        tune_max_rows, n_trials = _resolve_hpo_budget(
            n_rows=len(y),
            n_trials=self.n_trials,
            tune_max_rows=self.tune_max_rows,
        )

        self.hpo_n_rows_ = tune_max_rows
        self.hpo_n_trials_ = n_trials

        print(
            f"[XGBoost HPO] rows={tune_max_rows:,}, "
            f"trials={n_trials}"
        )

        tuning_metric = self.tuning_metric
        if tuning_metric is None:
            if self.classification:
                tuning_metric = _default_classification_metric(y)
            else:
                tuning_metric = "rmse"

        self.tuning_metric = tuning_metric

        print(
            f"[XGBoost HPO] metric={tuning_metric}"
        )

        train_idx, val_idx = _make_tuning_indices(
            y=y,
            classification=self.classification,
            validation_fraction=self.validation_fraction,
            tune_max_rows=tune_max_rows,
            seed=self.seed,
        )

        X_raw = _as_dataframe(X)
        X_train_raw = X_raw.iloc[train_idx].copy()
        X_val_raw = X_raw.iloc[val_idx].copy()

        y_train = y[train_idx]
        y_val = y[val_idx]

        X_train = self._fit_preprocessor(X_train_raw)
        X_val = self._transform_X(X_val_raw)

        native_metric = _xgb_early_stopping_metric(
            classification=self.classification,
            tuning_metric=tuning_metric,
            y_train=y_train,
        )

        trial_results = []

        def objective(trial):
            if self.search_space is None:
                params = _suggest_xgb_params(trial)
            else:
                params = _sample_custom_space(
                    trial,
                    self.search_space,
                )

            model_params = dict(params)
            model_params["n_estimators"] = self.max_estimators
            model_params[
                "early_stopping_rounds"
            ] = self.early_stopping_rounds
            model_params["eval_metric"] = native_metric

            model = self._make_model(
                params=model_params,
                seed=self.seed,
            )

            try:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                raw_score, utility = _score_candidate(
                    model=model,
                    X=X_val,
                    y=y_val,
                    classification=self.classification,
                    metric=tuning_metric,
                )

                best_iteration = getattr(
                    model,
                    "best_iteration",
                    None,
                )

                if best_iteration is None:
                    best_n_estimators = self.max_estimators
                else:
                    best_n_estimators = int(best_iteration) + 1

                trial.set_user_attr(
                    "raw_score",
                    float(raw_score),
                )
                trial.set_user_attr(
                    "best_n_estimators",
                    int(best_n_estimators),
                )

                trial_results.append({
                    "trial": trial.number,
                    "params": dict(params),
                    "validation_score": float(raw_score),
                    "utility": float(utility),
                    "best_n_estimators": int(best_n_estimators),
                    "status": "success",
                })

                return float(utility)

            except Exception as exc:

                print(
                    f"\n[XGBoost HPO] Trial {trial.number} FAILED"
                )
                print(
                    f"{type(exc).__name__}: {exc}"
                )

                trial_results.append({
                    "trial": trial.number,
                    "params": dict(params),
                    "validation_score": None,
                    "utility": float("-inf"),
                    "best_n_estimators": None,
                    "status": "failed",
                    "error": repr(exc),
                })

                raise optuna.TrialPruned(
                    str(exc)
                )

            finally:
                del model
                gc.collect()

        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        self.study_ = study

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
        )

        completed = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        if not completed:

            failed_errors = [
                r["error"]
                for r in trial_results
                if r["status"] == "failed"
            ]

            unique_errors = list(
                dict.fromkeys(
                    failed_errors
                )
            )

            error_text = "\n".join(
                f"  - {e}"
                for e in unique_errors[:5]
            )

            raise RuntimeError(
                "All XGBoost tuning trials failed.\n"
                "Example underlying errors:\n"
                f"{error_text}"
            )

        best_trial = study.best_trial

        # selects STRUCTURAL hyperparameters.

        best_params = dict(
            best_trial.params
        )

        stage1_best_n_estimators = int(
            best_trial.user_attrs[
                "best_n_estimators"
            ]
        )

        self.stage1_best_n_estimators_ = (
            stage1_best_n_estimators
        )

        self.best_validation_score_ = float(
            best_trial.user_attrs[
                "raw_score"
            ]
        )

        self.tuning_results_ = (
            trial_results
        )

        print(
            f"[XGBoost HPO Stage 1] "
            f"best score="
            f"{self.best_validation_score_:.6f}"
        )

        print(
            f"[XGBoost HPO Stage 1] "
            f"temporary best trees="
            f"{stage1_best_n_estimators}"
        )

        print(
            f"[XGBoost HPO Stage 1] "
            f"best structural params="
            f"{best_params}"
        )

        return best_params

    def _calibrate_n_estimators(
        self,
        X,
        y,
        structural_params,
    ):
        """
        Stage 2 of HPO.

        Structural hyperparameters are FIXED.

        A larger/full training sample is split into
        calibration-train/calibration-validation.

        We then use early stopping once to determine the
        appropriate tree count at the real dataset scale.
        """

        y = _as_1d_y(y)

        calibration_max_rows = (
            self.round_calibration_max_rows
        )

        if calibration_max_rows is None:
            calibration_max_rows = len(y)

        calibration_max_rows = min(
            int(calibration_max_rows),
            len(y),
        )

        self.round_calibration_n_rows_ = (
            calibration_max_rows
        )

        print(
            f"[XGBoost HPO Stage 2] "
            f"Calibrating tree count on "
            f"{calibration_max_rows:,} rows"
        )

        train_idx, val_idx = (
            _make_tuning_indices(
                y=y,
                classification=(
                    self.classification
                ),
                validation_fraction=(
                    self.round_calibration_fraction
                ),
                tune_max_rows=(
                    calibration_max_rows
                ),
                seed=self.seed + 1,
            )
        )

        X_raw = _as_dataframe(X)

        X_train_raw = (
            X_raw.iloc[
                train_idx
            ].copy()
        )

        X_val_raw = (
            X_raw.iloc[
                val_idx
            ].copy()
        )

        y_train = y[
            train_idx
        ]

        y_val = y[
            val_idx
        ]

        print(
            f"[XGBoost HPO Stage 2] "
            f"train={len(y_train):,}, "
            f"val={len(y_val):,}"
        )


        X_train = (
            self._fit_preprocessor(
                X_train_raw
            )
        )

        X_val = (
            self._transform_X(
                X_val_raw
            )
        )


        native_metric = (
            _xgb_early_stopping_metric(
                classification=(
                    self.classification
                ),
                tuning_metric=(
                    self.tuning_metric
                ),
                y_train=y_train,
            )
        )

        model_params = dict(
            structural_params
        )

        model_params[
            "n_estimators"
        ] = self.max_estimators

        model_params[
            "early_stopping_rounds"
        ] = (
            self.early_stopping_rounds
        )

        model_params[
            "eval_metric"
        ] = native_metric

        model = self._make_model(
            params=model_params,
            seed=self.seed,
        )

        try:

            model.fit(
                X_train,
                y_train,
                eval_set=[
                    (
                        X_val,
                        y_val,
                    )
                ],
                verbose=False,
            )

            best_iteration = getattr(
                model,
                "best_iteration",
                None,
            )

            if best_iteration is None:

                best_n_estimators = (
                    self.max_estimators
                )

            else:

                best_n_estimators = (
                    int(best_iteration)
                    + 1
                )

            raw_score, _ = (
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

            self.round_calibration_score_ = (
                float(raw_score)
            )

        finally:

            del model
            gc.collect()


        final_params = dict(
            structural_params
        )

        final_params[
            "n_estimators"
        ] = int(
            best_n_estimators
        )

        self.best_n_estimators_ = int(
            best_n_estimators
        )

        self.best_iteration_ = (
            int(best_n_estimators)
            - 1
        )

        self.best_params_ = dict(
            final_params
        )

        print(
            f"[XGBoost HPO Stage 2] "
            f"score="
            f"{self.round_calibration_score_:.6f}"
        )

        print(
            f"[XGBoost HPO Stage 2] "
            f"best trees="
            f"{best_n_estimators}"
        )

        print(
            f"[XGBoost HPO] "
            f"final params="
            f"{final_params}"
        )

        return final_params


    def fit(self, X, y):

        y = _as_1d_y(y)
        if self.classification:

            self._label_encoder = (
                LabelEncoder()
            )

            y_model = (
                self._label_encoder
                .fit_transform(y)
            )

            # Keep ORIGINAL labels here.
            self._classes = np.asarray(
                self._label_encoder.classes_
            )

            print(
                f"[XGBoost] "
                f"{len(self._classes)} classes"
            )

            print(
                "[XGBoost] original classes:",
                self._classes,
            )

            print(
                "[XGBoost] encoded classes:",
                np.unique(y_model),
            )

        else:

            y_model = y

        # structural hyperparameter tuning

        if self.tune:

            structural_config = (
                self._tune_parameters(
                    X,
                    y_model,
                )
            )

            # recalibrate tree count at larger/full scale

            if self.calibrate_rounds:

                best_config = (
                    self._calibrate_n_estimators(
                        X,
                        y_model,
                        structural_config,
                    )
                )

            else:

                best_config = dict(
                    structural_config
                )

                best_config[
                    "n_estimators"
                ] = int(
                    self.stage1_best_n_estimators_
                )

                self.best_n_estimators_ = int(
                    self.stage1_best_n_estimators_
                )

                self.best_iteration_ = (
                    self.best_n_estimators_
                    - 1
                )

                self.best_params_ = dict(
                    best_config
                )

        else:

            best_config = {}
            self.best_params_ = {}

        # refit preprocessing on ALL available training rows

        X_full = (
            self._fit_preprocessor(
                X
            )
        )

        self._models = []
        self.ensemble_params_ = []

        # Final full-data ensemble

        for ensemble_id in range(
            self.n_ensemble
        ):

            ensemble_seed = (
                self.seed
                + 10_000
                + ensemble_id
            )

            model = self._make_model(
                params=best_config,
                seed=ensemble_seed,
            )

            model.fit(
                X_full,
                y_model,
            )

            self._models.append(
                model
            )

            self.ensemble_params_.append({
                "seed": ensemble_seed,
                "params": dict(
                    best_config
                ),
            })

        self._model = (
            self._models[0]
        )

        return self

    @property
    def classes_(self):
        if not self.classification:
            return None
        return self._classes

    def predict_proba(self, X):
        if not self.classification:
            return None

        if self._label_encoder is None:
            raise RuntimeError(
                "Label encoder is not fitted."
            )

        X = self._transform_X(X)

        probabilities = []

        n_classes = len(
            self._classes
        )

        for model in self._models:

            p = np.asarray(
                model.predict_proba(X)
            )

            # model.classes_ are encoded class IDs.
            encoded_classes = np.asarray(
                model.classes_,
                dtype=int,
            )

            aligned = np.zeros(
                (
                    len(X),
                    n_classes,
                ),
                dtype=np.float64,
            )

            for j, encoded_cls in enumerate(
                encoded_classes
            ):

                if not (
                    0
                    <= encoded_cls
                    < n_classes
                ):
                    raise RuntimeError(
                        f"Unexpected encoded class "
                        f"{encoded_cls}. "
                        f"Expected 0..{n_classes - 1}."
                    )

                aligned[
                    :,
                    encoded_cls,
                ] = p[:, j]

            probabilities.append(
                aligned
            )

        return np.mean(
            probabilities,
            axis=0,
        )


    def predict(self, X):

        if self.classification:

            probabilities = (
                self.predict_proba(X)
            )

            encoded_predictions = (
                np.argmax(
                    probabilities,
                    axis=1,
                )
            )

            # Convert 0..K-1 back to original labels.
            return (
                self._label_encoder
                .inverse_transform(
                    encoded_predictions
                )
            )

        X = self._transform_X(X)

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



class CatBoostAdapter(BaseTFM):
    def __init__(
        self,
        task,
        device,
        seed,
        # HPO
        tune=True,
        n_trials=None,
        validation_fraction=0.20,
        tune_max_rows=None,
        tuning_metric=None,
        search_space=None,
        # Early stopping
        max_iterations=5000,
        early_stopping_rounds=100,
        calibrate_rounds=True,
        round_calibration_fraction=0.20,
        round_calibration_max_rows=None,
        # Final ensemble
        n_ensemble=5,
        ensemble_top_k=None,
        **kwargs,
    ):
        super().__init__(
            task=task,
            device=device,
            seed=seed,
        )

        self.classification = _is_classification(task)
        self.device_type = _device_type(device)

        self.tune = bool(tune)
        self.n_trials = n_trials
        self.validation_fraction = float(validation_fraction)
        self.tune_max_rows = tune_max_rows
        self.tuning_metric = tuning_metric
        self.search_space = search_space

        self.max_iterations = int(max_iterations)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.n_ensemble = int(n_ensemble)

        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        if self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be >= 1.")
        if self.n_ensemble < 1:
            raise ValueError("n_ensemble must be >= 1.")

        self.base_model_kwargs = dict(kwargs)

        self._feature_kinds = {}
        self._cat_features = []

        self._models = []
        self._model = None
        self._label_encoder = None
        self._classes = None

        self.best_params_ = None
        self.best_iteration_ = None
        self.best_iterations_ = None
        self.best_validation_score_ = None
        self.tuning_results_ = []
        self.ensemble_params_ = []
        self.study_ = None
        self.hpo_n_rows_ = None
        self.hpo_n_trials_ = None
        self.calibrate_rounds = bool(
            calibrate_rounds
        )

        self.round_calibration_fraction = float(
            round_calibration_fraction
        )

        self.round_calibration_max_rows = (
            round_calibration_max_rows
        )

        self.stage1_best_iterations_ = None
        self.round_calibration_n_rows_ = None
        self.round_calibration_score_ = None


    def _make_model(self, params, seed):
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
            common_kwargs["task_type"] = "GPU"
        else:
            common_kwargs["task_type"] = "CPU"

        common_kwargs.update(self.base_model_kwargs)
        common_kwargs.update(params)

        if self.classification:
            return CatBoostClassifier(**common_kwargs)

        return CatBoostRegressor(**common_kwargs)

    @staticmethod
    def _as_cat_string(series):
        values = _to_string(series)
        values = pd.Series(
            values,
            index=series.index,
            dtype=object,
        )
        values = values.where(
            pd.notna(values),
            "__MISSING__",
        )
        return values.astype(str)

    def _fit_preprocessor(self, X):
        X = _as_dataframe(X)

        self._feature_kinds = {}
        self._cat_features = []

        for col in X.columns:
            col_name = str(col)

            if col_name.startswith("num_"):
                kind = "numeric"
            elif col_name.startswith("cat_"):
                kind = "categorical"
            else:
                numeric = pd.to_numeric(
                    X[col],
                    errors="coerce",
                )

                original_nonmissing = int(
                    X[col].notna().sum()
                )
                numeric_nonmissing = int(
                    numeric.notna().sum()
                )

                kind = (
                    "numeric"
                    if numeric_nonmissing
                    == original_nonmissing
                    else "categorical"
                )

            self._feature_kinds[col] = kind

            if kind == "categorical":
                self._cat_features.append(col)
                X[col] = self._as_cat_string(
                    X[col]
                )
            else:
                X[col] = _to_numeric(X[col])

        return X, list(self._cat_features)

    def _transform_X(self, X):
        X = _as_dataframe(X)

        if not self._feature_kinds:
            raise RuntimeError(
                "CatBoost preprocessor is not fitted."
            )

        missing_cols = [
            col
            for col in self._feature_kinds
            if col not in X.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing columns at transform time: {missing_cols}"
            )

        X = X[list(self._feature_kinds.keys())].copy()

        for col, kind in self._feature_kinds.items():
            if kind == "categorical":
                X[col] = self._as_cat_string(
                    X[col]
                )
            else:
                X[col] = _to_numeric(X[col])

        return X

    def _tune_parameters(self, X, y):
        y = _as_1d_y(y)

        tune_max_rows, n_trials = _resolve_hpo_budget(
            n_rows=len(y),
            n_trials=self.n_trials,
            tune_max_rows=self.tune_max_rows,
        )

        self.hpo_n_rows_ = tune_max_rows
        self.hpo_n_trials_ = n_trials

        print(
            f"[CatBoost HPO] rows={tune_max_rows:,}, "
            f"trials={n_trials}"
        )

        tuning_metric = self.tuning_metric
        if tuning_metric is None:
            if self.classification:
                tuning_metric = _default_classification_metric(y)
            else:
                tuning_metric = "rmse"

        self.tuning_metric = tuning_metric

        print(
            f"[CatBoost HPO] metric={tuning_metric}"
        )

        train_idx, val_idx = _make_tuning_indices(
            y=y,
            classification=self.classification,
            validation_fraction=self.validation_fraction,
            tune_max_rows=tune_max_rows,
            seed=self.seed,
        )

        X_raw = _as_dataframe(X)
        X_train_raw = X_raw.iloc[train_idx].copy()
        X_val_raw = X_raw.iloc[val_idx].copy()

        y_train = y[train_idx]
        y_val = y[val_idx]

        X_train, cat_features = self._fit_preprocessor(
            X_train_raw
        )
        X_val = self._transform_X(
            X_val_raw
        )

        eval_metric = _catboost_eval_metric(
            classification=self.classification,
            tuning_metric=tuning_metric,
            y_train=y_train,
        )

        trial_results = []

        def objective(trial):
            if self.search_space is None:
                params = _suggest_catboost_params(trial)
            else:
                params = _sample_custom_space(
                    trial,
                    self.search_space,
                )

            model_params = dict(params)
            model_params["iterations"] = self.max_iterations
            model_params["eval_metric"] = eval_metric

            model = self._make_model(
                params=model_params,
                seed=self.seed,
            )

            try:
                model.fit(
                    X_train,
                    y_train,
                    cat_features=cat_features,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=(
                        self.early_stopping_rounds
                    ),
                    use_best_model=True,
                    verbose=False,
                )

                raw_score, utility = _score_candidate(
                    model=model,
                    X=X_val,
                    y=y_val,
                    classification=self.classification,
                    metric=tuning_metric,
                )

                best_iteration = model.get_best_iteration()

                if (
                    best_iteration is None
                    or best_iteration < 0
                ):
                    best_iterations = self.max_iterations
                else:
                    best_iterations = int(best_iteration) + 1

                trial.set_user_attr(
                    "raw_score",
                    float(raw_score),
                )
                trial.set_user_attr(
                    "best_iterations",
                    int(best_iterations),
                )

                trial_results.append({
                    "trial": trial.number,
                    "params": dict(params),
                    "validation_score": float(raw_score),
                    "utility": float(utility),
                    "best_iterations": int(best_iterations),
                    "status": "success",
                })

                return float(utility)

            except Exception as exc:
                trial_results.append({
                    "trial": trial.number,
                    "params": dict(params),
                    "validation_score": None,
                    "utility": float("-inf"),
                    "best_iterations": None,
                    "status": "failed",
                    "error": repr(exc),
                })
                raise optuna.TrialPruned(str(exc))

            finally:
                del model
                gc.collect()

        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        self.study_ = study

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
        )

        completed = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        if not completed:
            raise RuntimeError(
                "All CatBoost tuning trials failed."
            )

        best_trial = study.best_trial
        best_params = dict(
            best_trial.params
        )

        stage1_best_iterations = int(
            best_trial.user_attrs[
                "best_iterations"
            ]
        )

        self.stage1_best_iterations_ = (
            stage1_best_iterations
        )

        self.best_validation_score_ = float(
            best_trial.user_attrs[
                "raw_score"
            ]
        )

        self.tuning_results_ = (
            trial_results
        )

        print(
            f"[CatBoost HPO Stage 1] "
            f"best score="
            f"{self.best_validation_score_:.6f}"
        )

        print(
            f"[CatBoost HPO Stage 1] "
            f"temporary best iterations="
            f"{stage1_best_iterations}"
        )

        print(
            f"[CatBoost HPO Stage 1] "
            f"best structural params="
            f"{best_params}"
        )

        return best_params


    def fit(self, X, y):

        y = _as_1d_y(y)

        if self.classification:

            self._label_encoder = (
                LabelEncoder()
            )

            y_model = (
                self._label_encoder
                .fit_transform(y)
            )

            self._classes = np.asarray(
                self._label_encoder.classes_
            )

            print(
                f"[CatBoost] "
                f"{len(self._classes)} classes"
            )

            print(
                "[CatBoost] original classes:",
                self._classes,
            )

            print(
                "[CatBoost] encoded classes:",
                np.unique(y_model),
            )

        else:

            y_model = y

        if self.tune:

            structural_config = (
                self._tune_parameters(
                    X,
                    y_model,
                )
            )


            if self.calibrate_rounds:

                best_config = (
                    self._calibrate_iterations(
                        X,
                        y_model,
                        structural_config,
                    )
                )

            else:

                best_config = dict(
                    structural_config
                )

                best_config[
                    "iterations"
                ] = int(
                    self.stage1_best_iterations_
                )

                self.best_iterations_ = int(
                    self.stage1_best_iterations_
                )

                self.best_iteration_ = (
                    self.best_iterations_
                    - 1
                )

                self.best_params_ = dict(
                    best_config
                )

        else:

            best_config = {}
            self.best_params_ = {}


        X_full, cat_features = (
            self._fit_preprocessor(
                X
            )
        )

        self._cat_features = (
            cat_features
        )

        self._models = []
        self.ensemble_params_ = []

        for ensemble_id in range(
            self.n_ensemble
        ):

            ensemble_seed = (
                self.seed
                + 10_000
                + ensemble_id
            )

            model = self._make_model(
                params=best_config,
                seed=ensemble_seed,
            )

            model.fit(
                X_full,
                y_model,
                cat_features=(
                    self._cat_features
                ),
                verbose=False,
            )

            self._models.append(
                model
            )

            self.ensemble_params_.append({
                "seed": ensemble_seed,
                "params": dict(
                    best_config
                ),
            })

        self._model = (
            self._models[0]
        )

        return self

    def _calibrate_iterations(
        self,
        X,
        y,
        structural_params,
    ):
        y = _as_1d_y(y)

        calibration_max_rows = (
            self.round_calibration_max_rows
        )

        if calibration_max_rows is None:
            calibration_max_rows = len(y)

        calibration_max_rows = min(
            int(calibration_max_rows),
            len(y),
        )

        self.round_calibration_n_rows_ = (
            calibration_max_rows
        )

        print(
            f"[CatBoost HPO Stage 2] "
            f"Calibrating iterations on "
            f"{calibration_max_rows:,} rows"
        )

        train_idx, val_idx = (
            _make_tuning_indices(
                y=y,
                classification=(
                    self.classification
                ),
                validation_fraction=(
                    self.round_calibration_fraction
                ),
                tune_max_rows=(
                    calibration_max_rows
                ),
                seed=self.seed + 1,
            )
        )

        X_raw = _as_dataframe(X)

        X_train_raw = (
            X_raw.iloc[
                train_idx
            ].copy()
        )

        X_val_raw = (
            X_raw.iloc[
                val_idx
            ].copy()
        )

        y_train = y[
            train_idx
        ]

        y_val = y[
            val_idx
        ]

        print(
            f"[CatBoost HPO Stage 2] "
            f"train={len(y_train):,}, "
            f"val={len(y_val):,}"
        )

        X_train, cat_features = (
            self._fit_preprocessor(
                X_train_raw
            )
        )

        X_val = (
            self._transform_X(
                X_val_raw
            )
        )

        eval_metric = (
            _catboost_eval_metric(
                classification=(
                    self.classification
                ),
                tuning_metric=(
                    self.tuning_metric
                ),
                y_train=y_train,
            )
        )

        model_params = dict(
            structural_params
        )

        model_params[
            "iterations"
        ] = self.max_iterations

        model_params[
            "eval_metric"
        ] = eval_metric

        model = self._make_model(
            params=model_params,
            seed=self.seed,
        )

        try:

            model.fit(
                X_train,
                y_train,

                cat_features=(
                    cat_features
                ),

                eval_set=(
                    X_val,
                    y_val,
                ),

                early_stopping_rounds=(
                    self.early_stopping_rounds
                ),

                use_best_model=True,

                verbose=False,
            )

            best_iteration = (
                model.get_best_iteration()
            )

            if (
                best_iteration is None
                or best_iteration < 0
            ):

                best_iterations = (
                    self.max_iterations
                )

            else:

                best_iterations = (
                    int(best_iteration)
                    + 1
                )

            raw_score, _ = (
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

            self.round_calibration_score_ = (
                float(raw_score)
            )

        finally:

            del model
            gc.collect()

        final_params = dict(
            structural_params
        )

        final_params[
            "iterations"
        ] = int(
            best_iterations
        )

        self.best_iterations_ = int(
            best_iterations
        )

        self.best_iteration_ = (
            int(best_iterations)
            - 1
        )

        self.best_params_ = dict(
            final_params
        )

        print(
            f"[CatBoost HPO Stage 2] "
            f"score="
            f"{self.round_calibration_score_:.6f}"
        )

        print(
            f"[CatBoost HPO Stage 2] "
            f"best iterations="
            f"{best_iterations}"
        )

        print(
            f"[CatBoost HPO] "
            f"final params="
            f"{final_params}"
        )

        return final_params

    @property
    def classes_(self):
        if not self.classification:
            return None
        return self._classes
    def predict_proba(self, X):

        if not self.classification:
            return None

        if self._label_encoder is None:
            raise RuntimeError(
                "Label encoder is not fitted."
            )

        X = self._transform_X(X)

        probabilities = []

        n_classes = len(
            self._classes
        )

        for model in self._models:

            p = np.asarray(
                model.predict_proba(X)
            )

            encoded_classes = np.asarray(
                model.classes_,
                dtype=int,
            )

            aligned = np.zeros(
                (
                    len(X),
                    n_classes,
                ),
                dtype=np.float64,
            )

            for j, encoded_cls in enumerate(
                encoded_classes
            ):

                if not (
                    0
                    <= encoded_cls
                    < n_classes
                ):
                    raise RuntimeError(
                        f"Unexpected encoded class "
                        f"{encoded_cls}. "
                        f"Expected 0..{n_classes - 1}."
                    )

                aligned[
                    :,
                    encoded_cls,
                ] = p[:, j]

            probabilities.append(
                aligned
            )

        return np.mean(
            probabilities,
            axis=0,
        )


    def predict(self, X):

        if self.classification:

            probabilities = (
                self.predict_proba(X)
            )

            encoded_predictions = (
                np.argmax(
                    probabilities,
                    axis=1,
                )
            )

            return (
                self._label_encoder
                .inverse_transform(
                    encoded_predictions
                )
            )

        X = self._transform_X(X)

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