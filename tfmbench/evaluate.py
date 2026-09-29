from dataclasses import dataclass
from typing import Any
import time

import numpy as np
import torch

from .metrics import (
    classification_metrics,
    regression_metrics,
)

from .models.registry import create_model


@dataclass
class EvalResult:
    model_name: str
    dataset_name: str
    task: str

    n_train: int
    n_test: int
    n_features: int

    metrics: dict[str, float]

    fit_seconds: float
    predict_seconds: float

    peak_gpu_memory_mb: float

    predictions: Any = None
    probabilities: Any = None


def _slice_rows(X, start, end):
    """
    Slice rows from pandas DataFrame/Series, NumPy arrays,
    torch tensors, or similar indexable objects.
    """

    if hasattr(X, "iloc"):
        return X.iloc[start:end]

    return X[start:end]


def _concat_outputs(outputs):
    """
    Concatenate outputs returned by model.predict()
    or model.predict_proba().
    """

    if not outputs:
        return None

    first = outputs[0]

    if torch.is_tensor(first):
        return torch.cat(outputs, dim=0)

    return np.concatenate(
        [np.asarray(x) for x in outputs],
        axis=0,
    )


def _predict_batched(
    model,
    X,
    batch_size,
    classification=False,
):
    """
    Run prediction in batches.

    For classification, returns:
        y_pred, y_proba

    For regression, returns:
        y_pred, None
    """

    n_samples = len(X)

    # ---------------------------------------------------------
    # No batching
    # ---------------------------------------------------------

    if batch_size is None:
        y_pred = model.predict(X)

        y_proba = None

        if classification:
            y_proba = model.predict_proba(X)

        return y_pred, y_proba

    if batch_size <= 0:
        raise ValueError(
            f"test_batch_size must be > 0 or None, "
            f"got {batch_size}"
        )

    # ---------------------------------------------------------
    # Batched inference
    # ---------------------------------------------------------

    pred_batches = []

    proba_batches = (
        []
        if classification
        else None
    )

    for start in range(
        0,
        n_samples,
        batch_size,
    ):

        end = min(
            start + batch_size,
            n_samples,
        )

        X_batch = _slice_rows(
            X,
            start,
            end,
        )

        print(
            f"[Predict] "
            f"{start:,}:{end:,} "
            f"/ {n_samples:,}"
        )

        # -------------------------
        # Prediction
        # -------------------------

        batch_pred = model.predict(
            X_batch
        )

        pred_batches.append(
            batch_pred
        )

        # -------------------------
        # Probabilities
        # -------------------------

        if classification:

            batch_proba = (
                model.predict_proba(
                    X_batch
                )
            )

            proba_batches.append(
                batch_proba
            )

    y_pred = _concat_outputs(
        pred_batches
    )

    y_proba = None

    if classification:
        y_proba = _concat_outputs(
            proba_batches
        )

    return y_pred, y_proba


def evaluate(
    model_name,
    data,
    device="cuda",
    seed=42,
    model_kwargs=None,
    return_predictions=False,
    tabpfn_token=None,
    test_batch_size=None,
):

    # ---------------------------------------------------------
    # TabPFN token
    # ---------------------------------------------------------

    if tabpfn_token:
        import os

        os.environ[
            "TABPFN_TOKEN"
        ] = tabpfn_token

    model_kwargs = (
        model_kwargs or {}
    )

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = create_model(
        model_name=model_name,
        task=data.task,
        device=device,
        seed=seed,
        **model_kwargs,
    )

    # ---------------------------------------------------------
    # Reset device memory statistics
    # ---------------------------------------------------------

    if device.type == "cuda":

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize()

    elif device.type == "mps":

        torch.mps.empty_cache()

        torch.mps.synchronize()

    # =========================================================
    # Fit
    # =========================================================

    start = time.perf_counter()

    model.fit(
        data.X_train,
        data.y_train,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()

    fit_seconds = (
        time.perf_counter()
        - start
    )

    # =========================================================
    # Prediction
    # =========================================================

    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    y_pred, y_proba = (
        _predict_batched(
            model=model,
            X=data.X_test,
            batch_size=test_batch_size,
            classification=(
                data.task
                == "classification"
            ),
        )
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()

    predict_seconds = (
        time.perf_counter()
        - start
    )

    # =========================================================
    # Metrics
    # =========================================================

    if data.task == "classification":

        metrics = (
            classification_metrics(
                data.y_test,
                y_pred,
                y_proba,
            )
        )

    else:

        metrics = (
            regression_metrics(
                data.y_test,
                y_pred,
            )
        )

    # =========================================================
    # Memory
    # =========================================================

    peak_gpu_memory_mb = None

    if device.type == "cuda":

        peak_gpu_memory_mb = (
            torch.cuda
            .max_memory_allocated()
            / 1024**2
        )

    # Note:
    # PyTorch currently does not provide an equivalent
    # max_memory_allocated() metric for MPS.

    # =========================================================
    # Result
    # =========================================================

    return EvalResult(
        model_name=model_name,
        dataset_name=data.name,
        task=data.task,

        n_train=data.n_train,
        n_test=data.n_test,
        n_features=data.n_features,

        metrics=metrics,

        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,

        peak_gpu_memory_mb=(
            peak_gpu_memory_mb
        ),

        predictions=(
            y_pred
            if return_predictions
            else None
        ),

        probabilities=(
            y_proba
            if return_predictions
            else None
        ),
    )