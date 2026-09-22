from dataclasses import dataclass
from typing import Any
import time

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



def evaluate(
    model_name,
    data,
    device="cuda",
    seed=42,
    model_kwargs=None,
    return_predictions=False,
    tabpfn_token=None,
):
    
    if tabpfn_token:
        import os
        os.environ["TABPFN_TOKEN"] = tabpfn_token #https://ux.priorlabs.ai/accept-license?hf_repo_id=tabpfn_3
    model_kwargs = model_kwargs or {}

    model = create_model(
        model_name=model_name,
        task=data.task,
        device=device,
        seed=seed,
        **model_kwargs,
    )


    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.empty_cache()
        torch.mps.synchronize()

    start = time.perf_counter()

    model.fit(
        data.X_train,
        data.y_train,
    )

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

    fit_seconds = time.perf_counter() - start
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

    start = time.perf_counter()

    y_pred = model.predict(
        data.X_test
    )

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

    predict_seconds = time.perf_counter() - start

    y_proba = None

    if data.task == "classification":
        y_proba = model.predict_proba(
            data.X_test
        )
        metrics = classification_metrics(
            data.y_test,
            y_pred,
            y_proba,
        )

    else:
        metrics = regression_metrics(
            data.y_test,
            y_pred,
        )

    peak_gpu_memory_mb = None

    if device.type == 'cuda':
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )#fix for gpu and mps?
        

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

        peak_gpu_memory_mb=peak_gpu_memory_mb,

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