import pandas as pd


def benchmark_models(
    data,
    models=None,
    device="cuda",
    seed=42,
    tabpfn_token=None,
    model_kwargs=None,
    continue_on_error=True,
    sort_by=None,
):
    
    from tfmbench.evaluate import evaluate

    model_kwargs = model_kwargs or {}

    rows = []

    for model_name in models:

        print(f"Running {model_name}...")

        kwargs = model_kwargs.get(model_name, {})

        result = evaluate(
            model_name=model_name,
            data=data,
            device=device,
            seed=seed,
            model_kwargs=kwargs,
            return_predictions=False,
            tabpfn_token=tabpfn_token,
        )

        row = {
            "model": model_name,
            "status": "ok",
            "n_train": result.n_train,
            "n_test": result.n_test,
            "n_features": result.n_features,
            "fit_seconds": result.fit_seconds,
            "predict_seconds": result.predict_seconds,
            "total_seconds": (
                result.fit_seconds
                + result.predict_seconds
            ),

            "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
        }
            
        row.update(result.metrics)

        rows.append(row)

    df = pd.DataFrame(rows)

    if sort_by is not None and sort_by in df.columns:

        lower_is_better = {
            "rmse",
            "mae",
            "log_loss",
            "fit_seconds",
            "predict_seconds",
            "total_seconds",
            "peak_gpu_memory_mb",
        }

        ascending = sort_by in lower_is_better

        df = df.sort_values(
            by=sort_by,
            ascending=ascending,
            na_position="last",
        )

    return df.reset_index(drop=True)