import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def classification_metrics(
    y_true,
    y_pred,
    y_proba=None,
):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
        ),
    }

    if y_proba is not None:
        n_classes = y_proba.shape[1]

        if n_classes == 2:
            results["roc_auc"] = roc_auc_score(
                y_true,
                y_proba[:, 1],
            )
        else:
            results["roc_auc"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovo",
                average="macro",
            )

        results["log_loss"] = log_loss(
            y_true,
            y_proba,
        )

    return results


def regression_metrics(
    y_true,
    y_pred,
):
    return {
        "rmse": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "r2": r2_score(
            y_true,
            y_pred,
        ),
    }