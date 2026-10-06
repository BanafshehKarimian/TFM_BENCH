import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Summary tables
# ============================================================

# Exact large-scale methods used in the comparison.
# Change these IDs here if you later switch versions.
SUMMARY_MODELS = {
    "XGBoost": "xgboost",
    "CatBoost": "catboost",
    "TabPFN": "tabpfn_v3.5_fast",
    "TabICL": "tabicl_v2",
}


SUMMARY_METRICS = {
    "regression": [
        "rmse",
        "mae",
        "r2",
    ],

    "classification": [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "roc_auc",
        "log_loss",
    ],
}


SUMMARY_METRIC_LABELS = {
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R2",

    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1_macro": "Macro F1",
    "roc_auc": "ROC-AUC",
    "log_loss": "Log Loss",
}


SUMMARY_DATASET_LABELS = {
    "Airlines_DepDelay_10M": "Airlines",
    "US_Accidents_March23": "US Accidents",
    "delays_zurich_transport": "Zurich Transport",
    "friedman1": "Friedman1",
    "subset_higgs": "HIGGS",
    "KDDCup99": "KDDCup99",
    "mimic_extract_los_3": "MIMIC LOS-3",
    "sf-police-incidents": "SF Police",
    "jigsaw-unintended-bias-in-toxicity": "Jigsaw",
    "world_food_wealth_bank": "World Food",
}

# ============================================================
# Configuration
# ============================================================
# ============================================================
# Global visual identity for models
#
# IMPORTANT:
# A model keeps the SAME color in every figure/subplot,
# even if some other models are missing from a dataset.
# ============================================================

MODEL_COLORS = {
    "xgboost": "#4C78A8",
    "catboost": "#F58518",

    "tabpfn_v3": "#54A24B",
    "tabpfn_v3.5": "#E45756",
    "tabpfn_v3.5_fast": "#B279A2",

    "tabicl_v1": "#FF9DA6",
    "tabicl_v2": "#72B7B2",

    "tabdpt_v1.3": "#9D755D",
}


MODEL_MARKERS = {
    "xgboost": "o",
    "catboost": "s",

    "tabpfn_v3": "^",
    "tabpfn_v3.5": "D",
    "tabpfn_v3.5_fast": "P",

    "tabicl_v1": "v",
    "tabicl_v2": "X",

    "tabdpt_v1.3": "*",
}


# Preferred order in legends.
MODEL_ORDER = [
    "xgboost",
    "catboost",
    "tabpfn_v3",
    "tabpfn_v3.5",
    "tabpfn_v3.5_fast",
    "tabicl_v1",
    "tabicl_v2",
    "tabdpt_v1.3",
]



# ============================================================
# Final LaTeX summary tables
# ============================================================

def pretty_summary_dataset_name(dataset):
    return SUMMARY_DATASET_LABELS.get(
        dataset,
        dataset.replace("_", " "),
    )


def get_largest_available_result(
    df,
    dataset,
    model,
    metric,
):
    """
    Get the metric value from the LARGEST AVAILABLE
    training-set size for this dataset/model/metric.

    This does NOT choose the best metric value.

    Example:
        10K
        25K
        100K
        500K
        750K

    -> selects 750K.

    If a true full run exists at a larger N, selects full.
    """

    rows = df[
        (df["dataset"] == dataset)
        & (df["model"] == model)
        & (df["metric"] == metric)
    ].copy()

    if rows.empty:
        return np.nan, np.nan

    rows = rows[
        np.isfinite(
            rows["value"].astype(float)
        )
    ]

    if rows.empty:
        return np.nan, np.nan

    # Sort by training size.
    #
    # is_full is secondary:
    # if n_train happens to tie, prefer the explicitly
    # marked full-data result.
    rows = rows.sort_values(
        [
            "n_train",
            "is_full",
        ],
        ascending=[
            True,
            True,
        ],
    )

    selected = rows.iloc[-1]

    return (
        float(
            selected["value"]
        ),
        int(
            selected["n_train"]
        ),
    )


def build_summary_table(
    df,
    task_group,
):
    """
    Build:

                          Dataset A  Dataset B ... Average
    Metric  XGBoost
            CatBoost
            TabPFN
            TabICL

    For each model/dataset, use the result at the
    MAXIMUM EXISTING TRAINING SIZE.

    The Average column is computed only over datasets for which
    ALL FOUR methods have a result for that metric. This makes
    averages directly comparable between rows.
    """

    task_df = df[
        df["task_group"]
        == task_group
    ].copy()

    if task_df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    # --------------------------------------------------------
    # Dataset columns
    # --------------------------------------------------------

    datasets = sorted(
        task_df[
            "dataset"
        ].unique()
    )

    # --------------------------------------------------------
    # Only metrics relevant to this task and actually present
    # somewhere in the results.
    # --------------------------------------------------------

    existing_metrics = set(
        task_df[
            "metric"
        ].unique()
    )

    metrics = [
        metric
        for metric in SUMMARY_METRICS[
            task_group
        ]
        if metric in existing_metrics
    ]

    table_rows = []
    table_index = []

    # Same shape as value table, but stores which N was chosen.
    n_rows = []
    n_index = []

    # ========================================================
    # One block per metric
    # ========================================================

    for metric in metrics:

        values_by_model = {}
        n_by_model = {}

        # ----------------------------------------------------
        # First collect values for all four models.
        # ----------------------------------------------------

        for display_name, model_id in (
            SUMMARY_MODELS.items()
        ):

            values = []
            selected_ns = []

            for dataset in datasets:

                value, selected_n = (
                    get_largest_available_result(
                        task_df,
                        dataset=dataset,
                        model=model_id,
                        metric=metric,
                    )
                )

                values.append(
                    value
                )

                selected_ns.append(
                    selected_n
                )

            values_by_model[
                display_name
            ] = values

            n_by_model[
                display_name
            ] = selected_ns

        # ----------------------------------------------------
        # Fair average:
        #
        # Only datasets for which all FOUR methods have
        # a result for this metric contribute to Average.
        #
        # This avoids:
        #
        #   XGBoost average = 6 datasets
        #   TabPFN average  = 3 datasets
        #
        # which would not be directly comparable.
        # ----------------------------------------------------

        common_positions = []

        for dataset_idx in range(
            len(datasets)
        ):

            all_available = all(
                np.isfinite(
                    values_by_model[
                        model_name
                    ][
                        dataset_idx
                    ]
                )
                for model_name
                in SUMMARY_MODELS
            )

            if all_available:
                common_positions.append(
                    dataset_idx
                )

        # ----------------------------------------------------
        # Create four table rows.
        # ----------------------------------------------------

        metric_label = (
            SUMMARY_METRIC_LABELS.get(
                metric,
                pretty_metric_name(metric),
            )
        )

        for display_name in (
            SUMMARY_MODELS.keys()
        ):

            values = values_by_model[
                display_name
            ]

            selected_ns = n_by_model[
                display_name
            ]

            if common_positions:

                average = float(
                    np.mean(
                        [
                            values[i]
                            for i
                            in common_positions
                        ]
                    )
                )

            else:

                average = np.nan

            table_rows.append(
                values
                + [
                    average
                ]
            )

            table_index.append(
                (
                    metric_label,
                    display_name,
                )
            )

            # N table is useful for checking exactly what
            # "maximum existing" meant in each cell.
            n_rows.append(
                selected_ns
                + [
                    np.nan
                ]
            )

            n_index.append(
                (
                    metric_label,
                    display_name,
                )
            )

    # ========================================================
    # Build DataFrames
    # ========================================================

    columns = [
        pretty_summary_dataset_name(
            dataset
        )
        for dataset in datasets
    ] + [
        "Average"
    ]

    index = pd.MultiIndex.from_tuples(
        table_index,
        names=[
            "Metric",
            "Model",
        ],
    )

    table = pd.DataFrame(
        table_rows,
        index=index,
        columns=columns,
    )

    n_table = pd.DataFrame(
        n_rows,
        index=pd.MultiIndex.from_tuples(
            n_index,
            names=[
                "Metric",
                "Model",
            ],
        ),
        columns=columns,
    )

    return (
        table,
        n_table,
    )


def save_one_latex_summary(
    table,
    path,
    caption,
    label,
):
    """
    Save DataFrame as a clean booktabs-style LaTeX table.
    """

    if table.empty:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    latex = table.to_latex(
        na_rep="N/A",
        float_format=lambda x: (
            f"{x:.4f}"
        ),
        index=True,
        escape=True,
        multicolumn=True,
        multirow=True,
        caption=caption,
        label=label,
        position="t",
        column_format=(
            "ll"
            + "r" * len(
                table.columns
            )
        ),
    )

    path.write_text(
        latex,
        encoding="utf-8",
    )


def save_summary_tables(
    df,
    output_dir,
):
    """
    Generate:

        tables/
            regression_summary.tex
            classification_summary.tex

            regression_summary.csv
            classification_summary.csv

            regression_selected_n.csv
            classification_selected_n.csv

            summary_tables.tex

    selected_n CSVs are for sanity checking which training
    size supplied every table entry.
    """

    table_dir = (
        output_dir
        / "tables"
    )

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_latex = []

    # ========================================================
    # Regression
    # ========================================================

    regression_table, regression_n = (
        build_summary_table(
            df,
            task_group="regression",
        )
    )

    if not regression_table.empty:

        regression_table.to_csv(
            table_dir
            / "regression_summary.csv",
            na_rep="N/A",
        )

        regression_n.to_csv(
            table_dir
            / "regression_selected_n.csv",
            na_rep="N/A",
        )

        regression_tex_path = (
            table_dir
            / "regression_summary.tex"
        )

        save_one_latex_summary(
            table=regression_table,
            path=regression_tex_path,
            caption=(
                "Regression performance at the "
                "largest available training-set size "
                "for each method and dataset."
            ),
            label=(
                "tab:large_scale_regression"
            ),
        )

        combined_latex.append(
            "% ================================\n"
            "% Regression\n"
            "% ================================\n\n"
            + regression_tex_path.read_text(
                encoding="utf-8"
            )
        )

    # ========================================================
    # Classification
    # ========================================================

    classification_table, classification_n = (
        build_summary_table(
            df,
            task_group="classification",
        )
    )

    if not classification_table.empty:

        classification_table.to_csv(
            table_dir
            / "classification_summary.csv",
            na_rep="N/A",
        )

        classification_n.to_csv(
            table_dir
            / "classification_selected_n.csv",
            na_rep="N/A",
        )

        classification_tex_path = (
            table_dir
            / "classification_summary.tex"
        )

        save_one_latex_summary(
            table=classification_table,
            path=classification_tex_path,
            caption=(
                "Classification performance at the "
                "largest available training-set size "
                "for each method and dataset."
            ),
            label=(
                "tab:large_scale_classification"
            ),
        )

        combined_latex.append(
            "% ================================\n"
            "% Classification\n"
            "% ================================\n\n"
            + classification_tex_path.read_text(
                encoding="utf-8"
            )
        )

    # ========================================================
    # Combined file containing both tables
    # ========================================================

    if combined_latex:

        (
            table_dir
            / "summary_tables.tex"
        ).write_text(
            "\n\n".join(
                combined_latex
            ),
            encoding="utf-8",
        )

    # ========================================================
    # Console output
    # ========================================================

    print()
    print("=" * 80)
    print("SUMMARY TABLES")
    print("=" * 80)

    if not regression_table.empty:

        print()
        print("REGRESSION")
        print(
            regression_table.to_string(
                na_rep="N/A",
                float_format=lambda x: (
                    f"{x:.4f}"
                ),
            )
        )

    if not classification_table.empty:

        print()
        print("CLASSIFICATION")
        print(
            classification_table.to_string(
                na_rep="N/A",
                float_format=lambda x: (
                    f"{x:.4f}"
                ),
            )
        )

    return {
        "regression": regression_table,
        "classification": classification_table,
        "regression_n": regression_n,
        "classification_n": classification_n,
    }
def infer_task_group(record, metrics):
    """
    Normalize task names to:
        classification
        regression

    Uses record["task"] when available.
    Falls back to metric names otherwise.
    """

    raw_task = str(
        record.get(
            "task",
            ""
        )
    ).lower()

    if raw_task in {
        "classification",
        "binclass",
        "multiclass",
        "binary",
        "binary_classification",
        "multiclass_classification",
    }:
        return "classification"

    if raw_task == "regression":
        return "regression"

    # Fallback for older JSON files that may not contain task.
    metric_names = {
        str(m).lower()
        for m in metrics.keys()
    }

    regression_metrics = {
        "rmse",
        "mae",
        "mse",
        "r2",
    }

    classification_metrics = {
        "accuracy",
        "balanced_accuracy",
        "f1",
        "f1_macro",
        "roc_auc",
        "auc",
        "log_loss",
        "logloss",
    }

    if metric_names & regression_metrics:
        return "regression"

    if metric_names & classification_metrics:
        return "classification"

    return None
    
def model_sort_key(model):
    """
    Stable global model ordering.
    Unknown models go at the end alphabetically.
    """

    try:
        return (
            0,
            MODEL_ORDER.index(model),
        )
    except ValueError:
        return (
            1,
            str(model),
        )


def get_model_color(model):
    """
    Return globally consistent color for a model.

    Known models use MODEL_COLORS.
    Unknown models receive a deterministic fallback color.
    """

    if model in MODEL_COLORS:
        return MODEL_COLORS[model]

    # Deterministic fallback.
    import hashlib

    palette = [
        "#59A14F",
        "#EDC948",
        "#AF7AA1",
        "#76B7B2",
        "#E15759",
        "#BAB0AC",
        "#FF9DA7",
        "#9C755F",
    ]

    digest = hashlib.md5(
        str(model).encode("utf-8")
    ).hexdigest()

    index = int(
        digest[:8],
        16,
    ) % len(palette)

    return palette[index]


def get_model_marker(model):
    return MODEL_MARKERS.get(
        model,
        "o",
    )

# Metrics for which LOWER is better.
# Everything else is assumed higher-is-better.
LOWER_IS_BETTER = {
    "rmse",
    "mae",
    "mse",
    "log_loss",
    "logloss",
    "cross_entropy",
    "nll",
}

# Pretty names for known models.
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "tabpfn_v3": "TabPFN v3",
    "tabpfn_v3.5": "TabPFN v3.5",
    "tabpfn_v3.5_fast": "TabPFN v3.5-fast",
    "tabicl_v1": "TabICL v1",
    "tabicl_v2": "TabICL v2",
    "tabdpt_v1.3": "TabDPT v1.3",
}

# Raw averaging makes sense for scale-comparable metrics.
# For RMSE/MAE across unrelated datasets, normalized average
# is usually more meaningful.
RAW_AVERAGE_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "f1",
    "f1_macro",
    "roc_auc",
    "auc",
    "log_loss",
    "logloss",
    "r2",
}

# Draw uncertainty bands on average plots.
DRAW_SEM = True

# Minimum number of datasets required for an average point.
MIN_DATASETS_FOR_AVERAGE = 2

# Average only over datasets for which ALL models available
# at a given N have a result for that metric.
#
# This is safer than allowing each model to average over a
# completely different set of datasets.
USE_COMMON_DATASETS_PER_SIZE = True


# ============================================================
# Helpers
# ============================================================

def safe_float(value):
    """
    Convert a value to finite float.
    Returns None for invalid/NaN/inf values.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return value


def pretty_model_name(model):
    return MODEL_LABELS.get(
        model,
        model.replace("_", " "),
    )


def pretty_metric_name(metric):
    replacements = {
        "roc_auc": "ROC-AUC",
        "auc": "AUC",
        "f1_macro": "Macro F1",
        "f1": "F1",
        "balanced_accuracy": "Balanced Accuracy",
        "accuracy": "Accuracy",
        "log_loss": "Log Loss",
        "logloss": "Log Loss",
        "rmse": "RMSE",
        "mae": "MAE",
        "mse": "MSE",
        "r2": "R²",
    }

    return replacements.get(
        metric.lower(),
        metric.replace("_", " ").title(),
    )


def format_n(value):
    """
    Human-readable training size.
    """
    value = float(value)

    if value >= 1_000_000:
        x = value / 1_000_000

        if abs(x - round(x)) < 1e-8:
            return f"{int(round(x))}M"

        return f"{x:.2g}M"

    if value >= 1_000:
        x = value / 1_000

        if abs(x - round(x)) < 1e-8:
            return f"{int(round(x))}K"

        return f"{x:.2g}K"

    return f"{int(value)}"


def metric_is_lower_better(metric):
    metric = metric.lower()

    if metric in LOWER_IS_BETTER:
        return True

    # Some custom metric names may contain these strings.
    if "rmse" in metric:
        return True

    if metric == "mae" or metric.endswith("_mae"):
        return True

    if "log_loss" in metric or "logloss" in metric:
        return True

    return False


# ============================================================
# Filename parsing
# ============================================================

def parse_result_filename(path):
    """
    Expected format:

        dataset__model__n10000.json

    or:

        dataset__model__full.json

    We split from the RIGHT so underscores/hyphens inside
    dataset/model names are fine.
    """

    stem = path.stem

    parts = stem.rsplit("__", 2)

    if len(parts) != 3:
        return None

    dataset, model, size_token = parts

    if size_token == "full":
        return {
            "dataset": dataset,
            "model": model,
            "size_token": size_token,
            "is_full": True,
            "sample_size": None,
        }

    match = re.fullmatch(
        r"n(\d+)",
        size_token,
    )

    if match is None:
        return None

    return {
        "dataset": dataset,
        "model": model,
        "size_token": size_token,
        "is_full": False,
        "sample_size": int(match.group(1)),
    }


# ============================================================
# JSON loading
# ============================================================

def infer_full_train_size(record):
    """
    For __full.json, try several possible metadata fields.

    IMPORTANT:
    We do NOT invent a location for "full" if the true
    number of rows is unavailable.
    """

    possible_keys = [
        "n_train",
        "actual_train_size",
        "original_train_size",
        "full_train_size",
        "train_size",
    ]

    for key in possible_keys:
        if key in record:
            value = safe_float(record[key])

            if value is not None and value > 0:
                return int(value)

    # Sometimes metadata may be nested.
    for parent_key in [
        "data",
        "metadata",
        "dataset_info",
    ]:
        nested = record.get(parent_key)

        if not isinstance(nested, dict):
            continue

        for key in possible_keys:
            if key not in nested:
                continue

            value = safe_float(
                nested[key]
            )

            if value is not None and value > 0:
                return int(value)

    return None


def load_all_results(results_dir):
    """
    Return one long DataFrame:

    dataset | model | n_train | metric | value | is_full | file
    """

    rows = []

    json_files = sorted(
        results_dir.glob("*.json")
    )

    print(
        f"Found {len(json_files):,} JSON files."
    )

    skipped_failed = 0
    skipped_bad_name = 0
    skipped_bad_json = 0
    skipped_full_without_size = 0

    for path in json_files:

        parsed = parse_result_filename(path)

        if parsed is None:
            skipped_bad_name += 1
            continue

        try:
            with open(path, "r") as f:
                record = json.load(f)
        except Exception as exc:
            print(
                f"[WARNING] Could not read "
                f"{path.name}: {exc}"
            )
            skipped_bad_json += 1
            continue

        status = str(
            record.get(
                "status",
                "success",
            )
        ).lower()

        if status != "success":
            skipped_failed += 1
            continue

        metrics = record.get(
            "metrics"
        )

        task_group = infer_task_group(
            record,
            metrics,
        )

        if not isinstance(metrics, dict):
            print(
                f"[WARNING] No metric dictionary in "
                f"{path.name}"
            )
            continue

        # ----------------------------------------------------
        # Determine numeric x-axis location
        # ----------------------------------------------------

        if parsed["is_full"]:

            n_train = infer_full_train_size(
                record
            )

            if n_train is None:

                print(
                    f"[WARNING] Skipping FULL point "
                    f"because true n_train is missing: "
                    f"{path.name}"
                )

                skipped_full_without_size += 1
                continue

        else:
            n_train = parsed[
                "sample_size"
            ]

        # ----------------------------------------------------
        # Add every numeric metric
        # ----------------------------------------------------

        for metric, value in metrics.items():

            value = safe_float(value)

            if value is None:
                continue

            rows.append({
                "dataset": parsed["dataset"],
                "task_group": task_group,
                "model": parsed["model"],
                "n_train": int(n_train),
                "metric": str(metric),
                "value": value,
                "is_full": bool(
                    parsed["is_full"]
                ),
                "file": path.name,
            })

    df = pd.DataFrame(rows)

    print()
    print("Load summary")
    print("-" * 60)
    print(
        f"Valid metric rows       : {len(df):,}"
    )
    print(
        f"Failed runs skipped     : {skipped_failed:,}"
    )
    print(
        f"Bad filenames skipped   : {skipped_bad_name:,}"
    )
    print(
        f"Bad JSON skipped        : {skipped_bad_json:,}"
    )
    print(
        f"Full w/o size skipped   : "
        f"{skipped_full_without_size:,}"
    )

    if not df.empty:
        print(
            f"Datasets                : "
            f"{df['dataset'].nunique()}"
        )
        print(
            f"Models                  : "
            f"{df['model'].nunique()}"
        )
        print(
            f"Metrics                 : "
            f"{df['metric'].nunique()}"
        )

    return df


# ============================================================
# Plot utilities
# ============================================================

def setup_axis(ax, metric):
    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "Number of training examples"
    )

    ax.set_ylabel(
        pretty_metric_name(metric)
    )

    ax.grid(
        True,
        alpha=0.20,
        which="both",
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )


def set_training_size_ticks(ax, x_values):
    """
    Use only actual experiment sizes as x ticks.
    """

    x_values = sorted(
        set(
            int(x)
            for x in x_values
            if x > 0
        )
    )

    if not x_values:
        return

    ax.set_xticks(
        x_values
    )

    ax.set_xticklabels(
        [
            format_n(x)
            for x in x_values
        ],
        rotation=0,
    )


def save_figure(
    fig,
    output_base,
    dpi=250,
):
    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_base.with_suffix(
            ".png"
        ),
        dpi=dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        output_base.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Per-dataset plots
# ============================================================

def plot_dataset_metric(
    df_metric,
    dataset,
    metric,
    output_dir,
):
    """
    One dataset + one metric.

    Each model = one line.
    """

    fig, ax = plt.subplots(
        figsize=(7.5, 4.8)
    )

    plotted = False

    models = sorted(
        df_metric["model"].unique(),
        key=model_sort_key,
    )

    for model in models:

        model_df = (
            df_metric[
                df_metric["model"]
                == model
            ]
            .sort_values(
                "n_train"
            )
            .copy()
        )

        if model_df.empty:
            continue

        ax.plot(
            model_df["n_train"],
            model_df["value"],
            marker=get_model_marker(model),
            color=get_model_color(model),
            linewidth=2.2,
            markersize=5.5,
            label=pretty_model_name(model),
        )

        plotted = True

        # --------------------------------------------
        # Mark true "full" points.
        # --------------------------------------------

        full_rows = model_df[
            model_df["is_full"]
        ]

        for _, row in full_rows.iterrows():

            ax.annotate(
                "Full",
                xy=(
                    row["n_train"],
                    row["value"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

    if not plotted:
        plt.close(fig)
        return

    setup_axis(
        ax,
        metric,
    )

    set_training_size_ticks(
        ax,
        df_metric["n_train"],
    )

    direction = (
        "↓ lower is better"
        if metric_is_lower_better(
            metric
        )
        else "↑ higher is better"
    )

    ax.set_title(
        f"{dataset}\n"
        f"{pretty_metric_name(metric)} "
        f"({direction})"
    )

    ax.legend(
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()

    safe_metric = (
        metric
        .replace("/", "_")
        .replace(" ", "_")
    )

    save_figure(
        fig,
        output_dir
        / "by_dataset"
        / dataset
        / safe_metric,
    )


def make_all_dataset_plots(
    df,
    output_dir,
):
    datasets = sorted(
        df["dataset"].unique()
    )

    print()
    print(
        f"Creating per-dataset plots "
        f"for {len(datasets)} datasets..."
    )

    for dataset in datasets:

        dataset_df = df[
            df["dataset"] == dataset
        ]

        metrics = sorted(
            dataset_df[
                "metric"
            ].unique()
        )

        print(
            f"  {dataset}: "
            f"{len(metrics)} metrics"
        )

        for metric in metrics:

            metric_df = dataset_df[
                dataset_df[
                    "metric"
                ] == metric
            ]

            plot_dataset_metric(
                df_metric=metric_df,
                dataset=dataset,
                metric=metric,
                output_dir=output_dir,
            )

# ============================================================
# Multi-dataset grid plots
#
# One figure per metric.
# Each subplot = one dataset.
# ============================================================

def plot_metric_dataset_grid(
    df,
    metric,
    output_dir,
    ncols=3,
):
    """
    Create one multi-panel figure for a metric.

    Example:

        accuracy_grid.pdf

        ┌────────────┬────────────┬────────────┐
        │ Dataset A  │ Dataset B  │ Dataset C  │
        │ curves     │ curves     │ curves     │
        ├────────────┼────────────┼────────────┤
        │ Dataset D  │ Dataset E  │ Dataset F  │
        │ curves     │ curves     │ curves     │
        └────────────┴────────────┴────────────┘

    Model colors are globally fixed.
    """

    metric_df = df[
        df["metric"] == metric
    ].copy()

    if metric_df.empty:
        return

    datasets = sorted(
        metric_df[
            "dataset"
        ].unique()
    )

    n_datasets = len(
        datasets
    )

    if n_datasets == 0:
        return

    ncols = min(
        ncols,
        n_datasets,
    )

    nrows = int(
        math.ceil(
            n_datasets / ncols
        )
    )

    # --------------------------------------------------------
    # Figure size
    #
    # ~4.3 inches per column
    # ~3.4 inches per row
    # --------------------------------------------------------

    fig_width = (
        4.3 * ncols
    )

    fig_height = (
        3.4 * nrows
        + 1.0
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            fig_width,
            fig_height,
        ),
        squeeze=False,
    )

    axes_flat = axes.ravel()

    # --------------------------------------------------------
    # Which models exist anywhere for this metric?
    #
    # This gives us one shared legend.
    # --------------------------------------------------------

    global_models = sorted(
        metric_df[
            "model"
        ].unique(),
        key=model_sort_key,
    )

    # ========================================================
    # Plot each dataset
    # ========================================================

    for panel_idx, dataset in enumerate(
        datasets
    ):

        ax = axes_flat[
            panel_idx
        ]

        dataset_df = metric_df[
            metric_df[
                "dataset"
            ] == dataset
        ].copy()

        dataset_models = sorted(
            dataset_df[
                "model"
            ].unique(),
            key=model_sort_key,
        )

        # ----------------------------------------------------
        # One model = one line
        # ----------------------------------------------------

        for model in dataset_models:

            model_df = (
                dataset_df[
                    dataset_df[
                        "model"
                    ] == model
                ]
                .sort_values(
                    "n_train"
                )
                .copy()
            )

            if model_df.empty:
                continue

            ax.plot(
                model_df[
                    "n_train"
                ],
                model_df[
                    "value"
                ],
                color=get_model_color(
                    model
                ),
                marker=get_model_marker(
                    model
                ),
                linewidth=2.0,
                markersize=4.7,
                label=pretty_model_name(
                    model
                ),
            )

            # ------------------------------------------------
            # Mark true full-data point, if present.
            # ------------------------------------------------

            full_rows = model_df[
                model_df[
                    "is_full"
                ]
            ]

            for _, row in full_rows.iterrows():

                ax.annotate(
                    "Full",
                    xy=(
                        row[
                            "n_train"
                        ],
                        row[
                            "value"
                        ],
                    ),
                    xytext=(
                        4,
                        4,
                    ),
                    textcoords=(
                        "offset points"
                    ),
                    fontsize=6.5,
                    alpha=0.75,
                )

        # ----------------------------------------------------
        # Axis formatting
        # ----------------------------------------------------

        ax.set_xscale(
            "log"
        )

        ax.grid(
            True,
            alpha=0.18,
            which="both",
            linewidth=0.7,
        )

        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )

        ax.set_title(
            dataset,
            fontsize=10,
            fontweight="medium",
            pad=7,
        )

        # ----------------------------------------------------
        # Actual training sizes as ticks
        # ----------------------------------------------------

        set_training_size_ticks(
            ax,
            dataset_df[
                "n_train"
            ],
        )

        ax.tick_params(
            axis="both",
            labelsize=7.5,
        )

        # ----------------------------------------------------
        # Only bottom row gets x-axis label
        # ----------------------------------------------------

        row_idx = (
            panel_idx
            // ncols
        )

        col_idx = (
            panel_idx
            % ncols
        )

        if row_idx == (
            nrows - 1
        ):

            ax.set_xlabel(
                "Training examples",
                fontsize=8.5,
            )

        # If final row is incomplete, the actual last
        # occupied row still needs x labels.
        elif (
            panel_idx
            + ncols
            >= n_datasets
        ):

            ax.set_xlabel(
                "Training examples",
                fontsize=8.5,
            )

        else:

            ax.set_xlabel("")

        # ----------------------------------------------------
        # Only left column gets metric ylabel
        # ----------------------------------------------------

        if col_idx == 0:

            ax.set_ylabel(
                pretty_metric_name(
                    metric
                ),
                fontsize=8.5,
            )

        else:

            ax.set_ylabel("")

    # ========================================================
    # Hide unused subplot cells
    # ========================================================

    for idx in range(
        n_datasets,
        len(axes_flat),
    ):

        axes_flat[
            idx
        ].axis(
            "off"
        )

    # ========================================================
    # Figure-level title
    # ========================================================

    direction = (
        "lower is better"
        if metric_is_lower_better(
            metric
        )
        else "higher is better"
    )

    fig.suptitle(
        (
            f"{pretty_metric_name(metric)} "
            f"vs. Training Set Size"
        ),
        fontsize=15,
        fontweight="semibold",
        y=0.995,
    )


    # ========================================================
    # One global legend
    #
    # Use proxy artists so models remain globally consistent.
    # ========================================================

    from matplotlib.lines import Line2D

    legend_handles = []

    for model in global_models:

        handle = Line2D(
            [0],
            [0],
            color=get_model_color(
                model
            ),
            marker=get_model_marker(
                model
            ),
            linewidth=2.2,
            markersize=5.5,
            label=pretty_model_name(
                model
            ),
        )

        legend_handles.append(
            handle
        )

    if legend_handles:

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(
                0.5,
                0.005,
            ),
            ncol=min(
                4,
                len(
                    legend_handles
                ),
            ),
            frameon=False,
            fontsize=9,
            columnspacing=1.5,
            handlelength=2.2,
        )

    # Leave space for:
    # title at top
    # legend at bottom
    fig.tight_layout(
        rect=[
            0.02,
            0.07,
            0.98,
            0.94,
        ]
    )

    safe_metric = (
        metric
        .replace(
            "/",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    save_figure(
        fig,
        output_dir
        / "metric_grids"
        / safe_metric,
        dpi=300,
    )
def make_metric_grid_plots(
    df,
    output_dir,
    ncols=3,
):
    """
    Produce one multi-panel figure per metric.
    """

    metrics = sorted(
        df[
            "metric"
        ].unique()
    )

    print()
    print(
        "Creating metric grid plots..."
    )

    for metric in metrics:

        n_datasets = (
            df.loc[
                df[
                    "metric"
                ] == metric,
                "dataset",
            ]
            .nunique()
        )

        print(
            f"  {metric}: "
            f"{n_datasets} datasets"
        )

        plot_metric_dataset_grid(
            df=df,
            metric=metric,
            output_dir=output_dir,
            ncols=ncols,
        )
# ============================================================
# Normalization for cross-dataset averages
# ============================================================

def add_dataset_normalized_scores(df):
    """
    Normalize each dataset + metric independently to [0, 1].

    Higher normalized score is ALWAYS better.

    For higher-is-better:
        (x - min) / (max - min)

    For lower-is-better:
        (max - x) / (max - min)

    This makes things like RMSE across datasets with wildly
    different target scales comparable.
    """

    output_parts = []

    group_cols = [
        "dataset",
        "metric",
    ]

    for (
        dataset,
        metric
    ), group in df.groupby(
        group_cols,
        sort=False,
    ):

        group = group.copy()

        values = group[
            "value"
        ].astype(float)

        vmin = values.min()
        vmax = values.max()

        if (
            not np.isfinite(vmin)
            or not np.isfinite(vmax)
        ):
            group[
                "normalized_score"
            ] = np.nan

        elif np.isclose(
            vmax,
            vmin,
        ):

            # Every model/size is identical.
            group[
                "normalized_score"
            ] = 1.0

        elif metric_is_lower_better(
            metric
        ):

            group[
                "normalized_score"
            ] = (
                vmax - values
            ) / (
                vmax - vmin
            )

        else:

            group[
                "normalized_score"
            ] = (
                values - vmin
            ) / (
                vmax - vmin
            )

        output_parts.append(
            group
        )

    if not output_parts:
        return df.assign(
            normalized_score=np.nan
        )

    return pd.concat(
        output_parts,
        ignore_index=True,
    )


# ============================================================
# Fair average computation
# ============================================================

def select_common_datasets_per_size(
    metric_df,
):
    """
    At each training size, retain only datasets that have
    results for ALL models that appear at that training size.

    This avoids:

        TabPFN average = datasets A/B/C
        XGB average    = datasets A/B/C/D/E/F

    at the same N.
    """

    kept_parts = []

    for n_train, size_df in metric_df.groupby(
        "n_train"
    ):

        models = sorted(
            size_df["model"].unique()
        )

        if len(models) <= 1:
            continue

        dataset_model_counts = (
            size_df
            .groupby("dataset")[
                "model"
            ]
            .nunique()
        )

        common_datasets = (
            dataset_model_counts[
                dataset_model_counts
                == len(models)
            ]
            .index
        )

        if len(common_datasets) == 0:
            continue

        kept = size_df[
            size_df[
                "dataset"
            ].isin(
                common_datasets
            )
        ]

        kept_parts.append(
            kept
        )

    if not kept_parts:
        return metric_df.iloc[
            0:0
        ].copy()

    return pd.concat(
        kept_parts,
        ignore_index=True,
    )


def calculate_average_table(
    df,
    value_column,
    metric,
):
    """
    Average over datasets for each model / training size.
    """

    metric_df = df[
        df["metric"] == metric
    ].copy()

    # "Full" differs in N between datasets, so it does not
    # represent one shared x-coordinate in an average curve.
    #
    # Exclude full runs from average curves.
    metric_df = metric_df[
        ~metric_df["is_full"]
    ]

    if metric_df.empty:
        return pd.DataFrame()

    if USE_COMMON_DATASETS_PER_SIZE:
        metric_df = (
            select_common_datasets_per_size(
                metric_df
            )
        )

    if metric_df.empty:
        return pd.DataFrame()

    rows = []

    for (
        model,
        n_train
    ), group in metric_df.groupby(
        [
            "model",
            "n_train",
        ]
    ):

        values = (
            group[
                value_column
            ]
            .dropna()
            .astype(float)
        )

        n_datasets = len(values)

        if (
            n_datasets
            < MIN_DATASETS_FOR_AVERAGE
        ):
            continue

        mean = values.mean()

        if n_datasets > 1:
            std = values.std(
                ddof=1
            )
            sem = std / math.sqrt(
                n_datasets
            )
        else:
            std = 0.0
            sem = 0.0

        rows.append({
            "metric": metric,
            "model": model,
            "n_train": int(
                n_train
            ),
            "mean": float(mean),
            "std": float(std),
            "sem": float(sem),
            "n_datasets": int(
                n_datasets
            ),
            "average_type": (
                value_column
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# Average plots
# ============================================================

def plot_average_metric(
    average_df,
    metric,
    output_dir,
    normalized,
):
    if average_df.empty:
        return

    fig, ax = plt.subplots(
        figsize=(7.5, 4.8)
    )

    models = sorted(
        average_df["model"].unique(),
        key=model_sort_key,
    )

    for model in models:

        model_df = (
            average_df[
                average_df[
                    "model"
                ] == model
            ]
            .sort_values(
                "n_train"
            )
        )

        if model_df.empty:
            continue

        x = model_df[
            "n_train"
        ].to_numpy()

        y = model_df[
            "mean"
        ].to_numpy()

        ax.plot(
            x,
            y,
            marker=get_model_marker(model),
            color=get_model_color(model),
            linewidth=2.2,
            markersize=5.5,
            label=pretty_model_name(model),
        )

        if DRAW_SEM:

            sem = model_df[
                "sem"
            ].to_numpy()

            ax.fill_between(
                x,
                y - sem,
                y + sem,
                color=get_model_color(model),
                alpha=0.12,
            )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "Number of training examples"
    )

    set_training_size_ticks(
        ax,
        average_df[
            "n_train"
        ],
    )

    if normalized:

        ax.set_ylabel(
            "Mean normalized performance"
        )

        ax.set_ylim(
            -0.03,
            1.03,
        )

        title = (
            f"Average normalized "
            f"{pretty_metric_name(metric)}"
        )

        subtitle = (
            "Higher normalized score "
            "is always better"
        )

    else:

        ax.set_ylabel(
            f"Mean "
            f"{pretty_metric_name(metric)}"
        )

        direction = (
            "lower is better"
            if metric_is_lower_better(
                metric
            )
            else "higher is better"
        )

        title = (
            f"Average "
            f"{pretty_metric_name(metric)}"
        )

        subtitle = direction

    ax.set_title(
        f"{title}\n{subtitle}"
    )

    ax.grid(
        True,
        alpha=0.20,
        which="both",
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.legend(
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()

    safe_metric = (
        metric
        .replace("/", "_")
        .replace(" ", "_")
    )

    subdir = (
        "normalized"
        if normalized
        else "raw"
    )

    save_figure(
        fig,
        output_dir
        / "average"
        / subdir
        / safe_metric,
    )


def make_average_plots(
    df,
    output_dir,
):
    normalized_df = (
        add_dataset_normalized_scores(
            df
        )
    )

    metrics = sorted(
        df[
            "metric"
        ].unique()
    )

    all_average_tables = []

    print()
    print(
        "Creating average plots..."
    )

    for metric in metrics:

        # ----------------------------------------------------
        # Normalized average:
        # make for EVERY metric
        # ----------------------------------------------------

        avg_norm = (
            calculate_average_table(
                normalized_df,
                value_column=(
                    "normalized_score"
                ),
                metric=metric,
            )
        )

        if not avg_norm.empty:

            avg_norm[
                "plot_type"
            ] = "normalized"

            all_average_tables.append(
                avg_norm
            )

            plot_average_metric(
                average_df=avg_norm,
                metric=metric,
                output_dir=output_dir,
                normalized=True,
            )

        # ----------------------------------------------------
        # Raw average:
        # produce for scale-comparable metrics.
        # ----------------------------------------------------

        if (
            metric.lower()
            in RAW_AVERAGE_METRICS
        ):

            avg_raw = (
                calculate_average_table(
                    df,
                    value_column="value",
                    metric=metric,
                )
            )

            if not avg_raw.empty:

                avg_raw[
                    "plot_type"
                ] = "raw"

                all_average_tables.append(
                    avg_raw
                )

                plot_average_metric(
                    average_df=avg_raw,
                    metric=metric,
                    output_dir=output_dir,
                    normalized=False,
                )

    if all_average_tables:

        average_table = pd.concat(
            all_average_tables,
            ignore_index=True,
        )

    else:

        average_table = pd.DataFrame()

    return (
        normalized_df,
        average_table,
    )


# ============================================================
# Coverage table
# ============================================================

def make_coverage_table(df):
    """
    Useful sanity check.

    Shows how many datasets each model has at every
    metric / training size.
    """

    coverage = (
        df[
            ~df["is_full"]
        ]
        .groupby(
            [
                "metric",
                "model",
                "n_train",
            ]
        )[
            "dataset"
        ]
        .nunique()
        .reset_index(
            name="n_datasets"
        )
        .sort_values(
            [
                "metric",
                "n_train",
                "model",
            ]
        )
    )

    return coverage


# ============================================================
# Summary
# ============================================================

def print_summary(df):
    print()
    print("=" * 80)
    print("DISCOVERED RESULTS")
    print("=" * 80)

    print()
    print("Datasets:")
    for dataset in sorted(
        df["dataset"].unique()
    ):
        print(
            f"  - {dataset}"
        )

    print()
    print("Models:")
    for model in sorted(
        df["model"].unique()
    ):
        print(
            f"  - {model}"
        )

    print()
    print("Metrics:")
    for metric in sorted(
        df["metric"].unique()
    ):
        print(
            f"  - {metric}"
        )

    print()
    print("Training sizes:")

    sizes = sorted(
        df.loc[
            ~df["is_full"],
            "n_train",
        ].unique()
    )

    print(
        "  "
        + ", ".join(
            format_n(x)
            for x in sizes
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            "./results"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "./figures/scaling"
        ),
    )

    args = parser.parse_args()

    results_dir = (
        args.results_dir
    )

    output_dir = (
        args.output_dir
    )

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory does "
            f"not exist: "
            f"{results_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_all_results(
        results_dir
    )

    if df.empty:
        raise RuntimeError(
            "No valid successful result "
            "JSON files were found."
        )

    print_summary(
        df
    )

    # --------------------------------------------------------
    # Save raw long table
    # --------------------------------------------------------

    df.sort_values(
        [
            "dataset",
            "metric",
            "model",
            "n_train",
        ]
    ).to_csv(
        output_dir
        / "scaling_results_long.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Dataset-specific curves
    # --------------------------------------------------------

    make_all_dataset_plots(
        df,
        output_dir,
    )
    
    # --------------------------------------------------------
    # Multi-dataset metric grids
    # --------------------------------------------------------

    make_metric_grid_plots(
        df,
        output_dir,
        ncols=3,
    )

    # --------------------------------------------------------
    # Average curves
    # --------------------------------------------------------

    normalized_df, average_df = (
        make_average_plots(
            df,
            output_dir,
        )
    )

    normalized_df.to_csv(
        output_dir
        / "scaling_results_normalized.csv",
        index=False,
    )

    if not average_df.empty:

        average_df.to_csv(
            output_dir
            / "average_scaling_curves.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Coverage table
    # --------------------------------------------------------

    coverage = (
        make_coverage_table(
            df
        )
    )

    coverage.to_csv(
        output_dir
        / "coverage_by_metric_model_size.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(
        f"Figures saved to: "
        f"{output_dir}"
    )

    print()
    print(
        "Important output files:"
    )

    print(
        f"  {output_dir}/"
        f"scaling_results_long.csv"
    )

    print(
        f"  {output_dir}/"
        f"scaling_results_normalized.csv"
    )

    print(
        f"  {output_dir}/"
        f"average_scaling_curves.csv"
    )

    print(
        f"  {output_dir}/"
        f"coverage_by_metric_model_size.csv"
    )
    coverage.to_csv(
        output_dir
        / "coverage_by_metric_model_size.csv",
        index=False,
    )
    # --------------------------------------------------------
    # Final regression/classification summary tables
    # --------------------------------------------------------

    summary_tables = (
        save_summary_tables(
            df,
            output_dir,
        )
    )


if __name__ == "__main__":
    main()