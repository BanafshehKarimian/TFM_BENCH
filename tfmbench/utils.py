from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from tfmbench.datasets import BaseTabularDataset as TabularDataset
from tfmbench.evaluate import evaluate
from tfmbench.benchmark import benchmark_models
from tfmbench.datasets.talent import load_talent_dataset


def get_tab_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )


    data = TabularDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task="classification",
        name="breast_cancer",
    )
    return data