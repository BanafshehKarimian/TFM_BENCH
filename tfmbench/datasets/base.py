from dataclasses import dataclass
from typing import Literal, Any


TaskType = Literal["classification", "regression"]


@dataclass
class BaseTabularDataset:
    X_train: Any
    y_train: Any
    X_test: Any
    y_test: Any

    task: TaskType
    name: str | None = None

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]