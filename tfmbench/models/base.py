from abc import ABC, abstractmethod
from typing import Any


class BaseTFM(ABC):

    def __init__(
        self,
        task: str,
        device: str = "cuda",
        seed: int = 42,
        **kwargs,
    ):
        self.task = task
        self.device = device
        self.seed = seed
        self.kwargs = kwargs

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
