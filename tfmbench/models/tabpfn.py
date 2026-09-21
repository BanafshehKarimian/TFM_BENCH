from .base import BaseTFM
import os

class TabPFNAdapter(BaseTFM):

    def __init__(
        self,
        task,
        device="cuda",
        seed=42,
        tabpfn_token=None,
        **kwargs,
    ):
        super().__init__(
            task=task,
            device=device,
            seed=seed,
            **kwargs,
        )

        from tabpfn import TabPFNClassifier, TabPFNRegressor

        if task == "classification":
            self._model = TabPFNClassifier(
                device=device,
                random_state=seed,
                **kwargs,
            )

        else:
            self._model = TabPFNRegressor(
                device=device,
                random_state=seed,
                **kwargs,
            )
        self.tabpfn_token = tabpfn_token


    def fit(self, X, y):
        os.environ["TABPFN_TOKEN"] = self.tabpfn_token
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task == "classification":
            return self._model.predict_proba(X)
        return None
