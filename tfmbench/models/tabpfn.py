from .base import BaseTFM


class TabPFNAdapter(BaseTFM):

    def __init__(
        self,
        task,
        device="cuda",
        seed=42,
        **kwargs,
    ):
        super().__init__(
            task=task,
            device=device,
            seed=seed,
            **kwargs,
        )

        from tabpfn import TabPFNClassifier, TabPFNRegressor
        if kwargs.TABPFN_TOKEN:
            import os
            os.environ["TABPFN_TOKEN"] = kwargs.TABPFN_TOKEN
            del kwargs.TABPFN_TOKEN

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


    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task == "classification":
            return self._model.predict_proba(X)
        return None
