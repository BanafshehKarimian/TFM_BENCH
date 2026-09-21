from .base import BaseTFM


class TabICLAdapter(BaseTFM):

    def __init__(
        self,
        task,
        version="v2",
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

        from tabicl import TabICLClassifier, TabICLRegressor

        checkpoints = {
            "v1": "tabicl-classifier-v1-20250208.ckpt",
            "v1.1": "tabicl-classifier-v1.1-20250506.ckpt",
            "v2": "tabicl-classifier-v2-20260212.ckpt",
        }

        if task == "classification":
            model_kwargs = dict(
                device=device,
                random_state=seed,
                **kwargs,
            )

            model_kwargs["checkpoint_version"] = checkpoints[version]

            self._model = TabICLClassifier(**model_kwargs)

        else:
            if version != "v2":
                raise ValueError(
                    "Regression: v2"
                )

            self._model = TabICLRegressor(
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
