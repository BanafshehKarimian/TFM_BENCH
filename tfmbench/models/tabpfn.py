from .base import BaseTFM


class TabPFNAdapter(BaseTFM):

    def __init__(
        self,
        task,
        version="v3",
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
        from tabpfn.constants import ModelVersion

        version_map = {
            "v2": ModelVersion.V2,
            "v2.5": ModelVersion.V2_5,
            "v2.6": ModelVersion.V2_6,
            "v3": ModelVersion.V3,
            "v3.5": ModelVersion.V3_5,
            "v3.5-fast": ModelVersion.V3_5_FAST,
        }

        if version not in version_map:
            raise ValueError(
                f"Unknown TabPFN version '{version}'. "
                f"Available: {list(version_map)}"
            )

        self.version = version
        model_version = version_map[version]

        overrides = {
            "device": device,
            "random_state": seed,
            **kwargs,
        }

        if task == "classification":
            self._model = (
                TabPFNClassifier.create_default_for_version(
                    model_version,
                    **overrides,
                )
            )

        else:
            self._model = (
                TabPFNRegressor.create_default_for_version(
                    model_version,
                    **overrides,
                )
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
