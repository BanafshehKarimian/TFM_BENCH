from .base import BaseTFM


class XGBoostAdapter(BaseTFM):
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

        from xgboost import XGBClassifier, XGBRegressor

        xgb_device = "GPU" if device.type=="cuda" else "CPU" 

        common_kwargs = {
            "random_state": seed,
            "tree_method": "hist",
            "device": xgb_device,
            "n_jobs": -1,
        }

        common_kwargs.update(kwargs)

        if task == "classification":
            self._model = XGBClassifier(**common_kwargs)

        else:
            self._model = XGBRegressor(**common_kwargs)


    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task == "classification":
            return self._model.predict_proba(X)
        return None


class CatBoostAdapter(BaseTFM):
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

        from catboost import CatBoostClassifier, CatBoostRegressor

        common_kwargs = {
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
        }

        if device.type=="cuda":
            common_kwargs["task_type"] = "GPU"

            # cuda:0 -> devices="0"
            if ":" in device:
                common_kwargs["devices"] = device.split(":")[1]
        else:
            common_kwargs["task_type"] = "CPU"

        common_kwargs.update(kwargs)

        if task == "classification":
            self._model = CatBoostClassifier(**common_kwargs)

        else:
            self._model = CatBoostRegressor(**common_kwargs)

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X).reshape(-1)

    def predict_proba(self, X):
        if self.task == "classification":
            return self._model.predict_proba(X)
        return None
