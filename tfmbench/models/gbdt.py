from .base import BaseTFM


class XGBoostAdapter(BaseTFM):

    def __init__(
        self,
        task,
        device,
        seed,
        **kwargs,
    ):
        super().__init__(
            task=task,
            device=device,
            seed=seed,
        )

        from xgboost import XGBClassifier, XGBRegressor #the version should be > 3.1.0

        device_str = str(device)

        
        xgb_device = "gpu" if device.type=="cuda" else "cpu" 

        common_kwargs = {
            "random_state": seed,
            "tree_method": "hist",
            "device": xgb_device,
            "n_jobs": -1,

            # IMPORTANT
            "enable_categorical": True,
        }

        common_kwargs.update(kwargs)

        if task == "classification":
            self._model = XGBClassifier(
                **common_kwargs
            )

        elif task == "regression":
            self._model = XGBRegressor(
                **common_kwargs
            )

        else:
            raise ValueError(
                f"Unsupported task: {task}"
            )


    def _prepare_X(self, X):
        """
        Convert TALENT dataframe dtypes into types
        XGBoost can understand.

        num_* -> float32
        cat_* -> pandas category
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()

        for col in X.columns:

            if str(col).startswith("num_"):

                X[col] = pd.to_numeric(
                    X[col],
                    errors="coerce",
                ).astype("float32")

            elif str(col).startswith("cat_"):

                X[col] = X[col].astype(
                    "string"
                ).astype(
                    "category"
                )

            else:
                # fallback:
                # try numeric first
                try:
                    X[col] = pd.to_numeric(
                        X[col]
                    ).astype("float32")

                except Exception:
                    X[col] = (
                        X[col]
                        .astype("string")
                        .astype("category")
                    )

        return X


    def fit(self, X, y):

        X = self._prepare_X(X)

        self._model.fit(
            X,
            y,
        )

        return self


    def predict(self, X):

        X = self._prepare_X(X)

        return self._model.predict(X)




    def predict_proba(self, X):
        if self.task == "classification":
            X = self._prepare_X(X)
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

            common_kwargs["devices"] = device.type
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
