from .base import BaseTFM


class TabDPTAdapter(BaseTFM):

    def __init__(
        self,
        task,
        device="cuda",
        seed=42,

        #should these be fine-tuned or what?
        context_size=2048,
        n_ensembles=8,
        batch_size=None,
        temperature=1.0,
        context_reduction="subsample",
        feature_reduction="pca",
        normalizer="standard",
        missing_indicators=False,
        use_flash=True,
        compile=True,
        verbose=False,

        **kwargs,
    ):
        super().__init__(
            task=task,
            device=device,
            seed=seed,
            **kwargs,
        )

        from tabdpt import (
            TabDPTClassifier,
            TabDPTRegressor,
        )

        self.context_size = context_size
        self.n_ensembles = n_ensembles
        self.batch_size = batch_size
        self.temperature = temperature

        common_kwargs = {
            "device": device,
            "context_reduction": context_reduction,
            "feature_reduction": feature_reduction,
            "normalizer": normalizer,
            "missing_indicators": missing_indicators,
            "use_flash": use_flash,
            "compile": compile,
            "verbose": verbose,
        }

        common_kwargs.update(kwargs)

        if task == "classification":
            self._model = TabDPTClassifier(
                **common_kwargs,
            )

        else:
            self._model = TabDPTRegressor(
                **common_kwargs,
            )


    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):

        if self.task == "classification":
            return self._model.predict(
                X,
                n_ensembles=self.n_ensembles,
                temperature=self.temperature,
                context_size=self.context_size,
                batch_size=self.batch_size,
                seed=self.seed,
            )

        return self._model.predict(
            X,
            n_ensembles=self.n_ensembles,
            context_size=self.context_size,
            batch_size=self.batch_size,
            seed=self.seed,
            output_type="mean",
        )

    def predict_proba(self, X):

        if self.task != "classification":
            return None

        if self.n_ensembles == 1:
            return self._model.predict_proba(
                X,
                temperature=self.temperature,
                context_size=self.context_size,
                batch_size=self.batch_size,
                seed=self.seed,
            )

        return self._model.ensemble_predict_proba(
            X,
            n_ensembles=self.n_ensembles,
            temperature=self.temperature,
            context_size=self.context_size,
            batch_size=self.batch_size,
            seed=self.seed,
        )