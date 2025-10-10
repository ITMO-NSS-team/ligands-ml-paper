import json
import os
import warnings
from typing import Optional, Any, Callable

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split

from utils import filter_var

warnings.filterwarnings("ignore", category=UserWarning)


class FeatureSelector:
    def __init__(
            self,
            score_func: Optional[Callable] = None,
            k_percent: Optional[float] = None
    ):
        self.score_func = score_func
        self.k_percent = k_percent
        self.selector: Optional[SelectKBest] = None

    @property
    def name(self) -> str:
        return getattr(self.score_func, "__name__", "none") if self.score_func else "none"

    @property
    def key(self) -> str:
        return "all" if self.k_percent is None else str(int(round(self.k_percent * 100)))

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.score_func is None or self.k_percent is None:
            return X
        k_actual = max(1, int(X.shape[1] * self.k_percent))
        self.selector = SelectKBest(score_func=self.score_func, k=k_actual)
        return self.selector.fit_transform(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selector is None:
            return X
        return self.selector.transform(X)


class ModelFactory:
    def __init__(
            self,
            model_cls: Callable,
            model_kwargs: Optional[dict] = None
    ):
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs or {}

    def create(self) -> Any:
        kwargs = self.model_kwargs.copy()
        return self.model_cls(**kwargs)


class Evaluator:
    def __init__(
            self,
            metrics: dict[str, Callable] | None = None
    ):
        if metrics is None:
            metrics = {
                "rmse": lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),
                "r2": r2_score,
                "mae": mean_absolute_error,
            }
        self.metrics = metrics

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {name: func(y_true, y_pred) for name, func in self.metrics.items()}


class CrossValidator:
    def __init__(
            self,
            n_splits: Optional[int],
            random_state: int,
            model_factory: ModelFactory,
            evaluator: Optional[Evaluator] = None,
            save_path: Optional[str] = None,
            var_threshold: float = 0.00,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.model_factory = model_factory
        self.evaluator = evaluator or Evaluator()
        self.save_path = save_path
        self.var_threshold = var_threshold

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        selector: FeatureSelector,
    ) -> Optional[dict[int, dict[str, Any]]]:
        if self.n_splits is None or self.n_splits < 2:
            return None  # CV disabled

        results: dict[int, dict[str, Any]] = {}
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"\n--- Fold {fold + 1}/{self.n_splits} ---")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_filtered, var_selector = filter_var(X_train, threshold=self.var_threshold)
            X_train = X_filtered
            X_test = var_selector.transform(X_test)

            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)

            model = self.model_factory.create()
            pipeline = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results[fold] = self.evaluator.evaluate(y_test, y_pred)
            results[fold]["n_features"] = X_train.shape[1]
            results[fold]["pipeline_path"] = None

            if self.save_path is not None:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                pipeline_path = os.path.join(
                    self.save_path, f"{selector.key}_{selector.name}_fold_{fold}.json"
                )
                pipeline.save(
                    pipeline_path,
                    create_subdir=True,
                    is_datetime_in_path=False
                )
                results[fold]["pipeline_path"] = pipeline_path
        return results


class ModelTrainer:
    def __init__(
            self,
            random_state: int,
            model_factory: ModelFactory,
            evaluator: Optional[Evaluator] = None,
            save_path: Optional[str] = None,
            var_threshold: float = 0.00,
    ):
        self.random_state = random_state
        self.model_factory = model_factory
        self.evaluator = evaluator or Evaluator()
        self.save_path = save_path
        self.var_threshold = var_threshold

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        selector: FeatureSelector,
    ) -> dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        X_filtered, var_selector = filter_var(X_train, threshold=self.var_threshold)
        X_train = X_filtered
        X_test = var_selector.transform(X_test)

        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        model = self.model_factory.create()
        pipeline = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        scores = self.evaluator.evaluate(y_test, y_pred)
        scores["n_features"] = X_train.shape[1]
        scores["pipeline_path"] = None

        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            pipeline_path = os.path.join(
                self.save_path, f"{selector.key}_{selector.name}_train_test.json"
            )
            pipeline.save(
                pipeline_path,
                create_subdir=True,
                is_datetime_in_path=False
            )
            scores["pipeline_path"] = pipeline_path

        return scores


class Experiment:
    def __init__(
        self,
        datasets: dict[str, pd.DataFrame],
        target_col: str,
        results_dir: str,
        model_factory: ModelFactory,
        seed: int = 42,
        n_splits: Optional[int] = None,
        var_threshold: float = 0.01,
        fs_func: Optional[Callable] = None,
        k_percents: Optional[list[float]] = None,
        evaluator: Optional[Evaluator] = None,
        save_pipeline: bool = True,
    ):
        self.datasets = datasets
        self.target_col = target_col
        self.results_dir = results_dir
        self.model_factory = model_factory
        self.seed = seed
        self.n_splits = n_splits
        self.var_threshold = var_threshold
        self.fs_func = fs_func
        self.k_percents = [None] if k_percents is None else k_percents
        self.evaluator = evaluator or Evaluator()
        self.save_pipeline = save_pipeline

        os.makedirs(results_dir, exist_ok=True)

    def run(self) -> dict[str, Any]:
        all_results = {}
        for dataset_name, df in self.datasets.items():
            print(f"\n=== Dataset: {dataset_name} ===")

            X, y = df.drop(columns=[self.target_col]).values, df[self.target_col].values
            dataset_dir = os.path.join(self.results_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            dataset_results = {}
            for k_percent in self.k_percents:
                selector = FeatureSelector(self.fs_func, k_percent)
                result_file = os.path.join(dataset_dir, f"{selector.key}_{selector.name}.json")

                if os.path.exists(result_file):
                    print(f"Loading cached results for {dataset_name}, {selector.key}, {selector.name}")
                    with open(result_file, "r") as f:
                        dataset_results[selector.key] = json.load(f)
                    continue

                save_path = dataset_dir if self.save_pipeline else None
                dataset_results[selector.key] = self._run_pipeline(X, y, selector, save_path)

                with open(result_file, "w") as f:
                    json.dump(dataset_results[selector.key], f, indent=2)
                    print(f"Results saved to {result_file}")

            all_results[dataset_name] = dataset_results

        return all_results

    def _run_pipeline(
            self,
            X: np.ndarray,
            y: np.ndarray,
            selector: FeatureSelector,
            save_path: Optional[str] = None,
    ) -> dict[str, Any]:
        if np.all(np.nanstd(X, axis=0) == 0):
            return {"error": "All features constant."}

        cv = CrossValidator(
            self.n_splits,
            self.seed,
            self.model_factory,
            self.evaluator,
            save_path=save_path,
            var_threshold=self.var_threshold,
        )
        cv_scores = cv.run(X, y, selector) if self.n_splits else None

        trainer = ModelTrainer(
            self.seed,
            self.model_factory,
            self.evaluator,
            save_path=save_path,
            var_threshold=self.var_threshold,
        )
        test_scores = trainer.run(X, y, selector)

        return {"cv_scores": cv_scores, "test_scores": test_scores, 'fs_func': selector.name}
