import os
import json
import numpy as np
import warnings
from typing import Optional, Any

import pandas as pd
from fedot import Fedot
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split

from utils import filter_var

warnings.filterwarnings("ignore", category=UserWarning)


class FedotPipeline:
    def __init__(
        self,
        datasets,
        target_col,
        results_dir,
        n_splits,
        seed,
        var_threshold=0.01,
        fs_func=None,
        k_percents=None,
        fedot_kwargs=None,
    ):
        self.datasets = datasets
        self.target_col = target_col
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.n_splits = n_splits
        self.seed = seed
        self.fs_func = fs_func
        self.fs_name = getattr(fs_func, "__name__", str(fs_func)) if fs_func else "none"
        self.k_percents = [None] if k_percents is None else k_percents
        self.var_threshold = var_threshold
        self.fedot_kwargs = fedot_kwargs or {}

    def run(self):
        all_results = {}
        for dataset_name, df in self.datasets.items():
            X, y = df.drop(columns=[self.target_col]), df[self.target_col].values
            X = filter_var(X, threshold=self.var_threshold)
            print(f"\n=== Dataset: {dataset_name} | Samples: {X.shape[0]} | Features: {X.shape[1]} ===")

            results = {}
            dataset_dir = os.path.join(self.results_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            for p in self.k_percents:
                key = "all" if p is None else str(int(round(p * 100)))
                result_file = os.path.join(dataset_dir, f"{key}_{self.fs_name}.json")

                if os.path.exists(result_file):
                    print(f"Loading cached results for {dataset_name}, key={key}, fs={self.fs_name}")
                    with open(result_file, "r") as f:
                        results[key] = json.load(f)
                else:
                    run_result = self._safe_run_fedot(X, y, dataset_name, k_percent=p)
                    results[key] = {"fs_func": self.fs_name, "k_percent": p, "results": run_result}

                    with open(result_file, "w") as f:
                        json.dump(results[key], f, indent=2)
                        print(f"Results saved to {result_file}")

            all_results[dataset_name] = results

        return all_results

    def _safe_run_fedot(self, X, y, dataset_name, k_percent: Optional[float] = None):
        if self._invalid_features(X):
            print("⚠️ Skipping run: all features are constant.")
            return {"error": "All features constant."}

        return run_fedot(
            X,
            y,
            dataset_name,
            self.results_dir,
            n_splits=self.n_splits,
            k_percent=k_percent,
            score_func=self.fs_func,
            random_state=self.seed,
            fedot_kwargs=self.fedot_kwargs,
        )

    @staticmethod
    def _invalid_features(X):
        return np.all(np.nanstd(X, axis=0) == 0)


def init_fedot(
    problem: str = "regression",
    timeout: float = 10.0,
    n_jobs: int = -1,
    logging_level: int = 50,
    seed: int = 42,
    initial_assumption: Optional[Pipeline] = None,
    cv_folds: int = 10,
    metric: str = "rmse",
) -> Fedot:
    return Fedot(
        problem=problem,
        timeout=timeout,
        n_jobs=n_jobs,
        logging_level=logging_level,
        seed=seed,
        initial_assumption=initial_assumption,
        cv_folds=cv_folds,
        metric=metric,
    )


def cross_validate(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    save_path: str,
    n_splits: int = 5,
    k_percent: Optional[float] = None,
    score_func: Optional[Any] = None,
    random_state: int = 42,
    fedot_kwargs: Optional[dict[str, Any]] = None,
) -> dict[int, dict[str, Any]]:

    key = "all" if k_percent is None else str(int(round(k_percent * 100)))
    fs_name = getattr(score_func, "__name__", "none") if score_func else "none"
    dataset_dir = os.path.join(save_path, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    X_values = X.values if isinstance(X, pd.DataFrame) else X
    fedot_kwargs = fedot_kwargs or {}
    cv_results: dict[int, dict[str, Any]] = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_values)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        model = init_fedot(**fedot_kwargs, seed=random_state)

        X_train, X_test = X_values[train_idx], X_values[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if score_func is not None and k_percent is not None:
            k_actual = max(1, int(X_train.shape[1] * k_percent))
            selector = SelectKBest(score_func=score_func, k=k_actual)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            print(f"Selected {X_train.shape[1]} features using {fs_name} ({key}%)")

        pipeline = model.fit(X_train, y_train)
        pipeline_path = os.path.join(dataset_dir, f"{key}_{fs_name}_fold_{fold + 1}.json")
        pipeline.save(pipeline_path, create_subdir=True, is_datetime_in_path=False)

        y_pred = model.predict(X_test)
        cv_results[fold] = {
            "rmse": mean_squared_error(y_test, y_pred, squared=False),
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "n_features": X_train.shape[1],
            "pipeline_path": pipeline_path,
        }
        print(f"Fold {fold}: RMSE={cv_results[fold]['rmse']:.4f}, "
              f"R2={cv_results[fold]['r2']:.4f}, MAE={cv_results[fold]['mae']:.4f}")
        print(f"Pipeline saved to {pipeline_path}")

    return cv_results


def run_fedot(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    save_path: str,
    n_splits: int = 5,
    k_percent: Optional[float] = None,
    score_func: Optional[Any] = None,
    random_state: int = 42,
    fedot_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:

    key = "all" if k_percent is None else str(int(round(k_percent * 100)))
    fs_name = getattr(score_func, "__name__", "none") if score_func else "none"
    dataset_dir = os.path.join(save_path, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # cv_scores = cross_validate(
    #     X, y, dataset_name, save_path,
    #     n_splits=n_splits,
    #     k_percent=k_percent,
    #     score_func=score_func,
    #     random_state=random_state,
    #     fedot_kwargs=fedot_kwargs,
    # )
    # print("CV results:", cv_scores)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    if score_func is not None and k_percent is not None:
        k_actual = max(1, int(X_train.shape[1] * k_percent))
        selector = SelectKBest(score_func=score_func, k=k_actual)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        print(f"Selected {X_train.shape[1]} features using {fs_name} ({key}%)")

    model = init_fedot(**(fedot_kwargs or {}), seed=random_state)
    pipeline = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    pipeline_path = os.path.join(dataset_dir, f"{key}_{fs_name}_train_test_pipeline.json")
    pipeline.save(pipeline_path, create_subdir=True, is_datetime_in_path=False)

    print(f"Test RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, MAE: {test_mae:.4f}")
    print(f"Pipeline saved to {pipeline_path}")

    return {
        "cv_scores": {},
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "pipeline_path": pipeline_path,
    }
