import os
import random

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_datasets(path: str) -> dict[str, pd.DataFrame]:
    datasets = {}
    for file in os.listdir(path):
        if not file.endswith(".csv"):
            continue
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        dataset_name = os.path.splitext(file)[0]  # remove .csv extension
        datasets[dataset_name] = df
    return datasets


def load_v3_datasets(path: str) -> dict[str, dict[str, pd.DataFrame]]:
    result = {}

    for root, dirs, files in os.walk(path):
        rel = os.path.relpath(root, path)

        if rel == ".":
            continue

        section_name = rel.replace(os.sep, "_")

        csv_files = [f for f in files if f.endswith(".csv")]
        if not csv_files:
            continue

        section_dict = {}
        for file in csv_files:
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            csv_name = os.path.splitext(file)[0]
            section_dict[csv_name] = df

        result[section_name] = section_dict

    return result


def filter_var(
        x: np.ndarray,
        threshold: float = 0.01
) -> tuple[np.ndarray, VarianceThreshold]:
    selector = VarianceThreshold(threshold=threshold)
    x_filtered = selector.fit_transform(x)

    n_removed = x.shape[1] - x_filtered.shape[1]
    print(f"Removed {n_removed} low-variance features "
          f"(kept {x_filtered.shape[1]} of {x.shape[1]})")

    return x_filtered, selector
