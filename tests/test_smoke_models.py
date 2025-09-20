"""
Smoke tests for model training and evaluation functionality.

This module contains comprehensive smoke tests that verify the main
training and evaluation pipelines can run without errors using mocked
components and synthetic data.
"""

import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import types
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_processed_csv(path: Path, n=20):
    """
    Create a minimal processed CSV file with required housing data columns.

    Args:
        path (Path): Path where to save the CSV file
        n (int): Number of rows to generate

    Note: Categorical columns are encoded as numeric codes to avoid fit errors.
    """
    # ensure parent dir exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # create a minimal processed CSV with required columns
    rows = []
    for i in range(n):
        rows.append(
            {
                "buy_price": 200000 + i * 1000,  # Increasing price
                "sq_mt_built": 50 + i,  # Increasing built area
                "sq_mt_useful": 45 + i,  # Increasing useful area
                "n_rooms": 2 + (i % 3),  # Rooms between 2-4
                "n_bathrooms": 1 + (i % 2),  # Bathrooms between 1-2
                "has_parking": int(bool(i % 2)),  # Alternating parking
                "neighborhood_id": int(i % 4),  # 4 neighborhoods
                "house_type_id": int(i % 2),  # 2 house types
                "year_built": 1990 + (i % 30),  # Years between 1990-2019
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def make_splits_files(tmp_path: Path, n=20):
    """
    Create train/test split and cross-validation fold files.

    Args:
        tmp_path (Path): Temporary directory to save split files
        n (int): Total number of data points

    Returns:
        Path: Directory containing the split files
    """
    # create test_indices.json and cv_folds.json with sensible splits
    indices = list(range(n))
    test_indices = indices[int(n * 0.8) :]  # Last 20% for test
    train_indices = indices[: int(n * 0.8)]  # First 80% for training

    # simple 2-fold split for train part
    half = max(1, len(train_indices) // 2)
    folds = {
        "0": {"train_idx": train_indices[:half], "val_idx": train_indices[half:]},
        "1": {"train_idx": train_indices[half:], "val_idx": train_indices[:half]},
    }
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(exist_ok=True)
    (splits_dir / "test_indices.json").write_text(json.dumps(test_indices))
    (splits_dir / "cv_folds.json").write_text(json.dumps(folds))
    return splits_dir


class DummyEstimator:
    """Dummy estimator for testing that implements basic sklearn interface."""

    def fit(self, X, y=None):
        """Mock fit method that returns self."""
        return self

    def predict(self, X):
        """Mock predict method that returns zeros."""
        # return zeros vector of length n
        try:
            return np.zeros(len(X))
        except Exception:
            return np.zeros(1)


class DummyGrid:
    """Dummy GridSearchCV for testing."""

    def __init__(self, *args, **kwargs):
        self.best_params_ = {}
        self.best_estimator_ = DummyEstimator()

    def fit(self, X, y):
        """Mock fit method that returns self."""
        return self


def test_train_main_smoke(monkeypatch, tmp_path):
    """
    Run src.models.train.main() with mocked heavy pieces.

    Args:
        monkeypatch: Pytest monkeypatch fixture
        tmp_path: Pytest temporary directory fixture

    Verifies:
    - Training pipeline runs without errors
    - Metrics file is created with expected structure
    """
    # prepare data files
    processed_path = tmp_path / "data" / "processed" / "processed.csv"
    make_processed_csv(processed_path, n=30)

    splits_dir = make_splits_files(tmp_path, n=30)

    # preprocessor file - simple pipeline (scaler only is fine because features numeric)
    preproc = Pipeline([("scaler", StandardScaler())])
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    preproc_path = models_dir / "preprocessor.joblib"
    joblib.dump(preproc, preproc_path)

    # import module and monkeypatch module-level paths in src.models.train
    import src.models.train as train_mod

    monkeypatch.setattr(train_mod, "DATA_PATH", processed_path)
    monkeypatch.setattr(train_mod, "SPLITS_DIR", splits_dir)
    monkeypatch.setattr(train_mod, "PREPROCESSOR_PATH", preproc_path)
    # change outputs to tmp
    monkeypatch.setattr(
        train_mod, "OUTPUT_MODEL_PATH", models_dir / "final_model.joblib"
    )
    monkeypatch.setattr(train_mod, "METRICS_PATH", models_dir / "metrics.json")

    # monkeypatch heavy objects with simple dummies
    monkeypatch.setattr(train_mod, "LGBMRegressor", lambda **kwargs: DummyEstimator())
    monkeypatch.setattr(
        train_mod,
        "GridSearchCV",
        lambda estimator, param_grid, cv, scoring, n_jobs, verbose: DummyGrid(),
    )

    # run main (should not raise)
    train_mod.main()

    # assert metrics file written
    assert (models_dir / "metrics.json").exists()
    metrics = json.loads((models_dir / "metrics.json").read_text())
    assert "test_metrics" in metrics


def test_evaluate_all_smoke(monkeypatch, tmp_path):
    """
    Run src.models.evaluate_all.main() on a directory with simple models.

    Args:
        monkeypatch: Pytest monkeypatch fixture
        tmp_path: Pytest temporary directory fixture

    Verifies:
    - Evaluation pipeline runs without errors
    - Summary metrics file is created
    """
    # make processed csv and splits
    processed_path = tmp_path / "data" / "processed" / "processed.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    make_processed_csv(processed_path, n=10)

    splits_dir = make_splits_files(tmp_path, n=10)
    monkeypatch.setenv("PYTEST_RUNNING", "1")

    # prepare models dir with a simple pickled estimator (DummyEstimator)
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    simple_model = DummyEstimator()
    joblib.dump(simple_model, models_dir / "model1.joblib")

    # import module and monkeypatch paths
    import src.models.evaluate_all as eval_mod

    monkeypatch.setattr(eval_mod, "DATA_PATH", processed_path)
    monkeypatch.setattr(eval_mod, "SPLITS_PATH", splits_dir / "test_indices.json")
    monkeypatch.setattr(eval_mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(eval_mod, "REPORTS_DIR", tmp_path / "reports")

    # run evaluation
    eval_mod.main()

    # check reports summary exists
    assert (tmp_path / "reports" / "summary_metrics.csv").exists()


def test_run_experiments_smoke(monkeypatch, tmp_path):
    """
    Run src.models.run_experiments.main() but mock subprocess.run to avoid calling actual scripts.

    Args:
        monkeypatch: Pytest monkeypatch fixture
        tmp_path: Pytest temporary directory fixture

    Verifies:
    - Experiment runner processes multiple config files
    - Subprocess calls are made for each config
    """
    import src.models.run_experiments as re_mod

    called = []

    def fake_run(cmd, check):
        called.append(cmd)
        return 0

    monkeypatch.setattr(re_mod, "subprocess", types.SimpleNamespace(run=fake_run))
    # create dummy config files
    cfg1 = tmp_path / "cfg1.yaml"
    cfg1.write_text("dummy: true")
    cfg2 = tmp_path / "cfg2.yaml"
    cfg2.write_text("dummy: true")
    # call main with configs list
    re_mod.main([str(cfg1), str(cfg2)])
    assert len(called) == 2


def test_train_mlflow_smoke(monkeypatch, tmp_path):
    """
    Test that src.models.train_mlflow.run_training runs with mocked mlflow and gridsearch.

    Args:
        monkeypatch: Pytest monkeypatch fixture
        tmp_path: Pytest temporary directory fixture

    Verifies:
    - MLflow training pipeline runs without errors
    - Metrics files are created with expected naming pattern
    """
    import src.models.train_mlflow as tmod

    # create config YAML file with paths to tmp data
    cfg = tmp_path / "exp.yaml"
    proc = tmp_path / "data" / "processed" / "processed.csv"
    proc.parent.mkdir(parents=True, exist_ok=True)
    make_processed_csv(proc, n=12)
    splits_dir = make_splits_files(tmp_path, n=12)

    cfg_content = {
        "data_path": str(proc),
        "splits_dir": str(splits_dir),
        "preprocessor_path": str(tmp_path / "models" / "preprocessor.joblib"),
        "out_dir": str(tmp_path / "models"),
        "target_col": "buy_price",
        "use_log_target": False,
        "lgb_param_grid": {
            "model__n_estimators": [1],
            "model__learning_rate": [0.1],
            "model__max_depth": [2],
        },
        "inner_cv": 2,
        "n_jobs": 1,
    }
    import yaml

    cfg.write_text(yaml.safe_dump(cfg_content))

    # create a simple preprocessor file
    tmp_models = tmp_path / "models"
    tmp_models.mkdir(exist_ok=True)
    joblib.dump(
        Pipeline([("scaler", StandardScaler())]), tmp_models / "preprocessor.joblib"
    )

    # monkeypatch mlflow to a dummy object used in the module
    class DummyRun:
        def __init__(self):
            self.info = types.SimpleNamespace(run_id="rrr")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyMlflow:
        def set_experiment(self, *a, **kw):
            pass

        def start_run(self, *a, **kw):
            return DummyRun()

        class sklearn:
            @staticmethod
            def log_model(m, artifact_path):
                pass

        def log_param(self, *a, **kw):
            pass

        def log_metric(self, *a, **kw):
            pass

        def log_artifact(self, *a, **kw):
            pass

        def sklearn_log_model(self, *a, **kw):
            pass

    monkeypatch.setattr(tmod, "GridSearchCV", lambda *args, **kwargs: DummyGrid())
    monkeypatch.setattr(tmod, "LGBMRegressor", lambda **kw: DummyEstimator())
    monkeypatch.setattr("src.models.train_mlflow.mlflow", DummyMlflow())

    # run training (should not raise)
    tmod.run_training(cfg)

    # check that metrics file written locally
    out_metrics = list((tmp_models).glob("metrics_*.json"))
    assert len(out_metrics) >= 1
