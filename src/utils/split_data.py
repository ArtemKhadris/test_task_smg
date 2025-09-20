"""
Data splitting utility for Madrid Housing Price Prediction.

This module provides functions for creating train/test splits and
cross-validation folds for machine learning experiments.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
import json


def create_splits(
    data_path="data/processed/processed.csv",
    target_col="buy_price",
    test_size=0.2,
    n_splits=5,
    random_state=42,
    output_dir="data/splits",
):
    """
    Create train/test split and cross-validation folds for model evaluation.

    Args:
        data_path (str): Path to the processed dataset CSV file
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for test set (0.0 to 1.0)
        n_splits (int): Number of cross-validation folds
        random_state (int): Random seed for reproducibility
        output_dir (str): Directory to save the split indices

    Returns:
        None: Saves split indices to JSON files in the output directory

    Output Files:
        - test_indices.json: List of indices for the test set
        - cv_folds.json: Dictionary with train/validation indices for each fold
    """
    # Load processed dataset
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Save test indices to JSON file
    test_indices = X_test.index.tolist()
    with open(output_path / "test_indices.json", "w") as f:
        json.dump(test_indices, f)

    # Step 2: Create K-Fold cross-validation splits from training data
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = {}
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        folds[fold] = {
            "train_idx": X_train.iloc[train_idx].index.tolist(),
            "val_idx": X_train.iloc[val_idx].index.tolist(),
        }

    # Save cross-validation folds to JSON file
    with open(output_path / "cv_folds.json", "w") as f:
        json.dump(folds, f)

    print(f"Saved test indices and {n_splits}-fold CV splits to {output_path}")


if __name__ == "__main__":
    """
    Main execution block for creating data splits.

    When run as a script, this creates default train/test splits and
    cross-validation folds using the processed dataset.
    """
    create_splits()
