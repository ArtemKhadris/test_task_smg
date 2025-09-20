"""
Experiment runner for sequential execution of multiple ML training configurations.

This module allows running multiple machine learning experiments sequentially
using different configuration files. Each experiment is executed as a separate
subprocess calling the train_mlflow module.
"""

import argparse
import subprocess
from pathlib import Path
import sys


def main(config_paths):
    """
    Execute multiple experiment configurations sequentially.

    Args:
        config_paths (list): List of paths to experiment configuration files

    Raises:
        FileNotFoundError: If any configuration file does not exist
        subprocess.CalledProcessError: If any experiment execution fails
    """
    # Use current Python interpreter for subprocess execution
    python = sys.executable
    module = "src.models.train_mlflow"

    for cfg in config_paths:
        cfgp = Path(cfg)
        # Validate configuration file existence
        if not cfgp.exists():
            raise FileNotFoundError(f"Config not found: {cfg}")

        print(f"=== Launching experiment: {cfg} ===")

        # Execute training module with current configuration
        subprocess.run([python, "-m", module, "--config", str(cfgp)], check=True)

        print(f"=== Finished: {cfg} ===\n")


if __name__ == "__main__":
    # Configure command-line argument parser
    parser = argparse.ArgumentParser(
        description="Run multiple ML experiments sequentially from configuration files"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="List of experiment configuration file paths",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Execute main function with provided configuration paths
    main(args.configs)
