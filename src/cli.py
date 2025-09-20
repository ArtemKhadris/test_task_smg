"""
Command Line Interface (CLI) for Madrid Housing Price Prediction project.

This module provides a unified CLI for data preparation, model training,
evaluation, and serving using Click framework.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List
import click
import yaml
from typing import Union

# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Union[str, Path, None] = None, env_prefix: str = "CFG__") -> dict:
    """
    Load YAML configuration file with environment variable overrides.

    Args:
        path (Optional[str]): Path to YAML config file. If None, uses default config.
        env_prefix (str): Prefix for environment variables that override config values.

    Returns:
        dict: Configuration dictionary with environment overrides applied.

    Example:
        Environment variable CFG__data__raw_path overrides config['data']['raw_path']
    """
    if path is None:
        path = PROJECT_ROOT / "config" / "default.yaml"
    else:
        path = Path(path)

    # Load base configuration from YAML file
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Apply overrides from environment variables
    for k, v in os.environ.items():
        if not k.startswith(env_prefix):
            continue

        # Remove prefix and split by '__' to get nested key path
        key_path = k[len(env_prefix) :].split("__")

        # Navigate to the appropriate nested dictionary
        cur = cfg
        for sub in key_path[:-1]:
            if sub not in cur or not isinstance(cur[sub], dict):
                cur[sub] = {}
            cur = cur[sub]

        # Try to parse JSON value to allow booleans/lists/numbers
        try:
            parsed = json.loads(v)
        except Exception:
            parsed = v

        # Set the final value
        cur[key_path[-1]] = parsed

    return cfg


def run_module(module: str, args: List[str]):
    """
    Execute a Python module as a subprocess using the same interpreter.

    Args:
        module (str): Python module name to execute
        args (List[str]): List of command-line arguments for the module

    Raises:
        subprocess.CalledProcessError: If the subprocess returns non-zero exit code
    """
    cmd = [sys.executable, "-m", module] + args
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


@click.group()
def cli():
    """Project CLI for prepare-data / train / evaluate / serve"""
    pass


@cli.command("prepare-data")
@click.option("--config", default="config/default.yaml", help="Path to config YAML")
def prepare_data_cmd(config):
    """Prepare and preprocess the raw data for training."""
    _ = load_config(config)
    # Call data preparation module
    try:
        run_module("src.data.prepare_data", ["--config", config])
    except subprocess.CalledProcessError as e:
        print("Error running prepare_data:", e)
        sys.exit(1)


@cli.command("train")
@click.option("--config", default="config/default.yaml", help="Path to config YAML")
def train_cmd(config):
    """Train machine learning models with optional MLflow tracking."""
    cfg = load_config(config)
    # Determine training approach based on configuration
    training_cfg = cfg.get("training", {})
    use_mlflow = training_cfg.get("use_mlflow", True)

    try:
        if use_mlflow:
            # Use MLflow-enabled training with experiment tracking
            run_module("src.models.train_mlflow", ["--config", config])
        else:
            # Use basic training without MLflow
            run_module("src.models.train", ["--config", config])
    except subprocess.CalledProcessError as e:
        print("Error running training:", e)
        sys.exit(1)


@cli.command("evaluate")
@click.option("--config", default="config/default.yaml", help="Path to config YAML")
@click.option(
    "--all",
    "eval_all",
    is_flag=True,
    default=False,
    help="Evaluate all models in models/ (default: true)",
)
def evaluate_cmd(config, eval_all):
    """Evaluate trained models and generate performance reports."""
    _ = load_config(config)
    # Use comprehensive evaluation for all models
    try:
        # Default: evaluate all models in the models directory
        run_module("src.models.evaluate_all", [])
    except subprocess.CalledProcessError:
        # Fallback to single model evaluation if evaluate_all fails
        try:
            run_module("src.models.evaluate", [])
        except subprocess.CalledProcessError as e:
            print("Error running evaluation:", e)
            sys.exit(1)


@cli.command("serve")
@click.option("--config", default="config/default.yaml", help="Path to config YAML")
def serve_cmd(config):
    """Start FastAPI server for model inference."""
    cfg = load_config(config)
    serve_cfg = cfg.get("serve") or cfg.get("serve", cfg.get("serve", {}))

    # Get server configuration with defaults
    host = serve_cfg.get("host") or cfg.get("serve", {}).get("host", "0.0.0.0")
    port = int(serve_cfg.get("port") or cfg.get("serve", {}).get("port", 8000))
    reload = bool(serve_cfg.get("reload", True))

    # Start Uvicorn server for FastAPI application
    _ = "uvicorn"
    app_entry = "src.app.main:app"
    args = [f"{app_entry}", "--host", str(host), "--port", str(port)]

    if reload:
        args += ["--reload"]

    try:
        run_module("uvicorn", args)
    except subprocess.CalledProcessError as e:
        print("Error running server:", e)
        sys.exit(1)


@cli.command("run-experiments")
@click.option(
    "--configs", multiple=True, required=True, help="List of experiment config YAMLs"
)
def run_experiments_cmd(configs):
    """Run multiple experiments sequentially from configuration files."""
    # Execute each experiment configuration in sequence
    for cfg in configs:
        try:
            run_module("src.models.train_mlflow", ["--config", cfg])
        except subprocess.CalledProcessError as e:
            print(f"Experiment failed for {cfg}:", e)
            sys.exit(1)


if __name__ == "__main__":
    """Main entry point for the CLI application."""
    cli()
