#!/usr/bin/env python
"""Validate that all required packages and environment variables are present."""

import importlib
import sys

REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("torch", "PyTorch"),
    ("sklearn", "Scikit-Learn"),
    ("mlflow", "MLflow"),
    ("dvc", "DVC"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("pydantic_settings", "Pydantic Settings"),
    ("dotenv", "python-dotenv"),
]

REQUIRED_ENV_VARS: list[str] = [
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "SEED",
]


def check_packages() -> list[str]:
    """Check that all required packages are importable.

    Returns:
        List of missing package names.
    """
    missing: list[str] = []
    for module, display_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} — NOT FOUND")
            missing.append(display_name)
    return missing


def check_env_vars() -> list[str]:
    """Check that required environment variables are set.

    Returns:
        List of missing variable names.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv()
    missing: list[str] = []
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var} = {value}")
        else:
            print(f"  ❌ {var} — NOT SET")
            missing.append(var)
    return missing


def main() -> None:
    """Run all environment validations and exit with appropriate code."""
    print("\n🔍 Checking required packages...")
    missing_packages = check_packages()

    print("\n🔍 Checking environment variables...")
    missing_env_vars = check_env_vars()

    if missing_packages or missing_env_vars:
        print(f"\n❌ Validation FAILED. Missing packages: {missing_packages}")
        print(f"   Missing env vars: {missing_env_vars}")
        print("   Run: uv sync --extra dev")
        sys.exit(1)
    else:
        print("\n✅ Environment is valid. Ready to run the pipeline.")
        sys.exit(0)


if __name__ == "__main__":
    main()
