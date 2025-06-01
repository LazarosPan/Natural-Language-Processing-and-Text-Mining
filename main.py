#!/usr/bin/env python3
"""
main.py

Automated pipeline to execute all project notebooks in sequence.
Notebooks live under the `notebooks/` subdirectory:

  notebooks/00_split.ipynb
  notebooks/01_eda.ipynb
  notebooks/02_preprocessing.ipynb
  notebooks/03_feature_engineering.ipynb
  notebooks/04_models.ipynb
  notebooks/05_benchmarks.ipynb

Each notebook is executed in place (outputs are written back). If any
notebook fails, the script stops and reports the error.
"""

import subprocess
import sys
from pathlib import Path

# List of notebook filenames (order matters)
NOTEBOOKS = [
    "00_split.ipynb",
    "01_eda.ipynb",
    "02_preprocessing.ipynb",
    "03_feature_engineering.ipynb",
    "04_models.ipynb",
    "05_benchmarks.ipynb",
]

def check_nbconvert_installed() -> None:
    """
    Verify that `nbconvert` is importable. If not, print instructions and exit.
    """
    try:
        import nbconvert  # noqa: F401
    except ImportError:
        print(
            "\nError: the Python package 'nbconvert' is not installed.\n"
            "Please install it with:\n\n"
            "    pip install nbconvert\n\n"
            "Then re-run this script.\n"
        )
        sys.exit(1)

def run_notebook(nb_path: Path) -> None:
    """
    Execute a single notebook in place using Python's nbconvert module.
    Raises CalledProcessError if execution fails.
    """
    print(f"\n=== Executing {nb_path.relative_to(Path.cwd())} ===")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                str(nb_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f" {nb_path.name} executed successfully.")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="ignore")
        print(f"\n Error executing {nb_path.name}:")
        print(stderr)
        raise

def main():
    # First, ensure nbconvert is installed
    check_nbconvert_installed()

    # Determine project root (directory containing this script)
    root = Path(__file__).parent.resolve()

    # Directory where notebooks are stored
    notebook_dir = root / "notebooks"

    for nb_name in NOTEBOOKS:
        nb_path = notebook_dir / nb_name
        if not nb_path.exists():
            print(f"Error: Notebook not found: {nb_path}")
            sys.exit(1)

        try:
            run_notebook(nb_path)
        except subprocess.CalledProcessError:
            print(f"\nAborting pipeline due to error in {nb_name}.")
            sys.exit(1)

    print("\n=== All notebooks executed successfully ===")

if __name__ == "__main__":
    main()