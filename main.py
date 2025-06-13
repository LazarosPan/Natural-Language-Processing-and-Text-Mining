#!/usr/bin/env python3
"""
main.py

Automated pipeline to execute all project notebooks in sequence.
Notebooks live under the `notebooks/` subdirectory:

  notebooks/0_split.ipynb
  notebooks/1_eda.ipynb
  notebooks/2_preprocessing.ipynb
  notebooks/3_feature_engineering.ipynb
  notebooks/4_models.ipynb
  notebooks/5_benchmarks.ipynb
  notebooks/6_summary.ipynb

Each notebook is executed in place (outputs are written back). If any
notebook fails, the script stops and reports the error.
"""

import subprocess, sys
from pathlib import Path

NOTEBOOKS = [
    "0_split.ipynb",
    "1_eda.ipynb",
    "2_preprocessing.ipynb",
    "3_feature_engineering.ipynb",
    "4_models.ipynb",
    "5_benchmarks*.ipynb",
    "6_summary.ipynb"
]

def check_nbconvert_installed():
    try:
        import nbconvert  # noqa
    except ImportError:
        print("\nError: install nbconvert with `pip install nbconvert`")
        sys.exit(1)

def run_notebook(nb_path: Path):
    print(f"\n=== Executing {nb_path.relative_to(Path.cwd())} ===")
    subprocess.run(
        [sys.executable, "-m", "nbconvert",
         "--to", "notebook",
         "--execute",
         "--inplace",
         str(nb_path)],
        check=True
    )
    print(f"-> {nb_path.name} OK")

def main():
    check_nbconvert_installed()
    root = Path(__file__).parent.resolve()
    notebook_dir = root / "notebooks"

    for pattern in NOTEBOOKS:
        # expand wildcards
        candidates = list(notebook_dir.glob(pattern))
        if not candidates:
            print(f"Error: no notebook matches {pattern}")
            sys.exit(1)
        nb_path = candidates[0]
        try:
            run_notebook(nb_path)
        except subprocess.CalledProcessError as e:
            print(f"\nAborting: {nb_path.name} failed.")
            sys.exit(1)

    print("\nAll notebooks ran successfully.")

if __name__ == "__main__":
    main()