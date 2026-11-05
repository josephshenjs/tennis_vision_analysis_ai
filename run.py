"""Portable launcher: no installed Python package, parent folder or PYTHONPATH needed."""

from pathlib import Path
import importlib.util
import sys

APP = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "rally_analysis", APP / "__init__.py", submodule_search_locations=[str(APP)]
)
package = importlib.util.module_from_spec(spec)
sys.modules["rally_analysis"] = package
sys.modules["rally_analysis_ai"] = package
spec.loader.exec_module(package)
from rally_analysis.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
