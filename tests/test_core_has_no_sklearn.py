from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
CORE = ("decision_tree.py", "svm.py", "evaluate.py")
FORBIDDEN = ("sklearn", "scipy", "statsmodels", "xgboost", "lightgbm")


def _imported_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("filename", CORE)
def test_core_module_has_no_library_model_imports(filename):
    assert not (_imported_roots(SRC / filename) & set(FORBIDDEN)), filename


def test_sklearn_is_confined_to_comparison_code():
    allowed = {"baselines.py", "data_prep.py", "bonus_clustering.py"}
    users = {p.name for p in SRC.glob("*.py") if "sklearn" in _imported_roots(p)}
    assert users <= allowed, f"unexpected sklearn imports in: {sorted(users - allowed)}"
