"""Guard for the grading rule "library model as a core model = zero marks".

The decision tree, the SVM and the metrics must be our own NumPy code, so none
of those modules may import scikit-learn (or scipy). sklearn is allowed only in
baselines.py / benchmark.py (comparison) and in tests.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
CORE = ("decision_tree.py", "svm.py", "evaluate.py", "error_analysis.py")
BONUS_SKLEARN_REFERENCE = ("bonus_linear.py", "bonus_ensembles.py", "bonus_boosting.py")
FORBIDDEN = ("sklearn", "scipy", "statsmodels", "xgboost", "lightgbm")


def _imported_roots(path: Path) -> set:
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
    allowed = {"baselines.py", "benchmark.py", "data_prep.py", "bonus_clustering.py", *BONUS_SKLEARN_REFERENCE}
    users = {p.name for p in SRC.glob("*.py") if "sklearn" in _imported_roots(p)}
    assert users <= allowed, f"unexpected sklearn imports in: {sorted(users - allowed)}"


@pytest.mark.parametrize("filename", ("bonus_linear.py", "bonus_ensembles.py", "bonus_boosting.py"))
def test_bonus_sklearn_only_inside_reference_function(filename):
    """Our own bonus classes must not use sklearn: it may appear only in `_sklearn_reference`."""
    tree = ast.parse((SRC / filename).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {a.name.split(".")[0] for a in node.names} | ({node.module.split(".")[0]} if getattr(node, "module", None) else set())
            assert "sklearn" not in names, f"module-level sklearn import in {filename}"
        if isinstance(node, ast.ClassDef):
            for inner in ast.walk(node):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    mods = {a.name for a in getattr(inner, "names", [])} | {getattr(inner, "module", None) or ""}
                    assert not any("sklearn" in m for m in mods), f"sklearn used inside a class in {filename}"
