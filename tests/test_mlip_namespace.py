"""The dftgnn.mlip package must stay inert: every submodule name resolves to the module, never to a package-level
object that shadows it (a function named like a submodule broke dftgnn.mlip.geometry in the M1b merge)."""
import ast
import importlib
import inspect
from pathlib import Path

import pytest

import dftgnn.mlip

PKG_DIR = Path(dftgnn.mlip.__file__).parent
STEMS = sorted(p.stem for p in PKG_DIR.glob("*.py") if p.stem != "__init__")


@pytest.mark.parametrize("stem", STEMS)
def test_submodule_name_resolves_to_module(stem):
    # the import form used across the code base: ``from dftgnn.mlip import <stem>`` returns an existing package
    # attribute without importing the submodule, so a shadowing object would be returned here
    pkg = __import__("dftgnn.mlip", fromlist=[stem])
    assert inspect.ismodule(getattr(pkg, stem))
    assert getattr(pkg, stem) is importlib.import_module(f"dftgnn.mlip.{stem}")


def test_package_init_defines_no_callables():
    tree = ast.parse((PKG_DIR / "__init__.py").read_text())
    defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))]
    assert not defs
    own = [name for name, obj in vars(dftgnn.mlip).items()
           if callable(obj) and not inspect.ismodule(obj) and getattr(obj, "__module__", None) == "dftgnn.mlip"]
    assert not own
