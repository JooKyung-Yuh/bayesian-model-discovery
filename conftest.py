"""Explicit source requirements for the native benchmark test cases."""

import importlib
import importlib.machinery
import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

BENCHMARK_SOURCES = {
    "boxing_gym": (("BMD_BOXINGGYM_SRC",), "06c790aa8e23c1f8f1858f0955ec4ffc9e4c61ba"),
    "autoscilab": (
        ("BMD_AUTOSCILAB",),
        "9a37255ff5432a7155b6cde8985e9654f2f8f272",
    ),
    "neuronbench": (("NEURONBENCH_SRC",), "c354622458c460b419cab821d482c879f0578377"),
}


def _package_exists_in_head(source, package):
    prefix = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--show-prefix"],
        capture_output=True, text=True, check=False,
    )
    if prefix.returncode != 0:
        return False
    relative_prefix = prefix.stdout.rstrip("\n")
    tracked = subprocess.run(
        ["git", "-C", str(source), "cat-file", "-e",
         f"HEAD:{relative_prefix}{package}/__init__.py"],
        capture_output=True, text=True, check=False,
    )
    return tracked.returncode == 0


def _require_external_source(package, variables, expected_revision):
    requirement = (
        f"{package} requires {' or '.join(variables)} and importable source "
        f"at expected commit {expected_revision}"
    )
    source = None
    for variable in variables:
        value = os.environ.get(variable)
        if not value:
            continue
        candidate = Path(value).expanduser().resolve()
        if candidate.is_dir() and (
            (candidate / package / "__init__.py").is_file()
            or _package_exists_in_head(candidate, package)
        ):
            source = candidate
            break
    if source is None:
        pytest.skip(requirement + "; source directory is unavailable")

    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if revision.returncode != 0 or revision.stdout.strip() != expected_revision:
        pytest.fail(requirement + "; supplied source revision could not be verified", pytrace=False)
    package_root = source / package
    clean = subprocess.run(
        ["git", "-C", str(source), "diff", "--quiet", "HEAD", "--", str(package_root)],
        capture_output=True, text=True, check=False,
    )
    if clean.returncode != 0:
        pytest.fail(requirement + "; supplied source differs from its pinned revision", pytrace=False)

    untracked = subprocess.run(
        ["git", "-C", str(source), "ls-files", "--others", "-z", "--", str(package_root)],
        capture_output=True, text=True, check=False,
    )
    runtime_suffixes = (".py", ".pyc", ".pyo", *importlib.machinery.EXTENSION_SUFFIXES)
    if untracked.returncode != 0 or any(
        "__pycache__" not in Path(name).parts and name.endswith(runtime_suffixes)
        for name in untracked.stdout.split("\0") if name
    ):
        pytest.fail(requirement + "; supplied source contains untracked runtime code", pytrace=False)

    spec = importlib.util.find_spec(package)
    if spec is None:
        pytest.skip(requirement + "; source is not importable through PYTHONPATH")
    if spec.origin is None or not Path(spec.origin).resolve().is_relative_to(package_root):
        pytest.fail(requirement + "; import resolves outside the supplied source", pytrace=False)
    # An available package's import error is a real failure, not an optional-source skip.
    importlib.import_module(package)
    return source


@pytest.fixture
def require_boxinggym_source():
    """Return the verified BoxingGym source directory, or skip/fail the test."""
    return _require_external_source("boxing_gym", *BENCHMARK_SOURCES["boxing_gym"])


@pytest.fixture
def require_autoscilab_source():
    """Return the verified AutoSciLab source directory, or skip/fail the test."""
    return _require_external_source("autoscilab", *BENCHMARK_SOURCES["autoscilab"])


@pytest.fixture
def require_neuronbench_source():
    """Return the verified NeuronBench source directory, or skip/fail the test."""
    return _require_external_source("neuronbench", *BENCHMARK_SOURCES["neuronbench"])


@pytest.fixture(autouse=True)
def _reset_process_grammar_state():
    """Every test starts and ends on the process default grammar.

    A runner selects the expression grammar once per run and registers the
    task's input signs with it; a test that drives a runner in process does the
    same to the interpreter every other test then shares. The first version of
    this guard put back whatever it found BEFORE the test, which is not enough:
    a session-scoped fixture (test_frozen_kl_rescore's CORE cell) runs before any
    per-test fixture, so the leaked signs were what the guard "restored", and the
    grammar tests that ran after that module failed in CI while every file passed
    alone. No fixture of any scope deliberately holds a non-default grammar
    across tests, so the correct baseline is the default itself, on both sides.
    """
    from bayes_model_discovery.experimental.count_models import (
        DEFAULT_GRAMMAR_VERSION,
        restore_grammar_state,
    )

    restore_grammar_state((DEFAULT_GRAMMAR_VERSION, {}))
    yield
    restore_grammar_state((DEFAULT_GRAMMAR_VERSION, {}))
