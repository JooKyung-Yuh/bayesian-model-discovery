"""Native-source availability must never hide a supplied source's failures."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "_native_source_guards", Path(__file__).resolve().parents[2] / "conftest.py"
)
guards = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(guards)


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def native_source(tmp_path, monkeypatch):
    """Helper or fixture: native source."""
    package = "_bmd_test_native_source"
    root = tmp_path / "native"
    package_root = root / package
    package_root.mkdir(parents=True)
    module = package_root / "__init__.py"
    module.write_text("VALUE = 42\n")
    _git(root, "init", "--quiet")
    _git(root, "add", package)
    _git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "--quiet", "-m", "Add native fixture")
    revision = _git(root, "rev-parse", "HEAD")
    monkeypatch.setenv("BMD_TEST_NATIVE_SRC", str(root))
    monkeypatch.syspath_prepend(str(root))
    yield package, root, module, revision
    sys.modules.pop(package, None)


@pytest.mark.parametrize("package", guards.BENCHMARK_SOURCES)
def test_absent_source_names_the_configuration_and_revision(package, monkeypatch):
    """Check that absent source names the configuration and revision."""
    variables, revision = guards.BENCHMARK_SOURCES[package]
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(pytest.skip.Exception) as caught:
        guards._require_external_source(package, variables, revision)
    assert all(variable in str(caught.value) for variable in variables)
    assert revision in str(caught.value)


def test_pinned_source_imports_without_replacing_modules(native_source):
    """Check that pinned source imports without replacing modules."""
    package, root, _, revision = native_source
    original_path = list(sys.path)
    assert guards._require_external_source(package, ("BMD_TEST_NATIVE_SRC",), revision) == root
    imported = sys.modules[package]
    assert imported.VALUE == 42
    assert guards._require_external_source(package, ("BMD_TEST_NATIVE_SRC",), revision) == root
    assert sys.modules[package] is imported
    assert sys.path == original_path


def test_existing_source_without_import_path_is_an_explicit_skip(native_source, monkeypatch):
    """Check that existing source without import path is an explicit skip."""
    package, root, _, revision = native_source
    monkeypatch.setattr(sys, "path", [value for value in sys.path if value != str(root)])
    with pytest.raises(pytest.skip.Exception, match="not importable through PYTHONPATH"):
        guards._require_external_source(package, ("BMD_TEST_NATIVE_SRC",), revision)


def test_wrong_revision_fails_instead_of_skipping(native_source):
    """Check that wrong revision fails instead of skipping."""
    package, _, _, _ = native_source
    with pytest.raises(pytest.fail.Exception, match="revision could not be verified"):
        guards._require_external_source(package, ("BMD_TEST_NATIVE_SRC",), "0" * 40)


def test_modified_source_fails_instead_of_skipping(native_source):
    """Check that modified source fails instead of skipping."""
    package, _, module, revision = native_source
    module.write_text("VALUE = 99\n")
    with pytest.raises(pytest.fail.Exception, match="differs from its pinned revision"):
        guards._require_external_source(package, ("BMD_TEST_NATIVE_SRC",), revision)


def test_foreign_import_fails_instead_of_skipping(native_source, tmp_path, monkeypatch):
    """Check that foreign import fails instead of skipping."""
    package, _, _, revision = native_source
    foreign = tmp_path / "foreign"
    (foreign / package).mkdir(parents=True)
    (foreign / package / "__init__.py").write_text("VALUE = -1\n")
    monkeypatch.syspath_prepend(str(foreign))
    with pytest.raises(pytest.fail.Exception, match="outside the supplied source"):
        guards._require_external_source(package, ("BMD_TEST_NATIVE_SRC",), revision)


def test_broken_pinned_source_preserves_the_import_error(native_source):
    """Check that broken pinned source preserves the import error."""
    package, root, module, _ = native_source
    module.write_text("raise RuntimeError('native import is broken')\n")
    _git(root, "add", package)
    _git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "--quiet", "-m", "Break native import")
    revision = _git(root, "rev-parse", "HEAD")
    with pytest.raises(RuntimeError, match="native import is broken"):
        guards._require_external_source(package, ("BMD_TEST_NATIVE_SRC",), revision)


def test_autoscilab_never_falls_back_to_discoverphysics_source(native_source, monkeypatch):
    """Check that autoscilab never falls back to discoverphysics source."""
    package, root, _, revision = native_source
    variables, _ = guards.BENCHMARK_SOURCES["autoscilab"]
    assert variables == ("BMD_AUTOSCILAB",)
    monkeypatch.delenv("BMD_AUTOSCILAB", raising=False)
    monkeypatch.setenv("DP_SCIENCEAGENT_DIR", str(root))
    with pytest.raises(pytest.skip.Exception, match="source directory is unavailable"):
        guards._require_external_source(package, variables, revision)
    monkeypatch.setenv("BMD_AUTOSCILAB", str(root))
    assert guards._require_external_source(package, variables, revision) == root


@pytest.mark.parametrize("staged", [False, True])
def test_deleted_package_initializer_fails_instead_of_skipping(native_source, staged):
    """Check that deleted package initializer fails instead of skipping."""
    package, root, module, revision = native_source
    module.unlink()
    if staged:
        _git(root, "add", "-u")
    with pytest.raises(pytest.fail.Exception, match="differs from its pinned revision"):
        guards._require_external_source(package, ("BMD_TEST_NATIVE_SRC",), revision)


@pytest.mark.parametrize("ignored", [False, True])
def test_untracked_package_cannot_shadow_pinned_module(native_source, ignored):
    """Check that untracked package cannot shadow pinned module."""
    package, root, module, _ = native_source
    module.write_text("from .native import VALUE\n")
    (module.parent / "native.py").write_text("VALUE = 42\n")
    _git(root, "add", package)
    _git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "--quiet", "-m", "Add pinned native module")
    revision = _git(root, "rev-parse", "HEAD")
    shadow = module.parent / "native"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("VALUE = 99\n")
    if ignored:
        (root / ".git" / "info" / "exclude").write_text("native/\n")
    with pytest.raises(pytest.fail.Exception, match="untracked runtime code"):
        guards._require_external_source(package, ("BMD_TEST_NATIVE_SRC",), revision)
    assert package not in sys.modules
