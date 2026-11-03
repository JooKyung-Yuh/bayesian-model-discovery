"""The NewtonBench adapter reads the pinned law files by path and reproduces the benchmark's vanilla branch.

Needs `BMD_NEWTONBENCH` (a checkout of HKUST-KnowComp/NewtonBench at the pinned commit, see
`scripts/provision.sh --with newtonbench`); skips otherwise.
"""

import math

import pytest

from bayes_model_discovery.adapters import newtonbench_observation as nb

ROW = {"id": "m2_magnetic_force__medium__v0", "benchmark": "newtonbench", "domain": "m2_magnetic_force",
       "difficulty": "medium", "law_version": "v0", "input_box": "newtonbench_evaluate_law"}


def test_every_pinned_law_file_has_its_digest(require_newtonbench_source):
    root = nb.vendor_root()
    for relative, digest in nb.NEWTONBENCH_FILES.items():
        assert nb._sha256(root / "modules" / relative) == digest, relative


def test_a_law_loads_by_path_and_is_finite_on_its_box(require_newtonbench_source):
    bound = nb.bind(ROW)
    designs = bound.sample_designs(8, 3)
    assert len(designs) == 8 and all(set(d) == set(bound.provenance["input_names"]) for d in designs)
    law = nb._Law(ROW["domain"], ROW["difficulty"], ROW["law_version"], bound.provenance["input_names"], 0.0)
    assert all(math.isfinite(law.value(d)) for d in designs)
    assert bound.provenance["package_imported"] is False
    assert bound.provenance["law_files_identical_to"].endswith(nb.NEWTONBENCH_COMMIT)


def test_noise_is_deterministic_per_design(require_newtonbench_source):
    bound = nb.bind(ROW)
    law = nb._Law(ROW["domain"], ROW["difficulty"], ROW["law_version"], bound.provenance["input_names"], 0.01)
    x = bound.sample_designs(1, 5)[0]
    truth = law.value(x)
    assert law.noise(x, truth) == law.noise(x, truth)


def test_a_wrong_checkout_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("BMD_NEWTONBENCH", str(tmp_path))
    with pytest.raises(ValueError, match="modules"):
        nb.vendor_root()
