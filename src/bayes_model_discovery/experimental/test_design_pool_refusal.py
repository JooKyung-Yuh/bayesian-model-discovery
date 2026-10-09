"""Design-search pool refusal (acquisition-policy rule).

Pool expected information gain (EIG) and expected predictive information gain (EPIG) are forbidden at input dimension d >= 3 in every run; study policies are not
information estimators and pass. Every input here is synthetic: the unit cases pass literal
settings, policies and dimensions, and the runner cases drive a mock ChemBench run on the real
adapter (seven inputs, dev manifest) that stops at its first fit, so no sequential Monte Carlo (SMC) runs, no proposer
is called and no saved run output is read.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import design_search as search

UNSET = {}
EXPLICIT_POOL = {"candidate_source": "pool"}
ASCENT = {"design_search": "ascent", "candidate_source": "ascent_endpoints"}
POOL_SETTINGS = pytest.mark.parametrize("settings", [UNSET, EXPLICIT_POOL],
                                        ids=["unset", "explicit_pool"])
INFORMATION_POLICIES = pytest.mark.parametrize("policy", ["joint_eig", "epig"])
STUDY_POLICIES = pytest.mark.parametrize("policy", ["point_disagreement", "space_filling"])
REMEDY = {"joint_eig": "declare --design-search ascent for this cell",
          "epig": "ascent above two inputs is joint_eig only, so declare a joint_eig cell "
                  "with --design-search ascent"}
MANIFEST = Path(__file__).resolve().parents[3] / "manifests" / "chembench_dev_manifest.json"


def refusal(policy, dimension):
    """The refusal message for `policy` at `dimension`, escaped for `pytest.raises(match=)`."""
    return re.escape(
        f"design-search pool is refused for {policy} at input dimension {dimension} > 2 "
        "(the acquisition policy forbids pool EIG/EPIG at d >= 3 for every run); "
        f"{REMEDY[policy]}")


def refuse(settings, policy, applies, dimension):
    """Call the pool-dimension refusal check with the given settings, policy and input dimension."""
    return search.refuse_pool_above_two_dimensions(
        settings, design_policy=policy, design_policy_applies=applies, input_dimension=dimension)


# ---------------------------------------------------------------------------
# (a) The function on its own.
# ---------------------------------------------------------------------------

def test_the_two_pool_settings_are_what_declared_settings_returns():
    """Check that the two pool settings are what declared settings returns."""
    args = SimpleNamespace(cell="001", design_policy="joint_eig")
    assert search.declared_settings(args, "acquisition") == UNSET
    args.design_search = "pool"
    assert search.declared_settings(args, "acquisition") == EXPLICIT_POOL


@POOL_SETTINGS
@INFORMATION_POLICIES
def test_an_information_estimator_on_the_pool_at_seven_inputs_is_refused(settings, policy):
    """Check that an information estimator on the pool at seven inputs is refused."""
    with pytest.raises(ValueError, match=refusal(policy, 7)) as refused:
        refuse(settings, policy, True, 7)
    assert str(refused.value).isascii()


@POOL_SETTINGS
@STUDY_POLICIES
def test_a_study_policy_on_the_pool_at_seven_inputs_passes(settings, policy):
    """Check that a study policy on the pool at seven inputs passes."""
    assert refuse(settings, policy, True, 7) is None


@INFORMATION_POLICIES
def test_the_ascent_declaration_passes_at_seven_inputs(policy):
    """Check that the ascent declaration passes at seven inputs."""
    assert refuse(ASCENT, policy, True, 7) is None


@POOL_SETTINGS
def test_a_cell_whose_third_digit_selects_no_policy_passes_at_seven_inputs(settings):
    # cell[2] == "0": the resolved policy is the default joint_eig, check_policy returns False
    # and the run never reads the pool.
    """Check that a cell whose third digit selects no policy passes at seven inputs."""
    assert refuse(settings, "joint_eig", False, 7) is None


@POOL_SETTINGS
@INFORMATION_POLICIES
@pytest.mark.parametrize("dimension", [1, 2], ids=["d1_core_like", "d2_boundary"])
def test_pool_at_one_or_two_inputs_passes(settings, policy, dimension):
    """Check that pool at one or two inputs passes."""
    assert refuse(settings, policy, True, dimension) is None


@POOL_SETTINGS
@INFORMATION_POLICIES
def test_pool_at_three_inputs_is_refused(settings, policy):
    """Check that pool at three inputs is refused."""
    with pytest.raises(ValueError, match=refusal(policy, 3)):
        refuse(settings, policy, True, 3)


# ---------------------------------------------------------------------------
# (b) chembench_core.run on the real adapter (d = 7), mock proposer, dev profile.
# ---------------------------------------------------------------------------

class StopAtFirstFit(Exception):
    """Raised by the first SMC fit: the run got past every refusal before its output."""


@pytest.fixture
def stop_at_first_fit(monkeypatch):
    """Fixture that makes the first model fit raise so a run stops right after the refusal checks."""
    from . import chembench_core

    def first_fit(*args, **kwargs):
        """Stand-in for the timed fit that aborts the run at the first fit."""
        raise StopAtFirstFit

    monkeypatch.setattr(chembench_core, "_timed_fit", first_fit)


def chembench_args(output, cell, **options):
    """Build ChemBench runner arguments for a cell, leaving structure proposals and design policy to be resolved from the cell."""
    from .test_chembench_core import _args

    # structure_proposals None: the mode follows cell[0], as on the runner's command line.
    # design_policy None: the runner resolves it from cell[2], as when --design-policy is omitted.
    return _args(output, MANIFEST, cell=cell, structure_proposals=None,
                 **{"design_policy": None, **options})


@pytest.mark.usefixtures("require_autoscilab_source", "stop_at_first_fit")
@pytest.mark.parametrize("cell,policy", [("101", "joint_eig"), ("102", "epig")])
@pytest.mark.parametrize("options", [{}, {"design_search": "pool"}],
                         ids=["no_design_search_attribute", "explicit_pool"])
def test_a_real_information_cell_on_the_pool_path_is_refused_before_any_output(
        tmp_path, cell, policy, options):
    """Check that a real information cell on the pool path is refused before any output."""
    from . import chembench_core

    output = tmp_path / "run"
    args = chembench_args(output, cell, **options)
    assert hasattr(args, "design_search") == bool(options)
    with pytest.raises(ValueError, match=refusal(policy, 7)):
        chembench_core.run(args)
    assert not output.exists()


@pytest.mark.usefixtures("require_autoscilab_source", "stop_at_first_fit")
@pytest.mark.parametrize("cell,options,applies", [
    ("101", {"design_search": "ascent"}, True),
    ("100", {}, False),
    ("101", {"design_policy": "point_disagreement"}, True),
    ("001", {"design_policy": "space_filling", "design_search": "pool"}, True),
], ids=["101_ascent", "100_unset", "101_point_disagreement_unset", "001_space_filling_pool"])
def test_ascent_a_no_policy_cell_and_study_policies_get_past_the_refusal(
        tmp_path, cell, options, applies):
    """Check that ascent a no policy cell and study policies get past the refusal."""
    from . import chembench_core

    output = tmp_path / "run"
    with pytest.raises(StopAtFirstFit):
        chembench_core.run(chembench_args(output, cell, **options))
    config = json.loads((output / "config.json").read_text())
    assert config["cell"] == cell and config["design_policy_applies"] is applies
    assert config["design_policy"] == options.get("design_policy", "joint_eig")
    assert config.get("design_search") == ("ascent" if options.get("design_search") == "ascent"
                                           else None)
