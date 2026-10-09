"""The `--prior-template` and `--priors` ablation switches on the BoxingGym and ChemBench runners.

BoxingGym-chain only: `boxinggym_core` and `chembench_core` have no CORE-chain
counterpart, so their half of the two switches is pinned here rather than in the
shared `test_prior_switches.py`.
"""

import sys

import numpy as np
import pytest

from . import boxinggym_core, chembench_core, core_count
from .boxinggym_core import prior_particles as bg_prior_particles
from .count_models import (
    DEFAULT_PRIOR_TEMPLATE,
    PRIOR_TEMPLATE_NAMES,
    Hypothesis,
    active_prior_template,
    llm_priors_enabled,
    set_llm_priors,
    set_prior_template,
)


@pytest.fixture(autouse=True)
def restore_switches():
    """Restore switches."""
    template, priors = active_prior_template(), llm_priors_enabled()
    yield
    set_prior_template(template)
    set_llm_priors(priors)


def test_both_runner_clis_declare_the_two_switches_with_production_defaults(monkeypatch):
    """`--prior-template default --priors off` is what a run gets when it asks for neither."""
    for module, argv in (
        (
            core_count,
            ["core_count", "--output", "/dev/null/never", "--profile", "dev"],
        ),
        (
            boxinggym_core,
            ["boxinggym_core", "--output", "/dev/null/never", "--task", "death"],
        ),
        (
            chembench_core,
            ["chembench_core", "--output", "/dev/null/never", "--manifest", "m", "--noise", "0.01"],
        ),
    ):
        captured = []
        monkeypatch.setattr(module, "run", captured.append)
        monkeypatch.setattr(sys, "argv", argv)
        module.main()
        (args,) = captured
        assert args.prior_template == DEFAULT_PRIOR_TEMPLATE == "default"
        # ChemBench takes `--prior-template` only: its candidates are RegressionHypothesis/
        # ScalarHypothesis, which carry no `coefficient_priors`, so `--priors` is not
        # offered there at all rather than offered and silently dropped.
        assert getattr(args, "priors", "off") == "off"
    # and the three templates are the only accepted values
    monkeypatch.setattr(sys, "argv", ["core_count", "--output", "x", "--prior-template", "huge"])
    with pytest.raises(SystemExit):
        core_count.main()
    assert PRIOR_TEMPLATE_NAMES == ("default", "wide", "narrow")


@pytest.mark.parametrize("task", ["peregrines", "dugongs", "location", "emotion"])
def test_boxinggym_refuses_llm_priors_off_the_exhaustive_count_path(tmp_path, monkeypatch, task):
    """Refused BEFORE any output exists, not dropped silently at conversion."""
    out = tmp_path / task
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "boxinggym_core",
            "--output",
            str(out),
            "--task",
            task,
            "--profile",
            "dev",
            "--rounds",
            "1",
            "--priors",
            "llm",
        ],
    )
    with pytest.raises(ValueError, match="exhaustive count/binary path only"):
        boxinggym_core.main()
    assert not out.exists()


def test_boxinggym_points_variant_joins_the_composite_and_the_prior_tokens():
    """Check that boxinggym points variant joins the composite and the prior tokens."""
    single, composite = boxinggym_core.TASKS["death"], boxinggym_core.TASKS["emotion"]
    assert boxinggym_core.points_variant({"prior_template": "default", "priors": "off"}, single) == ""
    assert (
        boxinggym_core.points_variant({"prior_template": "wide", "priors": "off"}, single)
        == "prior_wide"
    )
    assert (
        boxinggym_core.points_variant({"prior_template": "default", "priors": "llm"}, single)
        == "prior_llm"
    )
    # sorted alphabetically, joined by `+`, exactly as core_count.points_variant
    assert boxinggym_core.points_variant(
        {"prior_template": "narrow", "priors": "llm"}, composite
    ) == "composite_over_8_components+prior_llm+prior_narrow"
    assert (
        boxinggym_core.points_variant({"prior_template": "default", "priors": "off"}, composite)
        == "composite_over_8_components"
    )


def test_chembench_points_variant_adds_the_prior_token_to_the_structure_token():
    """Check that chembench points variant adds the prior token to the structure token."""
    assert chembench_core._points_variant("catalog") == ""
    assert chembench_core._points_variant("free") == "structure_free"
    assert chembench_core._points_variant("catalog", "wide") == "prior_wide"
    assert chembench_core._points_variant("free", "narrow") == "prior_narrow+structure_free"


def test_boxinggym_prior_particles_follow_the_template_and_a_proposed_prior():
    """Check that boxinggym prior particles follow the template and a proposed prior."""
    parameters = (("b0", "intercept"), ("b1", "signed"))
    model = Hypothesis("b0+b1*z", "binomial", parameters)
    plain = bg_prior_particles(model, 4096)
    assert 1.7 < plain["b0"].std() < 2.3 and 0.85 < plain["b1"].std() < 1.15
    proposed = bg_prior_particles(
        Hypothesis(
            "b0+b1*z",
            "binomial",
            parameters,
            coefficient_priors=(("b1", ("Normal", 0.0, 0.25)),),
        ),
        4096,
    )
    assert 0.2 < proposed["b1"].std() < 0.3
    set_prior_template("wide")
    wide = bg_prior_particles(model, 4096)
    assert 3.5 < wide["b0"].std() < 4.5
    assert np.isfinite(wide["b0"]).all()


# --------------------------------- `--prior-template` refusal on scalar/real tasks

A3_SCALAR_REFUSAL = (
    "the proposed-prior ablation is defined on the count path; scalar/real noise priors do not "
    "follow the template"
)


@pytest.mark.parametrize("task", ["peregrines", "dugongs", "location", "emotion"])
def test_boxinggym_refuses_a_non_default_prior_template_off_the_exhaustive_count_path(
    tmp_path, monkeypatch, task
):
    """Refused BEFORE any output exists, exactly like the `--priors` refusal above.

    `peregrines`/`dugongs` (`positive_inference.ScalarHypothesis`) and
    `location`/`emotion` (`real_inference.RealHypothesis`) are the same four
    non-exhaustive tasks `test_boxinggym_refuses_llm_priors_off_the_exhaustive_
    count_path` uses - `boxinggym_core.SCALAR_NOISE_PRIORS`/`real_inference.
    REAL_NOISE_PRIORS`'s import-time role-prior copies and `ScalarHypothesis.
    as_dict`'s literal record block do not follow a non-default template
    (an earlier change note), so this refusal seals
    that off rather than building a partially-templated record.
    """
    out = tmp_path / task
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "boxinggym_core",
            "--output",
            str(out),
            "--task",
            task,
            "--profile",
            "dev",
            "--rounds",
            "1",
            "--prior-template",
            "wide",
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        boxinggym_core.main()
    assert not out.exists()
    assert A3_SCALAR_REFUSAL in str(excinfo.value)


def test_chembench_refuses_a_non_default_prior_template_unconditionally(tmp_path, monkeypatch):
    """ChemBench has no exhaustive-count path at all (always `lognormal`/`gamma`
    `positive_inference.ScalarHypothesis`), so the refusal is unconditional -
    checked before the (never-created) manifest is even read."""
    out = tmp_path / "chembench_wide"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chembench_core",
            "--output",
            str(out),
            "--manifest",
            "this-manifest-is-never-read.json",
            "--noise",
            "0.01",
            "--prior-template",
            "wide",
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        chembench_core.main()
    assert not out.exists()
    assert A3_SCALAR_REFUSAL in str(excinfo.value)
