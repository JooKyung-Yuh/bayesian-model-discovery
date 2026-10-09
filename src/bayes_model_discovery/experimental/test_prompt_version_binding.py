"""The numbered system prompt a run selects, records and registers.

Synthetic throughout. The tag every selection test uses is declared by a file this
fixture writes into a temporary directory - never under the package's own `prompts/`
and never named as a prompt version in shipped code, which `test_the_synthetic_tag_is
_never_a_shipped_prompt_version` below is what pins. No provider call is made here.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import count_proposer as cp
from .count_models import set_llm_priors
from .run_label_policy import bmd_label_receipt

#: A tag no packaged prompt file declares, and none may.
SYNTHETIC_TAG = "v9.9"
SYNTHETIC_FILE = "proposer_system_v9_9.txt"
SYNTHETIC_TEXT = (
    f"{cp.PROMPT_VARIANT_TAG_PREFIX} {SYNTHETIC_TAG}\n"
    "Reply with exactly one JSON object and nothing else.\n"
    "State the expression, the family and the parameters.\n"
)
PACKAGE_PROMPTS = Path(cp.__file__).with_name("prompts")


@pytest.fixture(autouse=True)
def restore_switches():
    """The prompt-variant override and the llm-priors switch are process-global."""
    yield
    cp.set_prompt_variant(None)
    set_llm_priors(False)


@pytest.fixture
def synthetic_prompts(tmp_path, monkeypatch):
    """A prompt directory carrying one synthetic numbered file, and nothing else."""
    directory = tmp_path / "prompts"
    directory.mkdir()
    (directory / SYNTHETIC_FILE).write_text(SYNTHETIC_TEXT)
    monkeypatch.setattr(cp, "PROMPT_DIRECTORY", directory)
    return directory


def test_the_synthetic_tag_is_never_a_shipped_prompt_version():
    """The fixture stays a fixture: no packaged file declares it and no source names it."""
    assert not (PACKAGE_PROMPTS / SYNTHETIC_FILE).exists()
    # `proposer_system_v2_3.txt` is shipped; the synthetic tag is still shipped by
    # nothing, and this list is what a further packaged file has to be declared in.
    # The arm-neutral pool generator and the five ChemBench stage-slot files (added on a
    # project decision) are packaged files too; none of them is
    # a numbered prompt.
    assert sorted(path.name for path in PACKAGE_PROMPTS.iterdir()) == [
        "pool_generator_v1.txt",
        "proposer_system_v2_0.txt", "proposer_system_v2_1.txt",
        "proposer_system_v2_2.txt", "proposer_system_v2_3.txt",
        "proposer_system_v2_4.txt", "proposer_system_v2_4_1.txt", "proposer_system_v2_5.txt",
        "proposer_system_v2_6_1.txt", "proposer_system_v2_6_2.txt",
        "proposer_system_v2_6_3.txt", "proposer_system_v2_6_4.txt",
        "proposer_system_v2_6_5.txt", "proposer_system_v2_6_6.txt",
        "proposer_system_v2_6_7.txt", "proposer_system_v2_6_7_1.txt",
        "proposer_system_v3_0_mechanism.txt",
        "slot_burst_paragraph_v1.txt", "slot_burst_refill_paragraph_v1.txt",
        "slot_fixed_role_prior_v1.json", "slot_repair_v1.txt", "slot_repair_v1_empty.txt"]
    for path in PACKAGE_PROMPTS.iterdir():
        assert SYNTHETIC_TAG not in path.read_text()
    for name in ("count_proposer.py", "boxinggym_core.py", "chembench_core.py", "core_count.py"):
        assert SYNTHETIC_TAG not in Path(cp.__file__).with_name(name).read_text()


def test_a_tag_resolves_to_the_numbered_file_beside_the_runner():
    """Test that a tag resolves to the numbered file beside the runner."""
    assert cp.prompt_version_file("v2.0") == cp.PROMPT_FILE
    assert cp.prompt_version_file("v2.2") == cp.PROMPT_FILE_LLM_PRIORS
    assert cp.prompt_version_file("v2.1").name == "proposer_system_v2_1.txt"


#: The tag the prompt-variant cohort runs, and the one section its file adds to the
#: byte-frozen v2.0 text. The removal rule below is the whole claim "v2.3 is v2.0 plus this
#: section": the first line, then the heading through the blank line that closes the section.
PARSIMONY_TAG = "v2.3"
PARSIMONY_FILE = "proposer_system_v2_3.txt"
PARSIMONY_HEADING = "Parsimony"


def test_the_v2_3_text_is_the_frozen_v2_0_text_plus_a_header_and_one_section():
    """Take the header line and the Parsimony section out and the v2.0 bytes are left.

    Nothing else in the frozen text may have moved while the section was added: one edited
    space anywhere else in the file, and the remainder is no longer those bytes.
    """
    path = cp.prompt_version_file(PARSIMONY_TAG)
    raw = path.read_bytes()
    assert path.name == PARSIMONY_FILE
    lines = raw.decode().splitlines(keepends=True)
    assert lines[0] == f"{cp.PROMPT_VARIANT_TAG_PREFIX} {PARSIMONY_TAG}\n"
    body = lines[1:]
    heading = next(i for i, line in enumerate(body) if line.startswith(PARSIMONY_HEADING))
    blank = next(i for i in range(heading, len(body)) if not body[i].strip())
    # The section is its heading, three rules and the blank line that ends it.
    assert blank - heading == 4
    assert "".join(body[:heading] + body[blank + 1:]).encode() == cp.PROMPT_FILE.read_bytes()
    # The file ends the way the v2.0 file does: one newline, and no blank line after it.
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")


#: The tag the structural-novelty arm runs. Its file is the v2.3 text with ONE more
#: section, so two removals - Structural novelty, then Parsimony - and the header line leave
#: the v2.0 bytes; removing Structural novelty alone leaves the v2.3 body.
NOVELTY_TAG = "v2.4"
NOVELTY_FILE = "proposer_system_v2_4.txt"
NOVELTY_HEADING = "Structural novelty"


def _without_section(body, heading, rules):
    heading_index = next(i for i, line in enumerate(body) if line.startswith(heading))
    blank = next(i for i in range(heading_index, len(body)) if not body[i].strip())
    assert blank - heading_index == rules + 1
    return body[:heading_index] + body[blank + 1:]


def test_the_v2_4_text_is_the_v2_3_text_plus_one_section_and_the_v2_0_text_plus_two():
    """Test that the v2 4 text is the v2 3 text plus one section and the v2 0 text plus two."""
    path = cp.prompt_version_file(NOVELTY_TAG)
    raw = path.read_bytes()
    assert path.name == NOVELTY_FILE
    lines = raw.decode().splitlines(keepends=True)
    assert lines[0] == f"{cp.PROMPT_VARIANT_TAG_PREFIX} {NOVELTY_TAG}\n"
    body = lines[1:]
    # Structural novelty is its heading, four rules and the blank line that ends it.
    without_novelty = _without_section(body, NOVELTY_HEADING, 4)
    v2_3 = cp.prompt_version_file(PARSIMONY_TAG).read_bytes().decode().splitlines(keepends=True)[1:]
    assert "".join(without_novelty) == "".join(v2_3)
    assert "".join(_without_section(without_novelty, PARSIMONY_HEADING, 3)).encode() == cp.PROMPT_FILE.read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")


def test_the_v2_4_tag_selects_the_packaged_file_and_is_the_identity_it_records():
    """Test that the v2 4 tag selects the packaged file and is the identity it records."""
    path = cp.prompt_version_file(NOVELTY_TAG)
    assert cp.select_prompt_version(NOVELTY_TAG) == path
    assert cp.prompt_variant_active()
    assert cp.system_prompt_bytes() == path.read_bytes()
    assert cp.prompt_identity() == {
        "prompt_version": NOVELTY_TAG,
        "prompt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_the_v2_3_tag_selects_the_packaged_file_and_is_the_identity_it_records():
    """The packaged file, the bytes sent and the recorded pair are one file, not three."""
    path = cp.prompt_version_file(PARSIMONY_TAG)
    assert cp.select_prompt_version(PARSIMONY_TAG) == path
    assert cp.prompt_variant_active()
    assert cp.system_prompt_bytes() == path.read_bytes()
    assert cp.prompt_identity() == {
        "prompt_version": PARSIMONY_TAG,
        "prompt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize("tag", ["v2", "2.3", "", None, "../secrets", "v2.3/../v2.0", "v2_3"])
def test_a_tag_that_is_not_a_version_number_is_refused(tag):
    """Test that a tag that is not a version number is refused."""
    with pytest.raises(ValueError, match="prompt version tag"):
        cp.prompt_version_file(tag)


def test_an_unknown_tag_is_refused_and_selects_nothing(synthetic_prompts):
    """Test that an unknown tag is refused and selects nothing."""
    with pytest.raises(ValueError, match="no packaged prompt file"):
        cp.select_prompt_version("v9.8")
    assert cp.prompt_identity()["prompt_version"] == cp.PROMPT_VERSION


def test_a_header_that_disagrees_with_the_tag_is_refused_and_selects_nothing(synthetic_prompts):
    """Test that a header that disagrees with the tag is refused and selects nothing."""
    (synthetic_prompts / SYNTHETIC_FILE).write_text(
        f"{cp.PROMPT_VARIANT_TAG_PREFIX} v9.7\nbody\n")
    with pytest.raises(ValueError, match="must declare"):
        cp.select_prompt_version(SYNTHETIC_TAG)
    assert not cp.prompt_variant_active()
    (synthetic_prompts / SYNTHETIC_FILE).write_text("no header at all\n")
    with pytest.raises(ValueError, match="must declare"):
        cp.select_prompt_version(SYNTHETIC_TAG)
    assert not cp.prompt_variant_active()


def test_the_default_tag_and_the_absent_option_select_nothing(synthetic_prompts):
    """Test that the default tag and the absent option select nothing."""
    assert cp.select_prompt_version(None) is None
    assert cp.select_prompt_version(cp.PROMPT_VERSION) is None
    assert not cp.prompt_variant_active()
    assert cp.prompt_identity() == {
        "prompt_version": "v2.0",
        "prompt_sha256": hashlib.sha256(cp.PROMPT_FILE.read_bytes()).hexdigest()}


def test_a_selected_tag_is_the_identity_every_record_reads(synthetic_prompts):
    """config.json, the receipt and the bytes actually sent are one pair, not three."""
    path = cp.select_prompt_version(SYNTHETIC_TAG)
    assert path == synthetic_prompts / SYNTHETIC_FILE
    sent = cp.system_prompt_bytes()
    assert sent == SYNTHETIC_TEXT.encode()
    assert cp.prompt_identity() == {
        "prompt_version": SYNTHETIC_TAG,
        "prompt_sha256": hashlib.sha256(sent).hexdigest()}


def test_stating_both_prompt_options_is_refused(synthetic_prompts, tmp_path):
    """Test that stating both prompt options is refused."""
    variant = tmp_path / "variant.txt"
    variant.write_text(f"{cp.PROMPT_VARIANT_TAG_PREFIX} v9.6\nbody\n")
    args = SimpleNamespace(prompt_version=SYNTHETIC_TAG, prompt_variant=str(variant))
    with pytest.raises(ValueError, match="state at most one"):
        cp.select_run_prompt(args)
    assert not cp.prompt_variant_active()


def receipt_settings(bundle, **overrides):
    """The settings a runner's own receipt would bind, resolved the way `run` resolves it."""
    values = {"profile": "dev", "rounds": 1, "cell": "100", "seed": 11, "task": "t",
              "proposer": "mock", "llm_model": None, "run_label": None, "priors": "off"}
    values.update(overrides)
    args = SimpleNamespace(**values)
    _, prompt = cp.select_run_prompt(args, llm_priors=args.priors == "llm")
    label_args = SimpleNamespace(**{**vars(args), **prompt})
    receipt = bmd_label_receipt(bundle, label_args, task="t", metadata={})
    return receipt["binding"]["settings"], prompt


@pytest.mark.parametrize("bundle", ["core", "boxinggym", "chembench"])
def test_the_receipt_binds_the_prompt_the_run_selected(synthetic_prompts, bundle):
    """Test that the receipt binds the prompt the run selected."""
    settings, prompt = receipt_settings(bundle, prompt_version=SYNTHETIC_TAG)
    assert settings["prompt_version"] == SYNTHETIC_TAG
    assert settings["prompt_sha256"] == hashlib.sha256(cp.system_prompt_bytes()).hexdigest()
    # `**prompt_identity()` is what config.json records, so the three agree by construction.
    assert prompt == cp.prompt_identity() == {k: settings[k] for k in prompt}


@pytest.mark.parametrize("bundle", ["core", "boxinggym", "chembench"])
def test_an_absent_option_binds_the_frozen_default(synthetic_prompts, bundle):
    """Test that an absent option binds the frozen default."""
    settings, _ = receipt_settings(bundle)
    assert settings["prompt_version"] == "v2.0"
    assert settings["prompt_sha256"] == hashlib.sha256(cp.PROMPT_FILE.read_bytes()).hexdigest()


#: A development variant's free-form tag (never a version).
VARIANT_TAG = "p1v_fixup_variant"


def write_variant(directory):
    """Write a development prompt variant file into a directory; return its path and the expected identity pair."""
    variant = directory / "variant.txt"
    variant.write_text(f"{cp.PROMPT_VARIANT_TAG_PREFIX} {VARIANT_TAG}\nSay something different.\n")
    return variant, {"prompt_version": VARIANT_TAG,
                     "prompt_sha256": hashlib.sha256(variant.read_bytes()).hexdigest()}


def test_a_development_variant_alone_is_the_identity_the_receipt_binds(synthetic_prompts, tmp_path):
    """The selection applies the variant itself, so the receipt never binds the default."""
    variant, expected = write_variant(tmp_path)
    settings, prompt = receipt_settings("boxinggym", prompt_variant=str(variant))
    assert prompt == cp.prompt_identity() == {k: settings[k] for k in prompt} == expected
    assert cp.system_prompt_bytes() == variant.read_bytes()
    # The next selection with no variant stated reads the default again: nothing leaks.
    settings, prompt = receipt_settings("boxinggym", prompt_variant=None)
    assert prompt == cp.prompt_identity() == {k: settings[k] for k in prompt}
    assert prompt["prompt_version"] == "v2.0" and not cp.prompt_variant_active()


def test_a_variant_only_development_run_records_one_pair_in_receipt_and_config(tmp_path, monkeypatch):
    """On a real run the receipt pair, the config.json pair and the variant file's own sha
    are one pair; a paper budget without production registration still accepts it."""
    bg = pytest.importorskip("bayes_model_discovery.experimental.boxinggym_core")
    pytest.importorskip("pymc")
    pytest.importorskip(bg.TASKS["death"].module)
    from .test_death_regime_switches import bg_run_args

    variant, expected = write_variant(tmp_path)
    out = tmp_path / "run"
    bg.run(bg_run_args(out, run_label="development", prompt_variant=str(variant)))
    config = json.loads((out / "config.json").read_text())
    assert config["label_receipt"]["run_label"] == "development"
    assert {k: config[k] for k in expected} == expected
    assert {k: config["label_receipt"]["binding"]["settings"][k] for k in expected} == expected

    from .test_run_label_policy import ReachedValidatedOutput

    paper = tmp_path / "paper"
    original_mkdir = Path.mkdir

    def stop_at_output(path, *args, **kwargs):
        """Raise at the guarded output directory; otherwise delegate to the original mkdir."""
        if path == paper:
            raise ReachedValidatedOutput
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", stop_at_output)
    with pytest.raises(ReachedValidatedOutput):
        bg.run(bg_run_args(paper, profile="paper", prompt_variant=str(variant)))
    assert not paper.exists() and cp.prompt_variant_active()
    assert cp.prompt_identity() == expected


@pytest.mark.parametrize("bundle", ["core", "boxinggym"])
def test_priors_llm_alone_binds_the_v2_2_prompt_before_the_switch_is_applied(bundle):
    """The receipt is resolved before `set_llm_priors` runs, so the decision is stated."""
    settings, prompt = receipt_settings(bundle, priors="llm")
    assert settings["prompt_version"] == "v2.2"
    assert settings["prompt_sha256"] == hashlib.sha256(
        cp.PROMPT_FILE_LLM_PRIORS.read_bytes()).hexdigest()
    set_llm_priors(True)
    assert cp.prompt_identity() == prompt


@pytest.mark.parametrize("tag", ["v2.0", "v2.2", SYNTHETIC_TAG])
def test_a_version_stated_beside_priors_llm_is_refused_and_selects_nothing(synthetic_prompts, tag):
    """`--priors llm` already determines the prompt (v2.2); a stated tag beside it would
    either be silently overridden (v2.0) or rename the bytes sent - refused, nothing applied."""
    args = SimpleNamespace(prompt_version=tag, priors="llm")
    with pytest.raises(ValueError, match="state at most one prompt determinant"):
        cp.select_run_prompt(args, llm_priors=True)
    assert not cp.prompt_variant_active()
    assert cp.prompt_identity()["prompt_version"] == "v2.0"


def test_a_version_stated_beside_priors_llm_is_refused_before_any_output(tmp_path):
    """Test that a version stated beside priors llm is refused before any output."""
    bg = pytest.importorskip("bayes_model_discovery.experimental.boxinggym_core")
    from .test_death_regime_switches import bg_run_args

    out = tmp_path / "never"
    with pytest.raises(ValueError, match="state at most one prompt determinant"):
        bg.run(bg_run_args(out, priors="llm", prompt_version="v2.0"))
    assert not out.exists()


@pytest.mark.parametrize("module,argv", [
    ("core_count", ["core_count", "--output", "/dev/null/never", "--structure", "out",
                    "--truth-family", "beta_binomial", "--seed", "11"]),
])
def test_an_absent_option_leaves_no_args_key_and_a_stated_one_arrives(monkeypatch, module, argv):
    """SUPPRESS: an unstated option leaves no `vars(args)` key, so config.json is unchanged."""
    import importlib
    import sys
    runner = importlib.import_module("." + module, package=cp.__package__)
    captured = []
    monkeypatch.setattr(runner, "run", captured.append)
    monkeypatch.setattr(sys, "argv", list(argv))
    runner.main()
    assert "prompt_version" not in vars(captured[-1])
    monkeypatch.setattr(sys, "argv", [*argv, "--prompt-version", SYNTHETIC_TAG])
    runner.main()
    assert captured[-1].prompt_version == SYNTHETIC_TAG


#: The domain arm. Its file is the v2.4 text with the second
#: Structural novelty rule (the sub-expression ban) removed and ONE more section, Domain
#: (one rule), so the header line and three removals - Domain, Structural novelty (three
#: rules now), Parsimony - leave the v2.0 bytes.
DOMAIN_TAG = "v2.5"
DOMAIN_FILE = "proposer_system_v2_5.txt"
DOMAIN_HEADING = "Domain"


def test_the_v2_5_text_is_the_v2_4_text_minus_one_rule_plus_one_section_and_the_v2_0_text_plus_three():
    """Test that the v2 5 text is the v2 4 text minus one rule plus one section and the v2 0 text plus three."""
    path = cp.prompt_version_file(DOMAIN_TAG)
    raw = path.read_bytes()
    assert path.name == DOMAIN_FILE
    lines = raw.decode().splitlines(keepends=True)
    assert lines[0] == f"{cp.PROMPT_VARIANT_TAG_PREFIX} {DOMAIN_TAG}\n"
    body = lines[1:]
    without_domain = _without_section(body, DOMAIN_HEADING, 1)
    v2_4 = cp.prompt_version_file(NOVELTY_TAG).read_bytes().decode().splitlines(keepends=True)[1:]
    novelty_index = next(i for i, line in enumerate(v2_4) if line.startswith(NOVELTY_HEADING))
    # v2.5 drops exactly the second rule of v2.4's Structural novelty section.
    dropped = v2_4[novelty_index + 2]
    assert dropped.startswith("- ") and "sub-expression" in dropped
    assert "".join(without_domain) == "".join(v2_4[:novelty_index + 2] + v2_4[novelty_index + 3:])
    without_novelty = _without_section(without_domain, NOVELTY_HEADING, 3)
    assert "".join(_without_section(without_novelty, PARSIMONY_HEADING, 3)).encode() == cp.PROMPT_FILE.read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")


def test_the_v2_5_tag_selects_the_packaged_file_and_is_the_identity_it_records():
    """Test that the v2 5 tag selects the packaged file and is the identity it records."""
    path = cp.prompt_version_file(DOMAIN_TAG)
    assert cp.select_prompt_version(DOMAIN_TAG) == path
    assert cp.prompt_variant_active()
    assert cp.system_prompt_bytes() == path.read_bytes()
    assert cp.prompt_identity() == {
        "prompt_version": DOMAIN_TAG,
        "prompt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


#: The novelty-only arm (a pilot, 2026-09-20).
#: Its file is the v2.4 text without the Parsimony section and without the fourth
#: Structural novelty rule (the coefficient-count preference), so the header line and ONE
#: removal - Structural novelty (three rules now) - leave the v2.0 bytes. It carries no
#: instruction about the number of coefficients at all.
NOVELTY_ONLY_TAG = "v2.4.1"
NOVELTY_ONLY_FILE = "proposer_system_v2_4_1.txt"


def test_the_v2_4_1_text_is_the_v2_4_text_minus_parsimony_and_one_rule_and_the_v2_0_text_plus_one():
    """Test that the v2 4 1 text is the v2 4 text minus parsimony and one rule and the v2 0 text plus one."""
    path = cp.prompt_version_file(NOVELTY_ONLY_TAG)
    raw = path.read_bytes()
    assert path.name == NOVELTY_ONLY_FILE
    lines = raw.decode().splitlines(keepends=True)
    assert lines[0] == f"{cp.PROMPT_VARIANT_TAG_PREFIX} {NOVELTY_ONLY_TAG}\n"
    body = lines[1:]
    v2_4 = cp.prompt_version_file(NOVELTY_TAG).read_bytes().decode().splitlines(keepends=True)[1:]
    v2_4_without_parsimony = _without_section(v2_4, PARSIMONY_HEADING, 3)
    novelty_index = next(i for i, line in enumerate(v2_4_without_parsimony)
                         if line.startswith(NOVELTY_HEADING))
    # v2.4.1 drops exactly the fourth rule of v2.4's Structural novelty section.
    dropped = v2_4_without_parsimony[novelty_index + 4]
    assert dropped.startswith("- ") and "fewer coefficients" in dropped
    assert "".join(body) == "".join(v2_4_without_parsimony[:novelty_index + 4]
                                    + v2_4_without_parsimony[novelty_index + 5:])
    assert "".join(_without_section(body, NOVELTY_HEADING, 3)).encode() == cp.PROMPT_FILE.read_bytes()
    text = raw.decode()
    assert PARSIMONY_HEADING not in text and "fewer coefficients" not in text
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")


def test_the_v2_4_1_tag_selects_the_packaged_file_and_is_the_identity_it_records():
    """Test that the v2 4 1 tag selects the packaged file and is the identity it records."""
    path = cp.prompt_version_file(NOVELTY_ONLY_TAG)
    assert cp.select_prompt_version(NOVELTY_ONLY_TAG) == path
    assert cp.prompt_variant_active()
    assert cp.system_prompt_bytes() == path.read_bytes()
    assert cp.prompt_identity() == {
        "prompt_version": NOVELTY_ONLY_TAG,
        "prompt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


#: The one-clause ablation family (2026-09-20). Each file is the v2.0 text with exactly
#: ONE section added, so an arm's effect belongs to that section alone. The heading and
#: the rule count are what `_without_section` needs to take the section back out.
ONE_CLAUSE_FAMILY = (
    ("v2.6.1", "Coefficient position", 2),
    ("v2.6.2", "Input scale", 1),
    ("v2.6.3", "Framing", 2),
    ("v2.6.4", "Interpretation", 2),
    ("v2.6.5", "Structural novelty", 1),
    ("v2.6.6", "Operator structure", 1),
    ("v2.6.7", "Domain", 1),
    ("v2.6.7.1", "Domain", 1),
)

#: The registered file of the seventh one-clause arm, Domain (ladder edition 20), and its corrected
#: successor (2026-09-23).
#: The predecessor is immutable: its bytes stay on disk under the hash the edition
#: sealed, and the correction lives only in the successor file under its own tag.
A7_PREDECESSOR_TAG, A7_PREDECESSOR_SHA256 = (
    "v2.6.7", "5b2172d70f93a21f1c46c2609c9b089a9062a9f9e49995953437fcbc98ff3b99")
A7_CORRECTED_TAG = "v2.6.7.1"

#: Words a proposal prompt may never carry: a benchmark or task name, or the name of a
#: domain form whose shape is a true law of one of them. The clause texts are general
#: statements about the grammar and about how to propose; naming a mechanism family here
#: would hand the proposer the answer (the standing rule against lifting a result into a
#: prompt, and the 2026-09-20 gate condition on the interpretation clause).
FORBIDDEN_IN_A_CLAUSE = (
    "boxinggym", "chembench", "forcebench", "neuron", "death", "peregrin", "dugong",
    "survival", "hyperbolic", "moral", "emotion", "michaelis", "menten", "enzyme",
    "saturat", "hill ", "kinetic", "inhibition", "bisubstrate", "half-life", "decay",
)


@pytest.mark.parametrize("tag,heading,rules", ONE_CLAUSE_FAMILY)
def test_each_one_clause_file_is_the_v2_0_text_plus_exactly_that_clause(tag, heading, rules):
    """Test that each one clause file is the v2 0 text plus exactly that clause."""
    path = cp.prompt_version_file(tag)
    raw = path.read_bytes()
    lines = raw.decode().splitlines(keepends=True)
    assert lines[0] == f"{cp.PROMPT_VARIANT_TAG_PREFIX} {tag}\n"
    assert "".join(_without_section(lines[1:], heading, rules)).encode() == cp.PROMPT_FILE.read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")


@pytest.mark.parametrize("tag,heading,rules", ONE_CLAUSE_FAMILY)
def test_each_one_clause_tag_selects_its_file_and_is_the_identity_it_records(tag, heading, rules):
    """Test that each one clause tag selects its file and is the identity it records."""
    path = cp.prompt_version_file(tag)
    assert cp.select_prompt_version(tag) == path
    assert cp.system_prompt_bytes() == path.read_bytes()
    assert cp.prompt_identity() == {
        "prompt_version": tag,
        "prompt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize("tag,heading,rules", ONE_CLAUSE_FAMILY)
def test_a_one_clause_file_names_no_benchmark_and_no_domain_form(tag, heading, rules):
    """Test that a one clause file names no benchmark and no domain form."""
    text = cp.prompt_version_file(tag).read_text().lower()
    assert [word for word in FORBIDDEN_IN_A_CLAUSE if word in text] == []


def test_the_corrected_a7_file_keeps_its_predecessor_immutable_and_differs_only_in_the_domain_rule():
    """Test that the corrected a7 file keeps its predecessor immutable and differs only in the domain rule."""
    old = cp.prompt_version_file(A7_PREDECESSOR_TAG).read_bytes()
    new = cp.prompt_version_file(A7_CORRECTED_TAG).read_bytes()
    assert hashlib.sha256(old).hexdigest() == A7_PREDECESSOR_SHA256
    assert hashlib.sha256(new).hexdigest() != A7_PREDECESSOR_SHA256
    old_lines, new_lines = old.decode().splitlines(), new.decode().splitlines()
    assert len(old_lines) == len(new_lines)
    changed = [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
    assert changed == [0, 8]
    assert new_lines[7] == "Domain"
    # The predecessor's error: it called the [0, 1]-scaled and non-negative inputs safe for
    # log, sqrt and non-integer powers alike; log(0) is not finite. The successor separates them.
    assert "are safe arguments" in old_lines[8]
    assert "are safe arguments" not in new_lines[8]
    assert "safe for sqrt" in new_lines[8]
    assert "strictly positive range" in new_lines[8]
