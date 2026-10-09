"""The narration stub is scoped to the environment module (a blocker
fixed on 2026-09-13): moral_machines/emotion see a stub `openai.OpenAI()` for their
observation narration, while the structure proposer built with
`--proposer openai` gets the real SDK client. No key, no network, paid 0."""
import importlib
from types import SimpleNamespace
from typing import ClassVar

import openai
import pytest

from bayes_model_discovery.experimental import boxinggym_core as bc
from bayes_model_discovery.experimental import count_proposer

MODULES = ("boxing_gym.envs.moral_machines", "boxing_gym.envs.emotion")


class FakeRealClient:
    """Stands in for openai.OpenAI(): records that it was constructed, no I/O."""

    instances: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        FakeRealClient.instances.append(self)

    @staticmethod
    def _create(**body):
        raise AssertionError("the proposer must not be called in this test")


def test_both_narration_tasks_name_their_environment_module():
    """Both narration tasks, and only they, declare their environment module."""
    for task, module in (("moral_machines", MODULES[0]), ("emotion", MODULES[1])):
        spec = bc.TASKS[task]
        assert spec.openai_narration and spec.openai_narration_module == module
    assert not any(spec.openai_narration_module for name, spec in bc.TASKS.items()
                   if not spec.openai_narration)


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.usefixtures("require_boxinggym_source")
def test_the_stub_reaches_only_the_environment_module(module_name):
    """The narration stub is visible only inside the environment module, not globally."""
    module = importlib.import_module(module_name)
    real_openai_class = openai.OpenAI
    with bc.openai_narration_stub(module_name) as absorbed:
        # the global module is untouched ...
        assert openai.OpenAI is real_openai_class
        # ... the environment's own name yields the stub, other attributes forward
        stub = module.openai.OpenAI()
        assert isinstance(stub, bc._NarrationStubClient)
        assert module.openai.__name__ == "openai"
        stub.chat.completions.create(model="narration", messages=[{"role": "user", "content": "x"}])
        assert absorbed == [{"model": "narration", "messages": 1}]
        # the other narration module is not touched by this context
        other = importlib.import_module(next(m for m in MODULES if m != module_name))
        assert other.openai is openai
    assert module.openai is openai


@pytest.mark.usefixtures("require_boxinggym_source")
def test_the_openai_proposer_gets_the_real_sdk_client_under_the_stub(tmp_path, monkeypatch):
    """Under the stub, the openai proposer is built with the real SDK client."""
    monkeypatch.setattr(openai, "OpenAI", FakeRealClient)
    FakeRealClient.instances.clear()
    args = SimpleNamespace(proposer="openai", llm_model="gpt-4o")
    module = importlib.import_module(MODULES[0])
    with bc.openai_narration_stub(MODULES[0]) as absorbed:
        proposer = bc.build_count_proposer(args, tmp_path, narration_stubbed=True)
        assert isinstance(proposer.client, FakeRealClient)          # real SDK constructor path
        assert proposer.injected_client is False
        assert proposer.client.kwargs.get("max_retries") == 0
        # meanwhile the environment's narration still goes to the stub
        module.openai.OpenAI().chat.completions.create(model="m", messages=[])
        assert len(absorbed) == 1 and len(FakeRealClient.instances) == 1


def test_the_guard_refuses_a_proposer_that_received_the_stub(tmp_path, monkeypatch):
    # Simulate the old defect: something replaced the GLOBAL openai.OpenAI.
    """The guard refuses a proposer that received the narration stub."""
    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: bc._NarrationStubClient([]))
    args = SimpleNamespace(proposer="openai", llm_model="gpt-4o")
    with pytest.raises(RuntimeError, match="narration stub"):
        bc.build_count_proposer(args, tmp_path, narration_stubbed=True)
    # without a narration task the same client is not examined (no guard, no output)
    plain = tmp_path / "plain"
    plain.mkdir()
    proposer = bc.build_count_proposer(args, plain, narration_stubbed=False)
    assert isinstance(proposer.client, bc._NarrationStubClient)


@pytest.mark.usefixtures("require_boxinggym_source")
def test_the_anthropic_path_is_untouched_by_the_stub(tmp_path, monkeypatch):
    """The anthropic proposer path is unaffected by the narration stub."""
    built = {}

    class FakeAnthropic:
        """Stand-in Anthropic proposer that records its transport directory."""
        def __init__(self, records, **kwargs):
            built["records"] = records
            self.client = object()
            self.injected_client = False

    monkeypatch.setattr(count_proposer, "AnthropicCountProposer", FakeAnthropic)
    args = SimpleNamespace(proposer="anthropic", llm_model="claude-opus-4-7")
    with bc.openai_narration_stub(MODULES[1]):
        proposer = bc.build_count_proposer(args, tmp_path, narration_stubbed=True)
    assert isinstance(proposer, FakeAnthropic) and built["records"] == tmp_path / "llm_transport"
    assert bc.build_count_proposer(SimpleNamespace(proposer="mock", llm_model=None), tmp_path) is None
