"""Write the scientist's first requests, as the kit would send them, for a refusal probe (0 USD here; the probe itself is
sent by whoever runs it).

  python -m appr_kit.probe_prompts --task dugongs --seed 31 --out <file.jsonl> [--variants released,no_thought]

One JSON line per (variant, request): {"variant", "request_index", "description", "body": {model, max_tokens, system,
messages, thinking, output_config}, "thought_tags": {"system": n, "user": n}, "prompt_filter": {...}}. The body is exactly
what claude_path.ClaudePath.call builds (max_tokens floor 16000, adaptive thinking, effort medium; the filter applied
under no_thought), captured from a client that records the call and answers with a fixed <observe> line. Two requests per
variant: the released first follow-up (agent.py 111, no result yet) and the second follow-up after a synthetic result
(agent.py 113). The system string is the environment's released system message plus our D0 block, as the driver builds
it at the first scientist call (checkpoint 0 adds the prior-mode program on top in a real run; the probe is about the
<thought> text, which lives in these two parts).
"""
import argparse
import contextlib
import json
import os
import sys
import tempfile
from types import SimpleNamespace

KIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KIT_DIR not in sys.path:
    sys.path.insert(0, KIT_DIR)
sys.dont_write_bytecode = True

from appr_kit import paths  # noqa: E402

SYNTHETIC_RESULT = "12"   # a plausible-looking observation string for the second follow-up; nothing from a run


class CapturingClient:
    def __init__(self):
        self.bodies = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model, max_tokens, system, messages, thinking=None, temperature=None, extra_body=None):
        # the same signature as the kit's ScriptedClient, so sdk_call_kwargs passes every field directly
        body = {"model": model, "max_tokens": max_tokens, "system": system, "messages": messages}
        if thinking is not None:
            body["thinking"] = thinking
        if temperature is not None:
            body["temperature"] = temperature
        body.update(extra_body or {})
        body = json.loads(json.dumps(body))
        self.bodies.append(body)
        text = "<observe>1.0</observe>"
        usage = SimpleNamespace(input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=usage, stop_reason="end_turn",
                               model=body.get("model"), _request_id=f"probe_{len(self.bodies)}")


def build(task, seed, variants):
    checkout = paths.checkout_dir()
    paths.import_checkout(checkout)
    from appr_kit import claude_path, prompt_filter, worlds
    from boxing_gym.agents import agent as agent_module
    RE, _info = paths.load_run_experiment(checkout)
    rows = []
    for variant in variants:
        tmp = tempfile.mkdtemp(prefix="appr_probe_")
        ledger = claude_path.UsageLedger(tmp, cap_cell_usd=1.0, cap_folder_usd=1.0, mock=True)
        client = CapturingClient()
        path = claude_path.ClaudePath(ledger, client, prompt_variant=variant)
        with contextlib.ExitStack() as stack:
            world = worlds.World(task, seed)
            world.build(stack, stdout_path=os.path.join(tmp, "world_build_stdout.txt"))
            env, goal = world.env, world.goal
            env.include_prior = True
            saved = agent_module.anthropic
            agent_module.anthropic = type("NoRealClient", (), {"Anthropic": staticmethod(lambda: None)})
            try:
                scientist = RE.LMExperimenter(model_name="claude-opus-5-5", temperature=0.0, max_tokens=512)
            finally:
                agent_module.anthropic = saved
            scientist.llm = claude_path.ScientistClient(path)
            base_system = goal.get_system_message(env.include_prior) + "\n" + world.d0_block()
            scientist.set_system_message(base_system)
            scientist.generate_actions(None)                 # agent.py 111: the first follow-up
            scientist.generate_actions(SYNTHETIC_RESULT)     # agent.py 113: after a result
        usage = [json.loads(l) for l in open(os.path.join(tmp, "llm_usage.jsonl")) if l.strip()]
        for i, (body, row) in enumerate(zip(client.bodies, usage)):
            user_text = "".join(
                (m["content"] if isinstance(m["content"], str) else "".join(b["text"] for b in m["content"]))
                for m in body["messages"] if m["role"] == "user")
            rows.append({"variant": variant, "request_index": i, "task": task, "seed": seed,
                         "description": ["first follow-up (no result yet)", "second follow-up (after a synthetic result)"][i],
                         "body": body,
                         "thought_tags": {"system": prompt_filter.remaining_tags(body["system"]),
                                          "user": prompt_filter.remaining_tags(user_text)},
                         "prompt_filter": row.get("prompt_filter"),
                         "system_sha256": prompt_filter.sha256(body["system"])})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("death", "dugongs", "peregrines", "hyperbolic", "survival"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="released,no_thought")
    args = ap.parse_args(argv)
    rows = build(args.task, args.seed, [v.strip() for v in args.variants.split(",") if v.strip()])
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    for r in rows:
        print(json.dumps({k: r[k] for k in ("variant", "request_index", "thought_tags", "system_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
