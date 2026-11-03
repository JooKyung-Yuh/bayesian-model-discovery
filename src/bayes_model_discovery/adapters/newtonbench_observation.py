"""NewtonBench observation bridge (the vanilla_equation system); no fitting, proposal service or LLM calls.

This is the adapter the paper's NewtonBench runs used (bound by the development tree's runner as
`--benchmark newtonbench`; that runner extension is not in this release, so here the adapter serves
the structural-recovery judge and any caller that needs the benchmark's laws, boxes and noise).

The trusted runner owns this object, exactly as it owns `adapters/chembench_observation.py`: pass only
schema/observation dictionaries and the observe callback to a learner. This Python interface is not an
OS sandbox. Save predictions during discovery; request reference values only after finish().

Law code. The noise-free law of a task is NewtonBench's own function
`_ground_truth_law_<difficulty>_<version>` in `modules/<domain>/laws.py`, read from the checkout
`BMD_NEWTONBENCH` names (or, without it, from the copy LLM-AutoSciLab vendors,
`autoscilab.oracle.newtonbench._DEFAULT_NB_PATH`). Every file read here is byte-identical to HKUST-KnowComp/NewtonBench at
912a4ba5f4356ddd06acc16e44460ca30be4abc2 (sha256 pinned in `NEWTONBENCH_FILES`; a different file is
refused). Only `laws.py` - and, for m4_snell_law, `physics.py` for `wrap_angle` - is loaded, by file
path: the module packages are never imported, because their `__init__` imports `core.py`, which imports
`modules.common.evaluation`, which imports `utils/call_llm_api.py`, and that file runs `load_dotenv()` at
import time (utils/call_llm_api.py:11).

Observation. The vanilla branch of the module's `run_experiment_for_module` is reproduced exactly
(`VANILLA`: argument order, m2's abs() of every input, m4's wrap_angle), and so is NewtonBench's noise,
`utils/noise.py` `inject_noise` scalar branch: y = law(x) + N(0, sigma) with
sigma = max(|law(x) * noise_level|, floor), the floor being the module's own ABSOLUTE_*_PRECISION constant
in that branch. NewtonBench draws the normal from Python's global `random` (noise.py:31); here it comes
from a generator seeded by the SHA-256 of the task and every input at full precision - the per-design
determinism AutoSciLab's ChemBench and GRN oracles use (`_deterministic_noise`), so the noise is
reproducible and repeats are not independent draws.

Input box (the manifest row's `input_box`):
  autoscilab_registry       `autoscilab.oracle.newtonbench.DOMAIN_REGISTRY[domain]["bounds"]`, every
                            input log-uniform - AutoSciLab's own oracle box and the sampler its
                            `evaluate_law` uses (newtonbench.py:300-309: default_rng(42), 1,000 draws per
                            input, one input after another in bounds order), so `sample_designs(1000, 42)`
                            reproduces those evaluation points.
  newtonbench_evaluate_law  NewtonBench's own `evaluate_law` sampling ranges (`NEWTONBENCH_EVAL_BOX`,
                            modules/<domain>/core.py), log-uniform or uniform as there; a seeded draw of
                            the distribution NewtonBench samples unseeded.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BENCHMARK = "newtonbench"
SYSTEM = "vanilla_equation"
INPUT_BOXES = ("autoscilab_registry", "newtonbench_evaluate_law")
#: HKUST-KnowComp/NewtonBench commit the vendored law files are byte-identical to.
NEWTONBENCH_COMMIT = "912a4ba5f4356ddd06acc16e44460ca30be4abc2"
#: sha256 of each file `_load` may execute, measured on the AutoSciLab-vendored copy (LLM-AutoSciLab
#: 9a37255) and on NewtonBench 912a4ba: equal for every file listed. Of the 78 vendored files 76 are
#: byte-identical to NewtonBench; the two that differ (modules/common/prompts_base.py and
#: utils/call_llm_api.py) are never read here.
NEWTONBENCH_FILES = {
    "m0_gravity/laws.py": "ada4e9cecae8e3420d1bb0760bf71a6de36ec0271fa01ad398847a1ee056be0b",
    "m1_coulomb_force/laws.py": "13fc65447751d5d44ed56bac1741b7a4f13e2b4f5a5f2501d324b18cb3ac0ab3",
    "m2_magnetic_force/laws.py": "d8548e77466013aee8fb2d3ed7edf86d5fb50b1747de19ffbfb99af83f1f5f97",
    "m3_fourier_law/laws.py": "27ae2e06874b9f5661d24a3041542777d1bb382e4abb3a54189b59937676a55a",
    "m4_snell_law/laws.py": "1faecfa85838eb9bcb781a71271da7e0ceb923b249584eac640787059709268f",
    "m4_snell_law/physics.py": "27b144d2f12c442cdc233691e38e22991d5b6292662017a8e014687a845c299f",
    "m5_radioactive_decay/laws.py": "5a8f2842dde390d2d8e7cbcd24bb5ea3da95ec37a4289f153e45057afda33e9c",
    "m6_underdamped_harmonic/laws.py": "02a02873e250524fa0d611bdaf5ea0915b5517b7a0d7dd5eec5001851206011f",
    "m7_malus_law/laws.py": "dc8a2d24ce1ebc17efe52b6d082216c06b165200738a3887abdbef289edc8224",
    "m8_sound_speed/laws.py": "f1644d1e90d922708cdc6d21cc25689334e0d57615a94e4a20a0a26aa5a74b33",
    "m9_hooke_law/laws.py": "017c150015aac67455096ef502a3bd423e8d79914858bcb81161a2241cc4801c",
    "m10_be_distribution/laws.py": "c6d065d3bde4aa89cb981a245f45126a9f2dee160d8881f8fca14d209c46c338",
    "m11_heat_transfer/laws.py": "4525f153163e0730faba2bac0257d87bea63f21f7fce22d65ef73953f69e994c",
}

#: The vanilla branch of each module's run_experiment_for_module (NewtonBench 912a4ba):
#: domain -> (observable name, floor constant name, floor value, the floor's file:line, the branch's
#: core.py lines, the law-argument preprocessing).
VANILLA = {
    "m0_gravity": ("force", "ABSOLUTE_FORCE_PRECISION", 1e-12, "modules/common/types.py:12",
                   "modules/m0_gravity/core.py:247-249", None),
    "m1_coulomb_force": ("force", "ABSOLUTE_FORCE_PRECISION", 1e-12, "modules/common/types.py:12",
                         "modules/m1_coulomb_force/core.py:184-192", None),
    "m2_magnetic_force": ("force", "ABSOLUTE_FORCE_PRECISION", 1e-12, "modules/common/types.py:12",
                          "modules/m2_magnetic_force/core.py:154-159", "abs"),
    "m3_fourier_law": ("power", "ABSOLUTE_POWER_PRECISION", 1e-9, "modules/m3_fourier_law/m3_types.py:15",
                       "modules/m3_fourier_law/core.py:117-127", None),
    "m4_snell_law": ("refraction_angle_degrees", "ABSOLUTE_ANGLE_PRECISION", 1e-6,
                     "modules/m4_snell_law/m4_types.py:4", "modules/m4_snell_law/core.py:100-111",
                     "wrap_angle"),
    "m5_radioactive_decay": ("activity", "ABSOLUTE_ACTIVITY_PRECISION", 1e-9,
                             "modules/m5_radioactive_decay/m5_types.py:14",
                             "modules/m5_radioactive_decay/core.py:135-147", None),
    "m6_underdamped_harmonic": ("angular_frequency", "ABSOLUTE_ANGULAR_VELOCITY_PRECISION", 1e-9,
                                "modules/m6_underdamped_harmonic/m6_types.py:4",
                                "modules/m6_underdamped_harmonic/core.py:76-81", None),
    "m7_malus_law": ("intensity", "ABSOLUTE_INTENSITY_PRECISION", 1e-9, "modules/m7_malus_law/m7_types.py:10",
                     "modules/m7_malus_law/core.py:146-155", None),
    "m8_sound_speed": ("speed", "ABSOLUTE_VELOCITY_PRECISION", 1e-9, "modules/common/types.py:11",
                       "modules/m8_sound_speed/core.py:80-86", None),
    "m9_hooke_law": ("energy", "ABSOLUTE_ENERGY_PRECISION", 1e-9, "modules/m9_hooke_law/m9_types.py:16",
                     "modules/m9_hooke_law/core.py:147-158", None),
    "m10_be_distribution": ("occupation_number", "ABSOLUTE_OCCUPATION_NUMBER_PRECISION", 1e-18,
                            "modules/m10_be_distribution/m10_types.py:6",
                            "modules/m10_be_distribution/core.py:70-75", None),
    "m11_heat_transfer": ("heat_transfer", "ABSOLUTE_HEAT_TRANSFER_PRECISION", 1e-9,
                          "modules/m11_heat_transfer/m11_types.py:18",
                          "modules/m11_heat_transfer/core.py:205-215", None),
}

#: NewtonBench's own evaluate_law sampling ranges (modules/<domain>/core.py, 5,000 unseeded draws
#: there), keyed by AutoSciLab's parameter names in law-argument order: (low, high, scale).
NEWTONBENCH_EVAL_BOX = {
    "m0_gravity": {"mass1": (1.0, 1e3, "log"), "mass2": (1.0, 1e3, "log"), "distance": (1.0, 1e1, "log")},
    "m1_coulomb_force": {"q1": (1e-1, 1e1, "log"), "q2": (1e-1, 1e1, "log"), "distance": (1e-1, 1e1, "log")},
    "m2_magnetic_force": {"current1": (1e-3, 1e-1, "log"), "current2": (1e-3, 1e-1, "log"),
                          "distance": (1e-3, 1e-1, "log")},
    "m3_fourier_law": {"k": (1e-1, 1e1, "log"), "A": (1e-4, 1e-2, "log"), "delta_T": (1e1, 1e3, "log"),
                       "d": (1e-2, 1e0, "log")},
    "m4_snell_law": {"refractive_index_1": (1.0, 1.5, "uniform"), "refractive_index_2": (1.0, 1.5, "uniform"),
                     "incidence_angle": (0.0, 90.0, "uniform")},
    "m5_radioactive_decay": {"N0": (1e0, 1e2, "log"), "lambda_constant": (1e-3, 1e-1, "log"),
                             "t": (1e-2, 1e1, "log")},
    "m6_underdamped_harmonic": {"k": (1e2, 1e4, "log"), "m": (1e-1, 1e1, "log"), "b": (1e-2, 1e0, "log")},
    "m7_malus_law": {"I_0": (100.0, 2000.0, "log"), "theta": (1e-6, float(np.pi / 2), "uniform")},
    "m8_sound_speed": {"gamma": (1.3, 1.7, "uniform"), "T": (1e1, 1e3, "log"), "M": (1e-3, 1e-1, "log")},
    "m9_hooke_law": {"x": (1e-3, 1e0, "log")},
    "m10_be_distribution": {"omega": (1e8, 1e10, "log"), "T": (1e1, 1e3, "log")},
    "m11_heat_transfer": {"m": (1e-3, 1e3, "log"), "c": (1e2, 1e4, "log"), "delta_T": (1e1, 1e3, "log")},
}
#: The file:line of each NewtonBench evaluate_law test_data block (recorded beside a box-B task).
NEWTONBENCH_EVAL_BOX_SOURCE = {
    "m0_gravity": "modules/m0_gravity/core.py:302-308", "m1_coulomb_force": "modules/m1_coulomb_force/core.py:254-263",
    "m2_magnetic_force": "modules/m2_magnetic_force/core.py:193-198", "m3_fourier_law": "modules/m3_fourier_law/core.py:186-198",
    "m4_snell_law": "modules/m4_snell_law/core.py:163-168", "m5_radioactive_decay": "modules/m5_radioactive_decay/core.py:205-215",
    "m6_underdamped_harmonic": "modules/m6_underdamped_harmonic/core.py:120-125", "m7_malus_law": "modules/m7_malus_law/core.py:197-205",
    "m8_sound_speed": "modules/m8_sound_speed/core.py:127-132", "m9_hooke_law": "modules/m9_hooke_law/core.py:211-217",
    "m10_be_distribution": "modules/m10_be_distribution/core.py:112-116", "m11_heat_transfer": "modules/m11_heat_transfer/core.py:266-276",
}


def _asl_newtonbench():
    from autoscilab.oracle import newtonbench as asl

    return asl


def vendor_root():
    """The NewtonBench checkout the law files are read from.

    `BMD_NEWTONBENCH` names a checkout of HKUST-KnowComp/NewtonBench (scripts/provision.sh
    `--with newtonbench` makes one at the pinned commit); without it, the copy AutoSciLab's own
    oracle vendors (`autoscilab.oracle.newtonbench._DEFAULT_NB_PATH`, the ChemBench extra) is
    used. Either way every file read is checked against `NEWTONBENCH_FILES`.
    """
    import os

    configured = os.environ.get("BMD_NEWTONBENCH")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "modules").is_dir():
            raise ValueError(f"BMD_NEWTONBENCH={configured!r} has no modules/ folder")
        return root
    try:
        return Path(_asl_newtonbench()._DEFAULT_NB_PATH).resolve()
    except ImportError as exc:
        raise ImportError("set BMD_NEWTONBENCH to a NewtonBench checkout at commit "
                          f"{NEWTONBENCH_COMMIT[:7]} (scripts/provision.sh --with newtonbench), or put "
                          "LLM-AutoSciLab on PYTHONPATH for its vendored copy") from exc


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_LOADED = {}


def _load(relative, name):
    """Execute one file of the vendored copy by path (never its package); sha256-checked."""
    path = vendor_root() / "modules" / relative
    digest = _sha256(path)
    if NEWTONBENCH_FILES.get(relative) != digest:
        raise ValueError(f"NewtonBench file {relative} has sha256 {digest}; this adapter runs only the "
                         f"pinned NewtonBench {NEWTONBENCH_COMMIT[:7]} bytes {NEWTONBENCH_FILES.get(relative)}")
    key = (str(path), digest)
    if key not in _LOADED:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOADED[key] = module
    return _LOADED[key]


def law_function(domain, difficulty, law_version):
    laws = _load(f"{domain}/laws.py", f"bmd_newtonbench_laws_{domain}")
    if difficulty not in laws.LAW_REGISTRY or law_version not in laws.LAW_REGISTRY[difficulty]:
        raise ValueError(f"NewtonBench {domain} has no {difficulty} {law_version} law")
    return laws.LAW_REGISTRY[difficulty][law_version]


def task_box(row):
    """(input names in law-argument order, {name: (low, high)}, log-sampled names) of a manifest row."""
    domain, box = row["domain"], row.get("input_box")
    if domain not in VANILLA:
        raise ValueError(f"unknown NewtonBench domain {domain!r}")
    if row.get("system", SYSTEM) != SYSTEM:
        raise ValueError(f"this adapter observes the {SYSTEM} system only")
    if box == "autoscilab_registry":
        cfg = _asl_newtonbench().DOMAIN_REGISTRY[domain]
        names = list(cfg["param_names"])
        if list(cfg["bounds"]) != names:
            raise ValueError(f"AutoSciLab's {domain} bounds are not in parameter order")
        return names, {n: (float(cfg["bounds"][n][0]), float(cfg["bounds"][n][1])) for n in names}, list(names)
    if box == "newtonbench_evaluate_law":
        spec = NEWTONBENCH_EVAL_BOX[domain]
        return (list(spec), {n: (float(lo), float(hi)) for n, (lo, hi, _) in spec.items()},
                [n for n, (_, _, kind) in spec.items() if kind == "log"])
    raise ValueError(f"input_box must be one of {INPUT_BOXES}; got {box!r}")


def _sampler(names, bounds, log_names):
    def sample_designs(count, seed):
        """Native input box, local RNG (no global RNG changes); inputs drawn one after another."""
        if count < 1:
            raise ValueError("count must be positive")
        rng = np.random.default_rng(seed)
        columns = {}
        for name in names:
            lo, hi = bounds[name]
            columns[name] = (np.exp(rng.uniform(np.log(lo), np.log(hi), count)) if name in log_names
                             else rng.uniform(lo, hi, count))
        return [{name: float(values[i]) for name, values in columns.items()} for i in range(count)]

    return sample_designs


class _Law:
    """The vanilla branch's noise-free value and its deterministic NewtonBench noise."""

    def __init__(self, domain, difficulty, law_version, names, noise_level):
        self.domain, self.difficulty, self.law_version, self.names = domain, difficulty, law_version, names
        self.noise_level = float(noise_level)
        self.fn = law_function(domain, difficulty, law_version)
        self.observable, self.floor_name, self.floor = VANILLA[domain][0], VANILLA[domain][1], VANILLA[domain][2]
        self.preprocess = VANILLA[domain][5]
        self.wrap = (_load("m4_snell_law/physics.py", "bmd_newtonbench_physics_m4").wrap_angle
                     if self.preprocess == "wrap_angle" else None)

    def value(self, x):
        args = [float(x[name]) for name in self.names]
        if self.preprocess == "abs":
            args = [abs(v) for v in args]
        elif self.preprocess == "wrap_angle":
            args[2] = self.wrap(args[2])
        with np.errstate(all="ignore"):
            return float(self.fn(*args))

    def noise(self, x, true):
        """inject_noise's scalar branch with a per-design seeded generator (module doc)."""
        if self.noise_level == 0.0:
            return 0.0
        sigma = max(abs(true * self.noise_level), self.floor)
        text = "|".join([BENCHMARK, self.domain, self.difficulty, self.law_version, SYSTEM,
                         *(f"{name}={float(x[name])!r}" for name in self.names)])
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        return float(np.random.default_rng(seed).normal(0.0, sigma))


class NewtonBenchSession:
    """Budgeted observations plus explicitly closed-run reference export (ChemBenchSession's contract)."""

    def __init__(self, row, domain, difficulty, law_version, *, noise_level, budget):
        if (domain, difficulty, law_version) != (row["domain"], row["difficulty"], row["law_version"]):
            raise ValueError("session arguments differ from the bound manifest row")
        if budget < 1 or not np.isfinite(noise_level) or noise_level < 0:
            raise ValueError("require positive budget and finite nonnegative noise")
        self._names, self._bounds, self._log = task_box(row)
        self._law = _Law(domain, difficulty, law_version, self._names, noise_level)
        self._budget, self._calls, self._finished = budget, 0, False

    @property
    def calls(self) -> int:
        return self._calls

    def schema(self) -> dict:
        return {
            "benchmark": BENCHMARK, "target": self._law.observable,
            "input_names": list(self._names),
            "input_bounds": {k: list(v) for k, v in self._bounds.items()},
            "log_sampled_inputs": sorted(self._log),
            "observable_names": [self._law.observable],
        }

    def observe(self, design: dict[str, float]) -> dict:
        if self._finished:
            raise RuntimeError("discovery is finished")
        if self._calls >= self._budget:
            raise RuntimeError("observation budget exhausted")
        if set(design) != set(self._names):
            raise ValueError("provide exactly the declared input variables")
        x = {k: float(design[k]) for k in self._names}
        for name, (lo, hi) in self._bounds.items():
            if not np.isfinite(x[name]) or not lo <= x[name] <= hi:
                raise ValueError(f"input outside the task's box: {name}")
        self._calls += 1  # a failed call also consumes the attempted call
        true = self._law.value(x)
        y = true + self._law.noise(x, true)
        if not np.isfinite(y):
            raise ValueError("NewtonBench law returned a nonfinite observation")
        if y <= 0:
            raise ValueError("NewtonBench observation is not positive; the positive-outcome likelihood "
                             "cannot hold it (the task's validity check rules this out)")
        return {"observation_index": self._calls, "x": x, "y": float(y),
                "outputs": {self._law.observable: float(y)}}

    def finish(self) -> None:
        self._finished = True

    def reference_values(self, designs: list[dict]) -> dict[str, np.ndarray]:
        """Evaluator-only targets, after discovery; never candidate selection.

        primary_rate and noiseless_observation are both the noise-free law (NewtonBench's vanilla
        system has no secondary factor); heldout_observation adds the same per-design noise observe()
        would have drawn.
        """
        if not self._finished:
            raise RuntimeError("reference values are unavailable during discovery")
        xs = [{k: float(x[k]) for k in self._names} for x in designs]
        primary = np.asarray([self._law.value(x) for x in xs], dtype=float)
        observed = np.asarray([t + self._law.noise(x, t) for x, t in zip(xs, primary, strict=True)])
        if not np.isfinite([primary, observed]).all():
            raise ValueError("nonfinite reference values")
        return {"primary_rate": primary, "noiseless_observation": primary.copy(),
                "heldout_observation": observed}


def bind(row):
    """The runner's adapter for one manifest row: `sample_designs`, `Session`, records."""
    if row.get("benchmark") != BENCHMARK:
        raise ValueError(f"manifest row {row.get('id')!r} is not a {BENCHMARK} task")
    names, bounds, log_names = task_box(row)
    law = _Law(row["domain"], row["difficulty"], row["law_version"], names, 0.0)

    def session(domain, difficulty, law_version, *, noise_level, budget):
        return NewtonBenchSession(row, domain, difficulty, law_version, noise_level=noise_level,
                                  budget=budget)

    vendor = vendor_root()
    files = [f"{row['domain']}/laws.py", *(["m4_snell_law/physics.py"] if law.wrap else [])]
    sampler_name = f"adapters.newtonbench_observation.bind({row['id']}).sample_designs"
    return SimpleNamespace(
        __file__=__file__, benchmark=BENCHMARK, task=dict(row),
        sample_designs=_sampler(names, bounds, log_names), Session=session,
        sampler_name=sampler_name,
        search_interface=(list(names), {n: tuple(b) for n, b in bounds.items()}, set(log_names)),
        provenance={
            "benchmark": BENCHMARK, "system": SYSTEM, "domain": row["domain"],
            "difficulty": row["difficulty"], "law_version": row["law_version"],
            "law_function": f"_ground_truth_law_{row['difficulty']}_{row['law_version']}",
            "law_docstring": (law.fn.__doc__ or "").strip(),
            "input_box": row["input_box"],
            "input_box_source": ("autoscilab/oracle/newtonbench.py DOMAIN_REGISTRY bounds (log-uniform)"
                                 if row["input_box"] == "autoscilab_registry"
                                 else f"NewtonBench {NEWTONBENCH_EVAL_BOX_SOURCE[row['domain']]} "
                                      "evaluate_law ranges"),
            "input_names": names, "input_bounds": {n: list(b) for n, b in bounds.items()},
            "log_sampled_inputs": sorted(log_names),
            "observable": law.observable,
            "noise_rule": ("y = law(x) + Normal(0, sigma), sigma = max(|law(x) * noise|, floor) "
                           "(NewtonBench utils/noise.py inject_noise, scalar branch); the normal is drawn "
                           "from default_rng(first 8 bytes of sha256('newtonbench|domain|difficulty|"
                           "law_version|vanilla_equation|name=repr(value)|...'), little-endian) instead "
                           "of Python's global random (noise.py:31)"),
            "noise_floor": {"name": law.floor_name, "value": law.floor,
                            "source": f"NewtonBench {VANILLA[row['domain']][3]}"},
            "vanilla_branch": f"NewtonBench {VANILLA[row['domain']][4]}",
            "preprocessing": law.preprocess,
            "law_files": {name: {"path": str(vendor / "modules" / name), "sha256": NEWTONBENCH_FILES[name]}
                          for name in files},
            "law_files_identical_to": f"HKUST-KnowComp/NewtonBench {NEWTONBENCH_COMMIT}",
            "vendor_root": str(vendor),
            "package_imported": False,
        },
    )

