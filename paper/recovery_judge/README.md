# Structural-recovery judge

The deterministic program that decides, for one proposed law and one benchmark task, whether the law's
family is equivalent to the true family (strict recovery) or contains it (containment). It is the judge
behind the structural-recovery results of the paper (rule version `c5-judge-1.3`; `judge/core.py`
`RULE_VERSION`). No language model is used anywhere.

- `judge/core.py`: the rule. Both expressions are reduced to a canonical form; strict recovery needs the same
  form and a coefficient mapping solved in both directions, agreement to 30 digits at random points, and
  admissible coefficients that reproduce the true setting and two further settings; a law is non-equivalent
  when its input dependence, number of identifiable coefficients or coefficient domain differs, or when an
  eight-start refit misses the target; containment tries up to 60 substitutions (coefficients to 0, 1, a limit,
  or tied together) and falls back to a numerical fit on separate inputs. Cases it cannot settle stay
  `unresolved`.
- `judge/chembench.py`, `judge/boxinggym.py`, `judge/forcebench.py`, `judge/newtonbench.py`: the declared target
  family of each task (written as sympy expressions in this repository and checked numerically against the
  benchmark's own generating code where it is available), the comparison scale, the representability of each
  task under the proposal grammar, and the candidate parser for that benchmark's run records.
- `judge/metrics.py`: the per-round bookkeeping of one run (top-weighted member, admitted / top-1 / weight
  on passing members under the strict, containment and membership rules) and the run summary.
- `tests/`: synthetic probes of every module (reparameterizations, renamed coefficients, superfamilies that
  must count as containment only, extra terms, unresolved and non-representable cases, the surrogate rule,
  the bookkeeping).

## Running

```bash
pip install -c requirements.lock -e ".[dev,judge]"        # adds sympy 1.14.0 and mpmath 1.4.1
cd paper/recovery_judge
PYTHONHASHSEED=0 PYTHONPATH=../../src pytest -q tests      # 48 tests
```

`PYTHONHASHSEED=0` fixes sympy's set iteration order, which the rule's canonical forms depend on; the tests
refuse to run without it. The ChemBench tests need `BMD_AUTOSCILAB` (the LLM-AutoSciLab checkout, whose
oracle holds the true rate laws) and the NewtonBench tests need `BMD_NEWTONBENCH` (a NewtonBench checkout at
the pinned commit); both come from `scripts/provision.sh --with chembench,newtonbench`. ForceBench and
BoxingGym need nothing external: their target families are declared in the modules.

What is not here: the drivers that walk a campaign's run records, judge every pool member of every round and
write the recovery tables and curves. They read the development machine's record layout and stay with the
development repository; the per-round and final-round results they produced are in `paper/data/`.
