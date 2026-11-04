"""The observation adapters under their development-tree name `adapters.*`, for the baseline wrappers and their tests.

The public package keeps them in `bayes_model_discovery.adapters`; the NewtonBench adapter is present only after the
NewtonBench pull request is merged.
"""
from bayes_model_discovery.adapters import chembench_observation  # noqa: F401

try:
    from bayes_model_discovery.adapters import newtonbench_observation  # noqa: F401
except ImportError:  # the NewtonBench adapter is not in this tree yet
    pass
