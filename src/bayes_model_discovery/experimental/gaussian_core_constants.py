"""Frozen Gaussian CORE constants and their declared development design rule.

Ported from the declared selector, with the quantity named as preregistration
section 2.3 requires: a known-sigma development design proxy, not evidence.
Only --select or an explicit function call evaluates the selection procedure.
"""

import argparse

import numpy as np

GAUSS_DEN_C = 0.07
GAUSS_POW_E = 0.40
GAUSS_CALL_K = 5
DESIGN_PROXY_NAME = "known_sigma_development_design_proxy"
NOISE_FRACTION = 0.05
OBSERVATIONS = 54
DESIGN_PROXY_FLOOR = 20.0
GRID = np.linspace(0.0, 1.0, 100)
DENOMINATORS = tuple(round(0.02 + 0.01 * i, 2) for i in range(19))
EXPONENTS = tuple(round(v, 2) for v in np.arange(0.20, 0.901, 0.05)
                  if abs(v - round(v)) > 1e-9 and abs(v - 0.5) > 1e-9)
CALL_SCALES = tuple(range(4, 21))


def standardized(x):
    """Map x from [0, 1] to [-1, 1]."""
    return 2.0 * x - 1.0


def basis(x):
    """Return the design matrix of the fixed basis functions evaluated at x."""
    z = standardized(x)
    return np.column_stack([np.ones_like(x), z, z**2, z**3,
                            np.tanh(z), np.exp(z), np.log1p(x)])


def diagnose(shape):
    """Return (r/sigma, design proxy in nats) for a shape function on the grid."""
    m = shape(GRID)
    sigma = NOISE_FRACTION * np.std(m, ddof=1)
    design = basis(GRID)
    residual = m - design @ np.linalg.lstsq(design, m, rcond=None)[0]
    rms = float(np.sqrt(np.mean(residual**2)))
    ratio = rms / sigma
    return float(ratio), float(OBSERVATIONS * ratio**2 / 2.0)


def selection_details():
    """Return, per regime, the chosen (constant, r/sigma, design proxy) row."""
    regimes = (
        ("gauss_den", DENOMINATORS, lambda c: lambda x: x / (c + x), max),
        ("gauss_pow", EXPONENTS, lambda e: lambda x: x**e, max),
        ("gauss_call", CALL_SCALES, lambda k: lambda x: np.tanh(k * standardized(x)), min),
    )
    chosen = {}
    for name, candidates, shape_of, gentlest in regimes:
        rows = [(value, *diagnose(shape_of(value))) for value in candidates]
        qualifying = [row for row in rows if row[2] >= DESIGN_PROXY_FLOOR]
        if not qualifying:
            raise ValueError(f"no {name} candidate meets the declared design proxy floor")
        chosen[name] = gentlest(qualifying)
    return chosen


def select_constants():
    """Return the selected (denominator, exponent, call scale) constants."""
    return tuple(row[0] for row in selection_details().values())


def main():
    """Command-line entry point: print the selection details."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="store_true", required=True)
    parser.parse_args()
    for name, (value, ratio, proxy) in selection_details().items():
        print(f"{name}: constant={value} r/sigma={ratio:.3f} "
              f"{DESIGN_PROXY_NAME}={proxy:.1f} nats")


if __name__ == "__main__":
    main()
