"""Which sub-expression made a candidate non-finite under the prior probe.

The prior probe (core_count.prior_probe, boxinggym_core.prior_probe,
chembench_core._prior_probe) rejects a candidate whose mean expression or
likelihood is non-finite anywhere on the interface's reference grid under its
own prior. Earlier versions of the rejection said only "non-finite under prior probe".
This module evaluates the expression tree one node at a time, with the same
prior particles and the same grid the probe used, and names the innermost
sub-expression whose value is non-finite while every operand of it is finite,
together with the input name and value at the first grid point where that
happens and the input's declared range. The sentence it builds is an engine
sentence (general wording, the candidate's own sub-expression, a number and a
range); it never quotes a reply or an example.

The walker mirrors count_models.evaluate node for node (the grammar it accepts
is exactly what validate_expression admits), with NumPy as the only backend and
under numpy.errstate(all="ignore") so that an overflow or an invalid operation
becomes inf or nan instead of a warning.
"""

from __future__ import annotations

import ast
import math

import numpy as np

from .count_models import CALL_ARITY

PROBE_SENTENCE_PREFIX = "Tried the prior probe; term "
PRIOR_PROBE_MESSAGE = "non-finite under prior probe"


class PriorProbeRejection(ValueError):
    """A candidate refused by the prior probe inside the proposer's admission hook.

    A ValueError, so the proposer's one repair turn (count_proposer.propose_one:
    the same conversation continued once with the verbatim message) applies to it
    exactly as to a grammar or contract failure; `repair_kind` names it
    "prior_probe" in the transport record.
    """


def prior_probe_reason(probe):
    """The archive reason of a prior-probe rejection: the fixed message, plus the
    detail sentence when the probe could name the non-finite sub-expression."""
    reason = {"validation_message": PRIOR_PROBE_MESSAGE}
    if probe.get("nonfinite") is not None:
        reason["detail"] = probe["detail"]
    return reason
PROBE_SENTENCE_GUIDANCE = (
    "standardized inputs can be negative; use a function defined on the declared range, "
    "or apply it to a raw or [0, 1]-scaled input whose declared range is non-negative."
)


def _finite(value):
    return bool(np.all(np.isfinite(np.asarray(value, dtype=float))))


def _input_names_in(node, input_names):
    """Input names read by `node`, in order of first appearance."""
    # A categorical input only indexes a coefficient (`a[level]`): it is a label,
    # not a value the term is evaluated at, so it is never the named input.
    labels = {item.slice.id for item in ast.walk(node) if isinstance(item, ast.Subscript)}
    seen = []
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and item.id in input_names and item.id not in labels and item.id not in seen:
            seen.append(item.id)
    return seen


def first_nonfinite_subexpression(expression, parameters, inputs, interface):
    """The innermost non-finite node of `expression`, or None when every node is finite.

    `parameters` maps each coefficient to a prior sample (shape [particle] or
    [particle, level]); `inputs` maps each declared and derived input to its
    values on the probe grid (shape [point]). Particles broadcast along the first
    axis and grid points along the last, exactly as the probe evaluates them.

    Returns a dict with `term` (the sub-expression's source text), `input_name`
    (the first input the term reads, or None), `input_value` (that input at the
    first offending grid point), `range` ([low, high]: the declared range of a
    base input, or the range a derived input takes on the probe grid), and
    `range_kind` ("declared" or "derived_on_grid").
    """
    names = list(inputs)
    values = {}
    for name, sample in parameters.items():
        sample = np.asarray(sample, dtype=float)
        values[name] = sample.reshape(sample.shape + (1,)) if sample.ndim else sample
    for name in names:
        values[name] = np.asarray(inputs[name], dtype=float).reshape(1, -1)
    tree = ast.parse(expression, mode="eval")
    found = {}

    def walk(node):
        """Depth-first search for the first non-finite subexpression."""
        if found:
            return None
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Subscript):
            # An indexed coefficient a[level]: the probe gathers its levels; here
            # every level takes part, so a non-finite level is found as well.
            base = values[node.value.id]
            return base.reshape(base.shape[0], -1)
        if isinstance(node, ast.BinOp):
            operands = [walk(node.left), walk(node.right)]
        elif isinstance(node, ast.UnaryOp):
            operands = [walk(node.operand)]
        elif isinstance(node, ast.Call) and node.func.id in CALL_ARITY:
            operands = [walk(arg) for arg in node.args]
        else:
            raise ValueError("node outside the expression grammar")
        if found or any(operand is None for operand in operands):
            return None
        with np.errstate(all="ignore"):
            if isinstance(node, ast.BinOp):
                a, b = operands
                if isinstance(node.op, ast.Add):
                    value = a + b
                elif isinstance(node.op, ast.Sub):
                    value = a - b
                elif isinstance(node.op, ast.Mult):
                    value = a * b
                elif isinstance(node.op, ast.Div):
                    value = a / b
                else:
                    value = np.power(a, b)
            elif isinstance(node, ast.UnaryOp):
                value = -operands[0] if isinstance(node.op, ast.USub) else operands[0]
            elif node.func.id == "square":
                value = np.square(operands[0])
            elif node.func.id == "power":
                value = np.power(operands[0], operands[1])
            else:
                value = getattr(np, node.func.id)(*operands)
        if not _finite(value) and all(_finite(operand) for operand in operands):
            grid = np.asarray(value, dtype=float).reshape(-1, np.asarray(value).shape[-1])
            column = int(np.argmax(~np.isfinite(grid).all(axis=0)))
            read = _input_names_in(node, names)
            found.update(
                term=ast.get_source_segment(expression, node) or ast.unparse(node),
                input_name=read[0] if read else None,
                input_value=float(values[read[0]][0, column]) if read else None,
                grid_index=column,
            )
            return None
        return value

    walk(tree)
    if not found:
        return None
    name = found["input_name"]
    if name is None:
        found.update(range=None, range_kind=None)
    else:
        declared = {item["name"]: item for item in interface.get("inputs", ())}
        if name in declared:
            low, high = declared[name]["range"]
            found.update(range=[float(low), float(high)], range_kind="declared")
        else:
            column = values[name].reshape(-1)
            found.update(range=[float(column.min()), float(column.max())], range_kind="derived_on_grid")
    return found


def _number(value):
    if value is None or not math.isfinite(value):
        return str(value)
    return f"{value:.4g}"


def nonfinite_probe_sentence(found):
    """The engine's rejection sentence for one `first_nonfinite_subexpression` result."""
    where = ""
    if found.get("input_name") is not None:
        low, high = found["range"]
        kind = "declared range" if found["range_kind"] == "declared" else "range on the declared inputs"
        where = (
            f" at {found['input_name']}={_number(found['input_value'])}"
            f" ({kind} [{_number(low)}, {_number(high)}])"
        )
    return f"{PROBE_SENTENCE_PREFIX}{found['term']} is non-finite{where}; {PROBE_SENTENCE_GUIDANCE}"
