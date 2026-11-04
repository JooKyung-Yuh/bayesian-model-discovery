"""ForceBench neutral interface (2026-10-11): what the language model is told in FB1_INTERFACE_NEUTRAL=1.

The engine is unchanged: it fits and simulates every expression exactly as before (the expression in X1..X4 is the force
law, the acceleration is minus the expression over X3 along the unit position vector, velocity Verlet). Only the text the
model reads changes. In neutral mode the model learns that a scalar expression in X1..X4 (the grammar's operators, the
coefficient roles as before) is handed to a fixed simulator that returns a sequence of two-component measurements for
the inputs X2, X3 and a fixed start state; nothing about mechanics.

Pure functions, no engine state, so the packet audit of the development tree applies the same rewrite to the
packet file and the registered texts before it compares them with what a neutral request sent:
  neutral_packet(packet)        a masked belief packet (dict) -> its neutral form
  neutral_packet_text(text)     the same on the engine's serialization (sorted keys, compact JSON); JSON out as JSON in
  neutral_prose(text)           a system text or a repair sentence -> its neutral form
  neutral_message(index, text)  a user message: index 0 (the packet) through neutral_packet_text, a repair sentence
                                through neutral_prose
The rewrite is deterministic and versioned (NEUTRAL_VERSION); the cell records the version. Numbers are never changed:
the start state keeps its numbers under s0a/s0b, the measurement schedule its values under X4_values_at_measurements.
Three blocks are left out instead of reworded: interface.integration (the integrator scheme, its step size dt and its step
count), constraints.grammar.roles (the internal role abbreviations) and constraints.grammar.catalogue_prior_scale_differences
(a catalogue-only note).
The run requires masked input names (the engine's own names r, p1, p2, t would leak the roles).
"""
import json
import re

NEUTRAL_VERSION = "fb_interface_neutral_v1"

DERIVED_X1 = "derived_input_computed_by_the_simulator_from_its_state_at_each_step"
DERIVED_X4 = "derived_input_computed_by_the_simulator_from_the_step_index"
SIMULATOR = ("a_fixed_simulator_receives_the_scalar_expression_in_X1_X2_X3_X4_with_its_coefficients_and_the_inputs_X2_X3_"
             "and_returns_a_sequence_of_two_component_measurements; the_start_state_is_fixed_and_identical_for_every_"
             "input_pair")

#: interface.summary_definitions (and best_model_residual_definitions, residual_definitions): the neutral text of a field,
#: by (group, field name after the key rename); a field not listed keeps its text through PROSE_RULES
DEFINITIONS = {
    ("observations", "X2"): "first_input_handed_to_the_simulator",
    ("observations", "X3"): "second_input_handed_to_the_simulator",
    ("observations", "start_state"): "object_with_s0a_and_s0b_each_a_length_2_vector_fixed_and_identical_for_every_input_pair",
    ("observations", "simulator_steps"): "count_of_internal_simulator_steps_up_to_the_last_measurement",
    ("residuals", "max_abs_term"): ("max_over_measurements_of_the_absolute_value_of_the_expression_over_X3_for_the_best_model_"
                                    "at_its_posterior_median_coefficients_along_its_mean_sequence"),
    ("residuals", "min_X1"): "min_over_measurements_of_X1_on_the_best_models_posterior_predictive_mean_sequence",
    ("residuals", "pred_mean"): "posterior_predictive_mean_of_each_measurement_component",
}
#: groups of definition blocks: interface.summary_definitions.<group>, and the two top-level residual blocks
RESIDUAL_BLOCKS = ("best_model_residual_definitions", "residual_definitions")

#: ordered (pattern, replacement) on every string and key, boundaries = any character that is not a letter or a digit
_B = r"(?<![A-Za-z0-9])"
_E = r"(?![A-Za-z0-9])"
PROSE_RULES = [(re.compile(_B + p + _E), r) for p, r in (
    (r"input times a coefficient", "input multiplied by a coefficient"),
    (r"integration_status", "simulation_status"),
    (r"integration_steps", "simulator_steps"),
    (r"initial_state", "start_state"),
    (r"max_abs_accel", "max_abs_term"),
    (r"accel_abs", "term_abs"),
    (r"residual_trajectory_8pt", "residual_sequence_8pt"),
    (r"time_grid", "X4_values_at_measurements"),
    (r"time grid", "X4 values at the measurements"),
    (r"measurement[_ ]times", "measurements"),
    (r"measurement[_ ]time", "measurement"),
    (r"particle_mean", "sample_mean"),
    (r"trajectories", "measurement sequences"),
    (r"trajectory", "measurement_sequence"),
    (r"force_magnitude", "expression_magnitude"),
    (r"fb1_force_structure", "fb1_expression_structure"),
    (r"free-form force law", "free-form expression"),
    (r"force law", "expression"),
    (r"force magnitude", "expression magnitude"),
    (r"forces", "expressions"),
    (r"force", "expression"),
    (r"integration", "simulation"),
)]
#: keys renamed only inside the probe blocks (constraints.grammar.probe and every archive row's prior_probe)
PROBE_KEYS = {"times": "X4_values", "n_times": "n_X4_values"}
#: keys renamed only inside an observation's start state
START_STATE_KEYS = {"pos": "s0a", "vel": "s0b"}


def neutral_prose(text):
    for pattern, replacement in PROSE_RULES:
        text = pattern.sub(replacement, text)
    return text


def _value(value, parent=None):
    """Every key and string through PROSE_RULES (with the probe and start-state key renames); numbers as they are."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if parent in ("probe", "prior_probe") and key in PROBE_KEYS:
                new = PROBE_KEYS[key]
            elif parent == "start_state" and key in START_STATE_KEYS:
                new = START_STATE_KEYS[key]
            else:
                new = neutral_prose(key) if isinstance(key, str) else key
            out[new] = _value(item, new)
        return out
    if isinstance(value, list):
        return [_value(item, parent) for item in value]
    if isinstance(value, str):
        return neutral_prose(value)
    return value


def _definitions(block, group):
    return {name: DEFINITIONS.get((group, name), text) for name, text in block.items()}


def _interface(interface):
    out = {key: interface[key] for key in ("inputs", "link", "observation_noise") if key in interface}
    out["derived_inputs"] = {"X1": DERIVED_X1, "X4": DERIVED_X4}
    out["simulator"] = SIMULATOR
    out["outcome_type"] = "sequence_of_two_component_measurements"
    out["support"] = "two_component_real_vector_at_each_measurement"
    if "X4_values_at_measurements" in interface:
        out["X4_values_at_measurements"] = interface["X4_values_at_measurements"]
    definitions = interface.get("summary_definitions") or {}
    out["summary_definitions"] = {
        group: _definitions(block, "residuals" if group == "best_model_residuals" else group)
        for group, block in definitions.items()}
    return out


def neutral_packet(packet):
    """A masked belief packet in its neutral form (a new dict; the input is not changed)."""
    if not isinstance(packet, dict):
        return _value(packet)
    out = _value(packet)
    if isinstance(out.get("interface"), dict):
        out["interface"] = _interface(out["interface"])
    for key in RESIDUAL_BLOCKS:
        if isinstance(out.get(key), dict):
            out[key] = _definitions(out[key], "residuals")
    grammar = (out.get("constraints") or {}).get("grammar")
    if isinstance(grammar, dict):
        # the role abbreviation table (positive -> pos, signed -> sig) is internal; constraints.roles names the roles
        grammar.pop("roles", None)
        # a catalogue-only note (catalogue on; the free-mode runs of the trig generation send no catalogue)
        grammar.pop("catalogue_prior_scale_differences", None)
    return out


def neutral_packet_text(text):
    value = json.loads(text)
    return json.dumps(neutral_packet(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def neutral_message(index, text):
    if index == 0:
        try:
            json.loads(text)
        except ValueError:
            return neutral_prose(text)
        return neutral_packet_text(text)
    return neutral_prose(text)
