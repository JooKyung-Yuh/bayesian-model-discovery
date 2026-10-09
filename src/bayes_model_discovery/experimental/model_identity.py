"""Version boundary for model keys and the RNG streams those keys select."""

from collections.abc import Mapping

MODEL_ID_SCHEMA = "ast_roles_family_ac_v2"
LEGACY_MODEL_ID_SCHEMA = "ast_roles_family_ordered_legacy"
# The power-notation rule `MODEL_ID_SCHEMA` applies (count_models.canonical_model_id),
# declared as text and recorded beside the schema wherever the schema is written.
NOTATION_NORMAL_FORM = (
    "pow_literal_nonneg_int: square(x)|power(x,k)|x**k -> POW(x,int(k)); "
    "k literal finite integer-valued >= 0 (2 == 2.0); "
    "x*x, non-integer, negative, non-literal, Sub/Div/assoc/distrib untouched"
)


def require_model_id_schema(record):
    """Refuse legacy/unknown records, including records with no schema field.

    A record of an earlier schema (`ast_roles_family_ac_v1`) is refused too: its
    ids were computed without the notation normal form.

    Pool reading belongs to the pool reader, which must call this before any output
    or reservation; this helper neither reads a file nor creates an artifact.
    """
    actual = record.get("model_id_schema", LEGACY_MODEL_ID_SCHEMA) if isinstance(record, Mapping) else None
    if actual != MODEL_ID_SCHEMA:
        raise ValueError(f"model_id_schema mismatch: expected {MODEL_ID_SCHEMA!r}, got {actual!r}")
    return MODEL_ID_SCHEMA
