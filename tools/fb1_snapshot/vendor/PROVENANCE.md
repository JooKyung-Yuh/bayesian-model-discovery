# vendor/: where the vendored grammar engine comes from

## count_models_v2_9.py

`count_models_v2_9.py` is a byte-for-byte copy (no edits) of
`src/bayes_model_discovery/experimental/count_models.py` as it stood on the engine's v2.9 chain, at the point the
OpenAI transport was added. Its sha256 is `38c5afe813f89342b50d3a7482a3129ce07ab90c266866100d849ebfebdf966b`
(`fb1_grammar.py` holds the same value). It was copied rather than imported because the ForceBench environment
(jax, numpy 2.x) cannot install the engine package (numpy 1.26.4), and rewriting the grammar would let it drift from
the engine's. It needs only the standard library (`ast`, `hashlib`, `json`, `math`) plus `numpy` and `scipy`
(`scipy.special.expit`, `scipy.stats`); no pymc, no sympy.

`fb1_grammar.py` uses four of its names: `validate_expression` (the grammar check: allowed tokens, roles,
coefficient count, operator bound), `canonical_model_id` (sha256 of the canonical AST, roles and family, used for
duplicate detection), `evaluate` (interprets one validated expression; with `backend=jnp` and named inputs it builds
the jax force function) and `CALL_ARITY` (the six allowed calls: `exp`, `log`, `sqrt`, `tanh`, `square`, `power`).
The count-task parts (`particle_logpmf`, `Hypothesis`, `initial_hypotheses`, `judge_candidate`, `FAMILIES`,
`PRIOR_RULES`) are not used; ForceBench's own families live in `fb1_core.FAMILIES` (gauss, student_t4).

To update it, copy the engine file again and change the sha256 here and in `fb1_grammar.py` together; never edit
the copy in place.

The copy keeps its original comments, including development labels, because every ForceBench run records its sha256
(`vendor_sha256`); editing even a comment would break that link to the paper's runs.
