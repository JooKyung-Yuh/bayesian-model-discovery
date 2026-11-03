# Prompt texts of the paper runs that use the belief-packet feedback variants

These files are the system prompts and paragraphs sent by the runs behind the paper's main
learning curves on ChemBench, BoxingGym and NewtonBench, the ForceBench runs with the full
interface, the feedback study (Full, Belief-only, Diagnostics-only, No feedback) and the
design-criterion study on NewtonBench and ForceBench. They are kept byte for byte: every run
records the sha256 of the text it sent, and `SHA256SUMS` beside them lists the digests. The
test `tests/unit/test_paper_prompts.py` checks the files against those digests.

The earlier prompt `v2.9.FA-C` (`tools/fb1_launch_c/proposer_system_v2_9_FA-C.txt`, digest
`196f82b7…`) stays where it is. It is the prompt of the first round of every run (the initial
batch of twelve proposals), of the ChemBench and BoxingGym design-criterion runs, of the 24
further ChemBench tasks, and of the ForceBench runs with the masked interface (its `_trig`
file, digest `4dac3b09…`).

| File | Digest (first 8) | Sent by |
| --- | --- | --- |
| `proposer_system_v2_9_FA-C-R.txt` | `e465265a` | every proposal round after the first of the main ChemBench, BoxingGym and NewtonBench runs (the Full feedback variant) and of the NewtonBench design-criterion runs |
| `proposer_system_v2_9_FA-C-B.txt` | `0a922682` | the Belief-only variant on ChemBench, BoxingGym and NewtonBench |
| `proposer_system_v2_9_FA-C-D.txt` | `37ab98c7` | the Diagnostics-only variant on the same benchmarks |
| `proposer_system_v2_9_FA-C-NF.txt` | `c53ba372` | the No-feedback variant on the same benchmarks |
| `proposer_system_v2_9_FA-C-R_trig.txt` | `445949c6` | the same as `FA-C-R` with sine and cosine in the operator list: every proposal round after the first of the ForceBench runs with the full interface (main curves, design criteria, Full feedback) |
| `proposer_system_v2_9_FA-C-B_trig.txt`, `…-D_trig.txt`, `…-NF_trig.txt` | `a846c5b4`, `b647a679`, `098ad442` | the three ablation variants on ForceBench with the full interface |
| `composition_paragraph_v3_neutral.txt` | `8f41727f` | the three-candidate paragraph appended to the system prompt in those rounds (replaces `COMPOSITION_PARAGRAPH_V2`, `782a30b7…`, of the first-round and `FA-C` runs); `{limit}` is filled with 3 |
| `archive_review_paragraph_v2_neutral.txt` | `1711c11a` | the archive-review paragraph of every fifth round in those runs (replaces `ARCHIVE_REVIEW_PARAGRAPH` v1, `8b4c5359…`) |
| `FB_FULL_DESCRIPTION.md` | `d6660770` | the task description that opens every system text of the ForceBench runs with the full interface (the paper's ForceBench setting); the masked-interface runs sent no description and named the inputs `X2`, `X3` |

The four `FA-C-*` prompts differ from `FA-C` in one paragraph: the description of the belief
packet's fields. `FA-C-R` describes the packet of the Full variant, whose residual block covers
every active model; `FA-C-B` describes a packet without the residual block; `FA-C-D` one without
evidence, weights, coefficient posteriors and the best-model marker; `FA-C-NF` one with neither,
and its candidate and diagnostic instructions no longer refer to a current best model. The
packet builder that masks those fields, replays the shared first round and binds each variant to
its prompt is part of the development tree used for the paper runs and is not in this repository;
here the files document what the LLM read. Any of them can be sent as a development prompt with
`--prompt-variant <file>` (the first line declares the version tag), which the runners refuse for
production-labelled runs.
