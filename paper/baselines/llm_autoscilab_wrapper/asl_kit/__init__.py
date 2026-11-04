"""Interface overlay that runs the released LLM-AutoSciLab (Kabra et al.) as an external baseline.

Nothing in the AutoSciLab checkout is edited. The overlay imports the checkout and changes
only interfaces: the LLM client path (Claude), the oracles (ChemBench native + leak block,
BoxingGym and ForceBench adapters), the initial data (our D0) and an evaluation hook that
records the loop's current law after every experiment and scores it offline.
"""
