# cvm-missing-variable-detection

A four-stage, white-box framework that extends the Creator Variable
Machine (CVM) concept — originally developed for improving prediction
accuracy via correlated-variable generation (Shishegaran & Varaee et
al., 2021; Shishegaran & Varaee, 2026) — to a different purpose:
detecting *when* a physical model is missing a variable altogether, and
inferring interpretable properties of that missing variable directly
from residual structure.

Given only observed variables, the pipeline (1) searches algebraic
combinations for the strongest Creator Variable, (2–3) fits the
simplest interpretable equation, and (4) classifies the residual as
sufficient, model-insufficient (wrong functional form), or
representation-insufficient (evidence of a missing variable) using
permutation-based mutual information and distribution tests. When a
variable is flagged missing, a Hypothesis Card infers its unit,
direction, interaction partners, scale, and smoothness — without ever
identifying the variable itself.

Validated on two synthetic systems with known ground truth (cantilever
beam deflection, ideal gas law) and on real reinforced-concrete deep
beam shear-strength data, where it independently rediscovers arch
action — confirmed under out-of-sample 5-fold cross-validation — and
surfaces a general selection-instability failure mode in greedy
top-1 candidate selection, addressed here with a bootstrap-based
stability diagnostic.

**Author:** Hesam Varaee

**Built on the Creator Variable Machine concept introduced in:**

Shishegaran, A., Varaee, H., Rabczuk, T., Shishegaran, G. (2021). High
correlated variables creator machine: Prediction of the compressive
strength of concrete. *Computers & Structures*, 247, 106479.
https://doi.org/10.1016/j.compstruc.2021.106479

Shishegaran, A., Varaee, H. (2026). Comparison among creator variable
machine methods: Compressive strength prediction of ultra-high-
performance concrete. *Case Studies in Construction Materials*, 25,
e06253. https://doi.org/10.1016/j.cscm.2026.e06253

**Companion paper (this repository):** [citation to be added upon
publication/preprint]
