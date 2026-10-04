# Addendum to PROTOCOL_real_mica.md — committed and pushed before the first run of experiment 3

PROTOCOL_real_mica.md is not edited; this addendum refines its reading before any run.

The MiCA recipe trains for 4 epochs at a per-device batch of 4, while our bottom arm and the free LoRA train on 600,000 supervised
tokens, about 5.76 epochs on OpenBookQA. The two sides therefore differ in training amount as well as in recipe. The comparison is
between recipes, not at equal training.

Reading, refined before any run: if G_MiCA is established, MiCA trained with its own recipe trails budget-matched LoRA trained with
ours; the paper will say so in these terms, and will not claim that the cost holds at equal training. If not, the paper must say that
MiCA's recipe closes the gap there.

Reported in addition: the number of supervised tokens and of optimiser steps of each arm.
