# Second addendum to PROTOCOL_real_mica.md — free LoRA with MiCA's recipe (committed and pushed before its runs)

**Declared as added after the fact.** This test was decided after the reading of experiment 3 was known (G_MiCA = +3.72, established).
It is pre-registered before its own runs, but not before that result. PROTOCOL_real_mica.md and its first addendum are not edited.

**Question.** In experiment 3, MiCA trained with its recipe (4 epochs, 417,792 supervised tokens) trails the budget-matched free LoRA
trained with ours (600,000 tokens): the comparison is between recipes. Does MiCA still trail when the free LoRA is trained with exactly
MiCA's recipe?

**Arm.** PEFT 0.21.0's standard LoRA (`init_lora_weights=True`, PEFT's default initialisation; A and B both trained), mode `lora_peft`,
rank 7 (budget-matched with MiCA at rank 16, as the free LoRA of experiment 3), α = 7 (α/r = 1, as MiCA's α = 16 at rank 16), on our
seven modules. **Recipe identical to the MiCA arm of experiment 3**, through the same code path: learning rate 5·10⁻⁴, cosine schedule
with warm-up ratio 0.1, 4 epochs, per-device batch 4 with no accumulation, weight decay 0.01, adapter dropout 0.05, gradient clipping at
0.3. Everything else copied from the reported free LoRA at rank 7. Seeds 2 to 6 (5 runs), RTX 4000 Ada. The MiCA runs of experiment 3
are its comparison; they are not rerun.
Note: MiCA's learning rate was tuned for MiCA, not for LoRA; the recipe can only disadvantage the free LoRA.

**Pre-registered test (one).** G_equal = free LoRA with MiCA's recipe − MiCA with MiCA's recipe > 0, paired over seeds 2 to 6, two-sided
t > 2.776 (4 d.f.).
**Reading fixed in advance.** If G_equal is established, the paper may say that MiCA trails budget-matched LoRA at equal recipe and equal
training amount. If not, the paper keeps the sentence of the first addendum (a comparison between recipes) and draws nothing more from it.
**Descriptive:** the free LoRA with MiCA's recipe against the free LoRA with ours (experiment 3); supervised tokens and optimiser steps of
each arm. Compute: about 0.3 PARTITION-hour.
