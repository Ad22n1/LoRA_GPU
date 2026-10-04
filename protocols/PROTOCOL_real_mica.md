# Experiment 3 — the real MiCA method at rank 16 (TMLR)

Written, committed and pushed before any run. Reuses no earlier run.

**Question.** At rank 16 on OpenBookQA with Qwen2.5-1.5B, our MiCA-like arm (`bottom`) trails the budget-matched free LoRA. Does MiCA, as
released in PEFT 0.21.0 and trained with the recipe of the MiCA paper, trail it too?

**MiCA arm.** PEFT 0.21.0, `init_lora_weights="mica"` (B on the r smallest left singular vectors, A = 0, B frozen), rank 16, α = 16
(α/r = 1). **Recipe of the MiCA paper** (Rüdiger and Raschka, arXiv:2604.01694), its only rank-16 configuration (Table 1, Llama-2-7B,
BLOGS) and its stated training settings (Appendix A): learning rate 5·10⁻⁴; cosine schedule with warm-up ratio 0.1; 4 epochs; weight decay
0.01; adapter dropout 0.05; gradient clipping at 0.3; per-device batch 4 with no accumulation. **Not stated by the MiCA paper, ours, and
declared:** the data, the evaluation and the checker (OpenBookQA as in our campaigns), the optimiser family (AdamW), precision (as ours).
**Declared choice:** the MiCA paper adapts `q_proj` and `v_proj` only; we adapt our seven modules, so that MiCA and our arm differ only by
their recipe, and so that the budget-matched free LoRA stays matched. PEFT is loaded from a separate directory, for this arm only; no other
run's environment changes.

**Same campaign, rerun:** our `bottom` arm at rank 16 at its selected rate, and the budget-matched free LoRA at rank 7 at its selected rate
(both found by their published scores, written into the check file before the first run). Seeds 2 to 6 for all three (15 runs), RTX 4000
Ada.

**Pre-registered test (one).** G_MiCA = free LoRA (r = 7) − MiCA (r = 16) > 0, paired over seeds 2 to 6, two-sided t > 2.776 (4 d.f.).
**Descriptive:** MiCA − our `bottom` at rank 16, with its interval.
**Reading fixed in advance.** If G_MiCA is established, the conclusion that the MiCA-like arm trails budget-matched LoRA at rank 16 extends
to MiCA's own recipe. If not, the paper must say that MiCA's recipe closes the gap there. Compute: about 3 PARTITION-hours.
