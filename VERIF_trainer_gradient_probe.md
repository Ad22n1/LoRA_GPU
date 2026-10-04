# Read-only checks of the training loop and of the gradient probe on W0 (30/09)

No code and no result was changed by these checks.

## 1. Gradient accumulation in the training loop
- **Where.** `src/lorasub/train.py`, function `_run`. The micro-batch counter `micro` is set once, before the loop over epochs, and never
  reset; a step fires when `micro % grad_accum == 0`, so every optimiser step averages exactly 16 micro-batches (`loss / grad_accum`), and a
  window spans an epoch boundary when an epoch is not a whole number of steps (8.5 on OpenBookQA, 99.3 on the format task). Gradients are
  zeroed after every step: clipping, `opt.step()`, `sched.step()`, `opt.zero_grad(set_to_none=True)`. Training stops right after the step
  that reaches its total, so no partial window remains. The supervised-token counter is incremented **at every forward pass**.
- **Provenance after 16/09.** `train.py` entered the repository on 16/09 at 10:37 (`<commit>`). Five later commits touch `_run` — `<commit>`
  (22/09), `<commit>` (26/09), `<commit>` (27/09), `<commit>` (28/09), `<commit>` (29/09) — for 27 changed lines: the answer-only dataset
  wrapper, the learned-basis argument, Adam's `eps` and the parameter groups of the `top_lrscale` arm, clipping-frequency logging, and the
  list of modes that build the SVD cache. **None touches the counter, the loss division, the clipping call, `opt.step`, `sched.step` or
  `zero_grad`.**
- **Runs before 16/09 (456 completed), not attestable by git.** Each was replayed from its own configuration with the current logic, and its
  replayed supervised-token count compared with its own log at every logged step: **448 match at every logged step, 0 diverge.** For 433 of
  them the test discriminates — a counter reset at each epoch, carrying the gradient of the partial window, would have given other counts;
  for 15, both logics give the same counts at the logged steps. **8 have a damaged `log.csv`** and could not be replayed: 4 knowledge-task
  runs (`facts`, not in the paper) and 4 format-task runs that are selection seeds or a report seed at a non-selected rate — none is in a
  reported cell.
- **Conclusion.** The defect documented in the rank paper's old trainer is not present in this trainer, and no run of the paper is touched.
  Whether this trainer shares code with the rank paper's could not be checked: its code is not available here.

## 2. The gradient probe on W0 (B@grad, the accumulated gradient, analysis A6)
- **Where.** `scripts/grad_bases.py`, top-level code (lines 37 to 67), MD5 `22cda03d9ec4c622be10333edea53e35` on the server.
- Each job is a fresh process that loads the model; every parameter is frozen, then only the target weights W0 get `requires_grad`; each
  module has its own fp32 accumulator; after each micro-batch's `backward()` the gradient is added to it and `lin.weight.grad = None`, so
  the next micro-batch starts from no gradient; the result is the mean over micro-batches. There is no optimiser and no optimisation step.
  One process computes one sequence set for one pair, so nothing can leak from another module, set or computation. Cross-check: the
  32-direction jobs of A6, separate processes, reproduce the `grad` bases with an overlap of 1.0000 on both pairs.

## Addendum (01/10) — correction de deux affirmations ci-dessus
- Les 8 runs au journal « endommagé » ne contenaient que 1 à 5 octets nuls ; une fois retirés, toutes les lignes se relisent (28/28 pas
  journalisés sur la tâche de format, 15/15 sur la tâche de connaissances). Rejoués : les 4 runs de la tâche de format CONCORDENT de façon
  discriminante ; les 4 de la tâche de connaissances concordent sans discriminer, car ils s'arrêtent à 147 pas, avant la première
  frontière d'époque (défaut impossible).
- Bilan des 456 runs d'avant le 16/09 : 437 prouvés par un rejeu discriminant ; 19 (tâche de connaissances) où le défaut ne pouvait pas se
  produire ; aucun invérifiable.
- CORRECTION : la tâche de connaissances N'EST PAS hors du papier — le compagnon en rapporte des résultats (§S5). Les affirmations « aucun
  n'est dans une cellule rapportée » ci-dessus sont retirées : jusqu'à 5 des 8 runs nourrissent un résultat du compagnon (4 dans l'énoncé
  descriptif de §S5.1, 1 possiblement dans l'inversion au taux commun) ; aucun ne nourrit l'article principal. Sans conséquence, puisque
  les 8 sont vérifiés ou à l'abri du défaut.
