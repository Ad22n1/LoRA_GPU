# A2 — Minimum detectable effect at 80 % power of the band comparisons not established (post hoc, descriptive)

| row | comparison | n | alpha (sided) | d_z | sigma_d | MDE | observed | how |
|---|---|---|---|---|---|---|---|---|
| 1 | beta=0.75 vs beta=0, first seven seeds (Llama) | 7 | 0.0125 (2) | 1.72 | 1.95 | 3.36 points | +2.21 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 1 | beta=0.75 vs beta=0.5, first seven seeds (Llama) | 7 | 0.0125 (2) | 1.72 | 2.62 | 4.51 points | +0.98 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 3 | beta=0.75 vs random_ortho, first seven seeds (Llama) | 7 | 0.01 (2) | 1.80 | 2.18 | 3.92 points | +2.42 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 4 | beta=0.75 vs beta=0.5, fresh seeds 9-16 (Llama) | 8 | 0.01 (1) | 1.40 | 3.58 | 5.01 points | +1.38 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 5 | beta=0.75 vs random_ortho, fresh seeds 9-16 (Llama) | 8 | 0.01 (1) | 1.40 | 3.46 | 4.84 points | +3.36 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 10 | beta=0.75 vs the four other positions, first five seeds (Qwen2.5-0.5B) | 5 | 0.0125 (2) | 2.48 | — | — | — | sigma_d from the runs (A2b): the paper prints no mean difference |
| 12 | beta=0.75 vs beta=1, ten added seeds (Qwen2.5-0.5B) | 10 | 0.0125 (1) | 1.14 | 3.66 | 4.17 points | +2.28 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 14 | beta=0.75 vs random_ortho, fifteen seeds (Qwen2.5-0.5B) | 15 | 0.0125 (1) | 0.88 | 2.48 | 2.17 points | +0.48 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 17 | bottom vs random_ortho at 5e-3 (Qwen2.5-0.5B) | 5 | 0.01667 (2) | 2.29 | 3.49 | 8.01 points | +1.50 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 17 | bottom vs random_ortho at 2e-3 (Qwen2.5-0.5B) | 5 | 0.01667 (2) | 2.29 | 1.43 | 3.28 points | +2.41 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 18 | common rate, Llama rank 2: lead of bottom at 2e-3 | 5 | 0.05 (2) | 1.68 | 1.96 | 3.29 points | +0.98 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 21 | common rate, Llama rank 4: lead of bottom at 2e-3 | 5 | 0.05 (2) | 1.68 | 2.08 | 3.50 points | +2.25 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 26 | common rate, Qwen2.5-1.5B: deficit of bottom at 1e-2 | 5 | 0.05 (2) | 1.68 | 3.27 | 5.50 points | +3.13 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 35 | rank 2, along the spectrum (bootstrap interval) | 5 | 0.05 (2) | 1.68 | 4.73 | 5.93 points | [-4.30, +4.00] | bootstrap interval: MDE = (z_alpha + z_0.80) SE, normal approximation |
| 42 | top vs random at 1e-2, perplexity | 5 | 0.05 (2) | 1.68 | 16.10 | 27.09 perplexity | +15.70 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 48 | top_sigma vs beta=0.75 (Llama) | 5 | 0.05 (2) | 1.68 | 1.90 | 3.20 points | +0.23 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 56 | OpenBookQA rank 2: lead of bottom (observed -2.88) | 5 | 0.025 (2) | 2.05 | 1.59 | 3.25 points | +2.88 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 58 | ARC-Easy: lead of bottom (observed -2.90) | 5 | 0.05 (2) | 1.68 | 0.60 | 1.02 points | +2.90 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 110 | Mistral-7B rank 2: band gap, B trained vs frozen | 5 | 0.05 (2) | 1.68 | — | — | — | sigma_d from the runs (A2b): the paper prints no mean difference |
| 125 | Mistral-7B rank 2: band component of factor x band (bootstrap interval) | 5 | 0.05 (2) | 1.68 | 1.59 | 1.99 points | [-2.34, +0.44] | bootstrap interval: MDE = (z_alpha + z_0.80) SE, normal approximation |
| 138 | OpenBookQA rank 8: band gap, A frozen vs B frozen | 5 | 0.05 (2) | 1.68 | 2.47 | 4.16 points | +0.52 | sigma_d = |mean| sqrt(n) / |t|, printed values |
| 142 | Qwen2.5-7B rank 16: band gap, B trained vs frozen | 5 | 0.05 (2) | 1.68 | 2.52 | 4.24 points | +2.12 | sigma_d = |mean| sqrt(n) / |t|, printed values |
