# A3c — PARTITION model behind each reported cell, ambiguities resolved (post hoc, descriptive; ANALYSES_tmlr_addendum3.md)

Published values matched (A3b): 133; of which ambiguous: 38, resolved by table, row and column: 38.
Distinct reported cells: 137; entirely on RTX 4000 Ada: 86; on more than one PARTITION model: 51; on a single other model: 0.
Runs by PARTITION model: RTX 4000 Ada 577, GeForce RTX 3090 39, RTX 2000 Ada 33, RTX A5000 27, RTX A4000 23, (none) 6.
Cells with runs without a PARTITION record: 2: 0.6443 (top r2: 199af6be21b2, f745e2681d95, ca5bfc6a0282); 0.5922 (bottom r2: 1884fded62c3, eefa005f4484, fcb6a3e1e5f6)
Problems: 0.

| PARTITION composition | cells |
|---|---|
| RTX 4000 Ada | 86 |
| GeForce RTX 3090 + RTX 4000 Ada | 7 |
| RTX 2000 Ada + RTX 4000 Ada + RTX A5000 | 5 |
| GeForce RTX 3090 + RTX 2000 Ada + RTX 4000 Ada | 5 |
| RTX 2000 Ada + RTX 4000 Ada | 5 |
| RTX 4000 Ada + RTX A5000 | 4 |
| RTX 4000 Ada + RTX A4000 | 4 |
| GeForce RTX 3090 + RTX 4000 Ada + RTX A5000 | 3 |
| RTX 2000 Ada + RTX 4000 Ada + RTX A4000 + RTX A5000 | 3 |
| GeForce RTX 3090 + RTX 4000 Ada + RTX A4000 | 3 |
| GeForce RTX 3090 + RTX 2000 Ada + RTX 4000 Ada + RTX A5000 | 3 |
| GeForce RTX 3090 + RTX 2000 Ada + RTX 4000 Ada + RTX A4000 | 2 |
| GeForce RTX 3090 + RTX 4000 Ada + RTX A4000 + RTX A5000 | 2 |
| (none) + RTX 4000 Ada | 1 |
| (none) + RTX 4000 Ada + RTX A5000 | 1 |
| RTX 2000 Ada + RTX 4000 Ada + RTX A4000 | 1 |
| RTX 4000 Ada + RTX A4000 + RTX A5000 | 1 |
| GeForce RTX 3090 + RTX 2000 Ada + RTX 4000 Ada + RTX A4000 + RTX A5000 | 1 |

## Every cell

| table(s) | task | model | arm | rank | rate | seeds | value | PARTITION models (runs) | resolved by |
|---|---|---|---|---|---|---|---|---|---|
| tab:answeronly, tab:central, tab:latescores, tab:placements | commonsense | Qwen2.5-1.5B | bottom | 2 | 0.002 | 2-6 | 0.4220 | GeForce RTX 3090 (2), RTX 4000 Ada (3) | unique |
| tab:answeronly | commonsense | Qwen2.5-1.5B | bottom | 2 | 0.001 | 2-6 | 0.4576 | RTX 4000 Ada (5) | unique |
| tab:answeronly, tab:central, tab:v2budget | commonsense | Qwen2.5-1.5B | dual_bottom | 2 | 0.01 | 2-6 | 0.5248 | RTX 4000 Ada (5) | unique |
| tab:answeronly | commonsense | Qwen2.5-1.5B | dual_bottom | 2 | 0.01 | 2-6 | 0.5548 | RTX 4000 Ada (5) | unique |
| tab:answeronly, tab:central, tab:v2budget | commonsense | Qwen2.5-1.5B | dual_top | 2 | 0.005 | 2-6 | 0.5308 | RTX 4000 Ada (5) | unique |
| tab:answeronly | commonsense | Qwen2.5-1.5B | dual_top | 2 | 0.005 | 2-6 | 0.5552 | RTX 4000 Ada (5) | table, row and column (tab:answeronly) |
| tab:answeronly, tab:central, tab:latescores, tab:placements | commonsense | Qwen2.5-1.5B | free | 1 | 0.002 | 2-6 | 0.5456 | GeForce RTX 3090 (2), RTX 2000 Ada (1), RTX 4000 Ada (1), RTX A5000 (1) | table, row and column (tab:answeronly) |
| tab:answeronly | commonsense | Qwen2.5-1.5B | free | 1 | 0.002 | 2-6 | 0.5572 | RTX 4000 Ada (5) | table, row and column (tab:answeronly) |
| tab:answeronly | commonsense | Qwen2.5-1.5B | top | 2 | 0.005 | 2-6 | 0.4836 | RTX 4000 Ada (5) | unique |
| tab:answeronly, tab:central, tab:latescores, tab:placements | commonsense | Qwen2.5-1.5B | top | 2 | 0.005 | 2-6 | 0.4652 | RTX 4000 Ada (5) | table, row and column (tab:answeronly) |
| tab:central | commonsense | Mistral-7B-v0.3 | bottom | 2 | 0.001 | 2-6 | 0.4776 | RTX 4000 Ada (5) | unique |
| tab:central | commonsense | Qwen2.5-7B | bottom | 2 | 0.0005 | 2-6 | 0.4840 | RTX 4000 Ada (5) | table, row and column (tab:central) |
| tab:central | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 1 | 0.002 | 2-6 | 0.5496 | RTX 4000 Ada (4), RTX A4000 (1) | unique |
| tab:central | commonsense | Qwen2.5-7B | bottom_unfrozen | 1 | 0.002 | 2-6 | 0.6264 | RTX 4000 Ada (5) | unique |
| tab:central | commonsense | Mistral-7B-v0.3 | bottom_unfrozen | 1 | 0.002 | 2-6 | 0.5924 | RTX 4000 Ada (5) | table, row and column (tab:central) |
| tab:central | commonsense | Qwen2.5-7B | bottom_unfrozen | 2 | 0.001 | 2-6 | 0.6312 | RTX 4000 Ada (5) | unique |
| tab:central | commonsense | Mistral-7B-v0.3 | bottom_unfrozen | 2 | 0.001 | 2-6 | 0.6048 | RTX 4000 Ada (5) | unique |
| tab:central, tab:mechreps, tab:placements | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 2 | 0.002 | 2-6 | 0.5528 | RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A5000 (2) | table, row and column (tab:central) |
| tab:central, tab:v2budget | commonsense | Qwen2.5-7B | dual_bottom | 2 | 0.002 | 2-6 | 0.6148 | RTX 4000 Ada (5) | unique |
| tab:central | commonsense | Mistral-7B-v0.3 | dual_bottom | 2 | 0.001 | 2-6 | 0.5860 | RTX 4000 Ada (5) | unique |
| tab:central, tab:v2budget | commonsense | Qwen2.5-7B | dual_top | 2 | 0.002 | 2-6 | 0.6364 | RTX 4000 Ada (5) | unique |
| tab:central | commonsense | Mistral-7B-v0.3 | dual_top | 2 | 0.0005 | 2-6 | 0.6076 | RTX 4000 Ada (5) | unique |
| tab:central | commonsense | Qwen2.5-7B | free | 1 | 0.002 | 2-6 | 0.6284 | RTX 4000 Ada (5) | unique |
| tab:central | commonsense | Mistral-7B-v0.3 | free | 1 | 0.001 | 2-6 | 0.6012 | RTX 4000 Ada (5) | unique |
| tab:central, tab:v2trend | commonsense | Qwen2.5-1.5B | free | 2 | 0.005 | 2-6 | 0.5524 | RTX 4000 Ada (5) | table, row and column (tab:v2trend) |
| tab:central, tab:mechreps, tab:placements | commonsense | Qwen2.5-1.5B | learned | 2 | 0.005 | 2-6 | 0.5528 | RTX 4000 Ada (5) | table, row and column (tab:mechreps) |
| tab:central | commonsense | Qwen2.5-7B | top | 2 | 0.005 | 2-6 | 0.5316 | RTX 4000 Ada (5) | unique |
| tab:central | commonsense | Mistral-7B-v0.3 | top | 2 | 0.001 | 2-6 | 0.4764 | RTX 4000 Ada (5) | table, row and column (tab:central) |
| tab:central | commonsense | Qwen2.5-7B | top_unfrozen | 1 | 0.002 | 2-6 | 0.6232 | RTX 4000 Ada (5) | unique |
| tab:central, tab:v2trend | commonsense | Qwen2.5-1.5B | top_unfrozen | 1 | 0.002 | 2-6 | 0.5524 | RTX 4000 Ada (5) | table, row and column (tab:central) |
| tab:central | commonsense | Mistral-7B-v0.3 | top_unfrozen | 1 | 0.002 | 2-6 | 0.5924 | RTX 4000 Ada (5) | table, row and column (tab:central) |
| tab:central | commonsense | Qwen2.5-1.5B | top_unfrozen | 2 | 0.002 | 2-6 | 0.5576 | RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A4000 (1), RTX A5000 (1) | unique |
| tab:central | commonsense | Qwen2.5-7B | top_unfrozen | 2 | 0.001 | 2-6 | 0.6292 | RTX 4000 Ada (5) | unique |
| tab:central | commonsense | Mistral-7B-v0.3 | top_unfrozen | 2 | 0.001 | 2-6 | 0.5988 | RTX 4000 Ada (5) | unique |
| tab:central, tab:placements | format | Llama-3.2-1B | bottom | 2 | 0.005 | 2-6 | 0.5922 | (none) (3), RTX 4000 Ada (1), RTX A5000 (1) | unique |
| tab:central, tab:latescores | format | Qwen2.5-1.5B | bottom | 2 | 0.005 | 22-26 | 0.6391 | RTX 4000 Ada (5) | unique |
| tab:central | format | Llama-3.2-1B | bottom_unfrozen | 1 | 0.005 | 2-6 | 0.7954 | GeForce RTX 3090 (1), RTX 4000 Ada (4) | unique |
| tab:central | format | Qwen2.5-1.5B | bottom_unfrozen | 1 | 0.005 | 22-26 | 0.7811 | RTX 4000 Ada (5) | table, row and column (tab:central) |
| tab:central | format | Llama-3.2-1B | bottom_unfrozen | 2 | 0.002 | 2-6 | 0.8072 | RTX 2000 Ada (1), RTX 4000 Ada (4) | unique |
| tab:central | format | Qwen2.5-1.5B | bottom_unfrozen | 2 | 0.005 | 22-26 | 0.7857 | RTX 4000 Ada (5) | unique |
| tab:central | format | Llama-3.2-1B | dual_bottom | 2 | 0.01 | 2-6 | 0.7818 | GeForce RTX 3090 (1), RTX 2000 Ada (2), RTX 4000 Ada (1), RTX A4000 (1) | table, row and column (tab:central) |
| tab:central | format | Llama-3.2-1B | dual_top | 2 | 0.01 | 2-6 | 0.7739 | RTX 4000 Ada (5) | unique |
| tab:central, tab:latescores | format | Qwen2.5-1.5B | free | 1 | 0.005 | 22-26 | 0.7879 | RTX 4000 Ada (5) | unique |
| tab:central, tab:mechreps, tab:placements | format | Llama-3.2-1B | free | 1 | 0.001 | 2-6 | 0.7684 | GeForce RTX 3090 (3), RTX 2000 Ada (1), RTX 4000 Ada (1) | table, row and column (tab:central) |
| tab:central, tab:mechreps, tab:placements | format | Llama-3.2-1B | top | 2 | 0.01 | 2-6 | 0.6443 | (none) (3), RTX 4000 Ada (2) | unique |
| tab:central, tab:latescores | format | Qwen2.5-1.5B | top | 2 | 0.01 | 22-26 | 0.6257 | RTX 4000 Ada (5) | table, row and column (tab:central) |
| tab:central | format | Llama-3.2-1B | top_unfrozen | 1 | 0.005 | 2-6 | 0.7928 | GeForce RTX 3090 (1), RTX 4000 Ada (3), RTX A4000 (1) | unique |
| tab:central | format | Qwen2.5-1.5B | top_unfrozen | 1 | 0.005 | 22-26 | 0.7896 | RTX 4000 Ada (5) | table, row and column (tab:central) |
| tab:central | format | Llama-3.2-1B | top_unfrozen | 2 | 0.005 | 2-6 | 0.8000 | GeForce RTX 3090 (1), RTX 4000 Ada (4) | unique |
| tab:central | format | Qwen2.5-1.5B | top_unfrozen | 2 | 0.005 | 22-26 | 0.7775 | RTX 4000 Ada (5) | unique |
| tab:grad, tab:placements | commonsense | Qwen2.5-1.5B | learned | 2 | 0.002 | 2-6 | 0.4272 | RTX 4000 Ada (5) | unique |
| tab:grad, tab:placements | format | Llama-3.2-1B | learned | 2 | 0.005 | 2-6 | 0.6658 | RTX 4000 Ada (5) | table, row and column (tab:grad) |
| tab:grad, tab:mechreps | format | Llama-3.2-1B | top | 2 | 0.01 | 2-6 | 0.6430 | RTX 4000 Ada (5) | unique |
| tab:gradacc, tab:mechreps, tab:warm | commonsense | Qwen2.5-1.5B | bottom | 2 | 0.002 | 2-6 | 0.4276 | RTX 4000 Ada (5) | unique |
| tab:gradacc, tab:mechreps, tab:warm | commonsense | Qwen2.5-1.5B | free | 1 | 0.002 | 2-6 | 0.5484 | RTX 4000 Ada (5) | unique |
| tab:gradacc | commonsense | Qwen2.5-1.5B | learned | 2 | 0.002 | 2-6 | 0.4304 | RTX 4000 Ada (5) | unique |
| tab:gradacc, tab:mechreps, tab:warm | format | Llama-3.2-1B | free | 1 | 0.001 | 2-6 | 0.7687 | RTX 4000 Ada (5) | unique |
| tab:gradacc | format | Llama-3.2-1B | learned | 2 | 0.005 | 2-6 | 0.6622 | RTX 4000 Ada (5) | unique |
| tab:gradacc, tab:warm | format | Llama-3.2-1B | top_sigma | 2 | 0.005 | 50-64 | 0.6631 | GeForce RTX 3090 (6), RTX 4000 Ada (7), RTX A5000 (2) | unique |
| tab:latescores | commonsense | Qwen2.5-1.5B | bottom | 2 | 0.002 | 2-6 | 0.4668 | GeForce RTX 3090 (1), RTX 4000 Ada (4) | table, row and column (tab:latescores) |
| tab:latescores | commonsense | Qwen2.5-1.5B | bottom | 16 | 0.0005 | 2-6 | 0.7792 | RTX 4000 Ada (3), RTX A4000 (2) | unique |
| tab:latescores, tab:rank16 | commonsense | Qwen2.5-1.5B | bottom | 16 | 0.001 | 2-6 | 0.4452 | GeForce RTX 3090 (1), RTX 4000 Ada (2), RTX A4000 (1), RTX A5000 (1) | unique |
| tab:latescores, tab:rank16 | commonsense | Qwen2.5-1.5B | bottom | 16 | 0.001 | 2-6 | 0.4984 | GeForce RTX 3090 (3), RTX 4000 Ada (2) | table, row and column (tab:latescores) |
| tab:latescores | commonsense | Qwen2.5-1.5B | bottom | 32 | 0.0002 | 2-6 | 0.4758 | RTX 4000 Ada (2), RTX A4000 (2), RTX A5000 (1) | unique |
| tab:latescores | commonsense | Qwen2.5-1.5B | free | 1 | 0.001 | 2-6 | 0.5044 | RTX 4000 Ada (4), RTX A5000 (1) | unique |
| tab:latescores, tab:rank16 | commonsense | Qwen2.5-1.5B | free | 7 | 0.002 | 2-6 | 0.5616 | RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A5000 (2) | unique |
| tab:latescores | commonsense | Qwen2.5-1.5B | free | 7 | 0.0001 | 2-6 | 0.7940 | RTX 2000 Ada (1), RTX 4000 Ada (4) | unique |
| tab:latescores, tab:rank16 | commonsense | Qwen2.5-1.5B | free | 7 | 0.0005 | 2-6 | 0.4952 | RTX 4000 Ada (4), RTX A4000 (1) | unique |
| tab:latescores | commonsense | Qwen2.5-1.5B | free | 14 | 0.0005 | 2-6 | 0.4888 | RTX 4000 Ada (4), RTX A5000 (1) | table, row and column (tab:latescores) |
| tab:latescores, tab:placements | commonsense | Qwen2.5-1.5B | random_ortho | 2 | 0.02 | 2-6 | 0.4592 | RTX 4000 Ada (5) | unique |
| tab:latescores | commonsense | Qwen2.5-1.5B | random_ortho | 2 | 0.005 | 2-6 | 0.4692 | RTX 2000 Ada (1), RTX 4000 Ada (4) | table, row and column (tab:latescores) |
| tab:latescores | commonsense | Qwen2.5-1.5B | random_ortho | 16 | 0.0005 | 2-6 | 0.7970 | RTX 4000 Ada (5) | unique |
| tab:latescores | commonsense | Qwen2.5-1.5B | random_ortho | 16 | 0.001 | 2-6 | 0.4860 | RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A4000 (1), RTX A5000 (1) | unique |
| tab:latescores | commonsense | Qwen2.5-1.5B | random_ortho | 32 | 0.0005 | 2-6 | 0.4960 | GeForce RTX 3090 (2), RTX 4000 Ada (3) | table, row and column (tab:latescores) |
| tab:latescores | commonsense | Qwen2.5-1.5B | top | 2 | 0.005 | 2-6 | 0.4842 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (3) | unique |
| tab:latescores, tab:rank16 | commonsense | Qwen2.5-1.5B | top | 16 | 0.002 | 2-6 | 0.5408 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (3) | unique |
| tab:latescores, tab:rank16 | commonsense | Qwen2.5-1.5B | top | 16 | 0.0005 | 2-6 | 0.4970 | GeForce RTX 3090 (1), RTX 4000 Ada (3), RTX A4000 (1) | unique |
| tab:latescores | commonsense | Qwen2.5-1.5B | top | 16 | 0.0001 | 2-6 | 0.7948 | RTX 2000 Ada (2), RTX 4000 Ada (2), RTX A5000 (1) | table, row and column (tab:latescores) |
| tab:latescores | commonsense | Qwen2.5-1.5B | top | 32 | 0.0005 | 2-6 | 0.4824 | GeForce RTX 3090 (1), RTX 4000 Ada (3), RTX A5000 (1) | table, row and column (tab:latescores) |
| tab:latescores | format | Llama-3.2-1B | bottom | 2 | 0.005 | 2-6 | 0.6313 | RTX 4000 Ada (5) | unique |
| tab:latescores | format | Llama-3.2-1B | bottom | 8 | 0.002 | 42-46 | 0.7450 | RTX 4000 Ada (5) | table, row and column (tab:latescores) |
| tab:latescores, tab:rank16 | format | Llama-3.2-1B | bottom | 16 | 0.0005 | 2-6 | 0.7554 | GeForce RTX 3090 (1), RTX 2000 Ada (2), RTX 4000 Ada (1), RTX A5000 (1) | unique |
| tab:latescores | format | Qwen2.5-1.5B | bottom | 16 | 0.001 | 22-26 | 0.7524 | RTX 4000 Ada (5) | table, row and column (tab:latescores) |
| tab:latescores | format | Llama-3.2-1B | free | 1 | 0.001 | 2-6 | 0.7984 | RTX 2000 Ada (1), RTX 4000 Ada (4) | table, row and column (tab:latescores) |
| tab:latescores | format | Llama-3.2-1B | free | 4 | 0.002 | 42-46 | 0.8293 | GeForce RTX 3090 (1), RTX 4000 Ada (3), RTX A5000 (1) | unique |
| tab:latescores | format | Qwen2.5-1.5B | free | 7 | 0.002 | 22-26 | 0.8078 | RTX 4000 Ada (5) | table, row and column (tab:latescores) |
| tab:latescores, tab:rank16 | format | Llama-3.2-1B | free | 8 | 0.002 | 2-6 | 0.8205 | GeForce RTX 3090 (2), RTX 4000 Ada (1), RTX A4000 (1), RTX A5000 (1) | unique |
| tab:latescores | format | Llama-3.2-1B | random | 2 | 0.01 | 2-6 | 0.6935 | RTX 2000 Ada (3), RTX 4000 Ada (1), RTX A5000 (1) | unique |
| tab:latescores | format | Llama-3.2-1B | random | 8 | 0.005 | 42-46 | 0.7550 | GeForce RTX 3090 (1), RTX 4000 Ada (4) | unique |
| tab:latescores | format | Llama-3.2-1B | random | 16 | 0.001 | 2-6 | 0.7700 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A5000 (1) | unique |
| tab:latescores | format | Qwen2.5-1.5B | random_ortho | 2 | 0.02 | 22-26 | 0.6238 | RTX 4000 Ada (5) | unique |
| tab:latescores | format | Llama-3.2-1B | random_ortho | 2 | 0.01 | 2-6 | 0.6896 | RTX 2000 Ada (1), RTX 4000 Ada (3), RTX A5000 (1) | table, row and column (tab:latescores) |
| tab:latescores | format | Llama-3.2-1B | random_ortho | 8 | 0.005 | 42-46 | 0.7505 | GeForce RTX 3090 (1), RTX 2000 Ada (2), RTX 4000 Ada (1), RTX A4000 (1) | unique |
| tab:latescores | format | Llama-3.2-1B | random_ortho | 16 | 0.001 | 2-6 | 0.7795 | RTX 2000 Ada (1), RTX 4000 Ada (3), RTX A4000 (1) | unique |
| tab:latescores | format | Qwen2.5-1.5B | random_ortho | 16 | 0.005 | 22-26 | 0.7521 | RTX 4000 Ada (5) | unique |
| tab:latescores | format | Llama-3.2-1B | top | 2 | 0.01 | 2-6 | 0.6967 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (3) | unique |
| tab:latescores | format | Llama-3.2-1B | top | 8 | 0.002 | 42-46 | 0.7450 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (3) | table, row and column (tab:latescores) |
| tab:latescores, tab:rank16 | format | Llama-3.2-1B | top | 16 | 0.002 | 2-6 | 0.7756 | RTX 4000 Ada (4), RTX A4000 (1) | unique |
| tab:latescores | format | Qwen2.5-1.5B | top | 16 | 0.002 | 22-26 | 0.7430 | RTX 4000 Ada (5) | unique |
| tab:mechreps, tab:placements | commonsense | Qwen2.5-1.5B | learned | 2 | 0.002 | 2-6 | 0.5588 | RTX 4000 Ada (5) | unique |
| tab:mechreps, tab:placements | format | Llama-3.2-1B | learned | 2 | 0.005 | 2-6 | 0.8081 | RTX 4000 Ada (5) | table, row and column (tab:mechreps) |
| tab:placements | format | Llama-3.2-1B | random_ortho | 2 | 0.01 | 2-6 | 0.6534 | GeForce RTX 3090 (1), RTX 4000 Ada (2), RTX A4000 (2) | table, row and column (tab:placements) |
| tab:possweep | format | Llama-3.2-1B | band | 2 | 0.01 | 2-8 | 0.6547 | RTX 2000 Ada (2), RTX 4000 Ada (1), RTX A4000 (2), RTX A5000 (2) | unique |
| tab:possweep | format | Llama-3.2-1B | band | 2 | 0.01 | 2-8 | 0.6368 | RTX 4000 Ada (6), RTX A5000 (1) | unique |
| tab:possweep | format | Llama-3.2-1B | band | 2 | 0.01 | 2-8 | 0.6768 | RTX 4000 Ada (6), RTX A5000 (1) | unique |
| tab:possweep | format | Llama-3.2-1B | band | 2 | 0.005 | 2-8 | 0.5928 | RTX 2000 Ada (2), RTX 4000 Ada (5) | table, row and column (tab:possweep) |
| tab:possweep | format | Llama-3.2-1B | band | 2 | 0.01 | 2-8 | 0.6671 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (1), RTX A4000 (3), RTX A5000 (1) | table, row and column (tab:possweep) |
| tab:v2budget | commonsense | Qwen2.5-1.5B | bottom | 3 | 0.002 | 2-6 | 0.4328 | RTX 4000 Ada (5) | unique |
| tab:v2budget | commonsense | Qwen2.5-7B | bottom | 3 | 0.0005 | 2-6 | 0.4924 | RTX 4000 Ada (5) | unique |
| tab:v2budget, tab:v2trend | commonsense | Qwen2.5-1.5B | bottom | 4 | 0.002 | 2-6 | 0.4524 | RTX 4000 Ada (5) | unique |
| tab:v2budget | commonsense | Qwen2.5-7B | bottom | 4 | 0.001 | 2-6 | 0.5136 | RTX 4000 Ada (5) | unique |
| tab:v2budget | commonsense | Qwen2.5-1.5B | top | 3 | 0.005 | 2-6 | 0.4680 | RTX 4000 Ada (5) | unique |
| tab:v2budget | commonsense | Qwen2.5-7B | top | 3 | 0.002 | 2-6 | 0.5464 | RTX 4000 Ada (5) | table, row and column (tab:v2budget) |
| tab:v2budget, tab:v2trend | commonsense | Qwen2.5-1.5B | top | 4 | 0.005 | 2-6 | 0.4644 | RTX 4000 Ada (5) | table, row and column (tab:v2budget) |
| tab:v2trend | commonsense | Qwen2.5-1.5B | bottom | 8 | 0.001 | 2-6 | 0.4656 | RTX 4000 Ada (5) | table, row and column (tab:v2trend) |
| tab:v2trend | commonsense | Qwen2.5-7B | bottom | 16 | 0.0005 | 2-6 | 0.5676 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 2 | 0.002 | 2-6 | 0.5520 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 4 | 0.002 | 2-6 | 0.5420 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-7B | bottom_unfrozen | 7 | 0.0005 | 2-6 | 0.6276 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 8 | 0.001 | 2-6 | 0.5556 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-7B | bottom_unfrozen | 16 | 0.0005 | 2-6 | 0.6200 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-1.5B | dual_bottom | 4 | 0.002 | 2-6 | 0.5388 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-7B | dual_bottom | 16 | 0.0005 | 2-6 | 0.6256 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-1.5B | dual_top | 4 | 0.005 | 2-6 | 0.5344 | RTX 4000 Ada (5) | table, row and column (tab:v2trend) |
| tab:v2trend | commonsense | Qwen2.5-1.5B | dual_top | 8 | 0.002 | 2-6 | 0.5628 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-7B | dual_top | 16 | 0.001 | 2-6 | 0.6240 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-1.5B | free | 4 | 0.002 | 2-6 | 0.5544 | RTX 4000 Ada (5) | table, row and column (tab:v2trend) |
| tab:v2trend | commonsense | Qwen2.5-7B | free | 7 | 0.0005 | 2-6 | 0.6320 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-1.5B | top | 8 | 0.005 | 2-6 | 0.4992 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-7B | top | 16 | 0.001 | 2-6 | 0.6020 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-1.5B | top_unfrozen | 2 | 0.002 | 2-6 | 0.5540 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-1.5B | top_unfrozen | 4 | 0.001 | 2-6 | 0.5504 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-7B | top_unfrozen | 7 | 0.0005 | 2-6 | 0.6248 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-1.5B | top_unfrozen | 8 | 0.001 | 2-6 | 0.5636 | RTX 4000 Ada (5) | unique |
| tab:v2trend | commonsense | Qwen2.5-7B | top_unfrozen | 16 | 0.0005 | 2-6 | 0.6332 | RTX 4000 Ada (5) | table, row and column (tab:v2trend) |
| tab:warm | commonsense | Qwen2.5-1.5B | learned | 2 | 0.005 | 2-6 | 0.5116 | RTX 4000 Ada (5) | table, row and column (tab:warm) |
| tab:warm | format | Llama-3.2-1B | learned | 2 | 0.005 | 2-6 | 0.7221 | RTX 4000 Ada (5) | unique |
