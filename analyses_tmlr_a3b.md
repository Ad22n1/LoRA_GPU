# A3b — PARTITION model behind each reported cell, corrected (post hoc, descriptive; ANALYSES_tmlr_addendum2.md)

4-decimal values in the main paper's tables: 134; matched by at least one cell: 133; matched by none: 0.5905.
Cells found: 182; entirely on RTX 4000 Ada: 108; on more than one PARTITION model: 74; with a run without a PARTITION record: 2.
Runs by PARTITION model: RTX 4000 Ada 772, GeForce RTX 3090 73, RTX 2000 Ada 51, RTX A4000 40, RTX A5000 36, (none) 6.

## Values matched by more than one cell (38)

| value | table(s) | cells (task, arm, rank, rate, seeds: PARTITION models) | same PARTITION composition |
|---|---|---|---|
| 0.4644 | tab:v2budget, tab:v2trend | format Qwen2.5-0.5B band r2 0.01 9-16: RTX 2000 Ada (1), RTX 4000 Ada (6), RTX A4000 (1); commonsense Qwen2.5-1.5B top r2 0.005 2-6: GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (3); commonsense Qwen2.5-1.5B top r4 0.005 2-6: RTX 4000 Ada (5) | NO |
| 0.4652 | tab:answeronly, tab:central, tab:latescores, tab:placements | commonsense Qwen2.5-1.5B top r2 0.005 2-6: RTX 4000 Ada (5); commonsense Qwen2.5-1.5B random r2 0.005 2-6: GeForce RTX 3090 (3), RTX 4000 Ada (1), RTX A4000 (1) | NO |
| 0.4656 | tab:v2trend | commonsense Qwen2.5-1.5B top r2 0.005 2-6: RTX 2000 Ada (1), RTX 4000 Ada (4); commonsense Qwen2.5-1.5B bottom r8 0.001 2-6: RTX 4000 Ada (5) | NO |
| 0.4668 | tab:latescores | commonsense Qwen2.5-1.5B top r2 0.005 2-6: RTX 2000 Ada (1), RTX 4000 Ada (3), RTX A4000 (1); commonsense Qwen2.5-1.5B bottom r2 0.002 2-6: GeForce RTX 3090 (1), RTX 4000 Ada (4) | NO |
| 0.4692 | tab:latescores | commonsense Qwen2.5-1.5B random_ortho r2 0.005 2-6: RTX 2000 Ada (1), RTX 4000 Ada (4); commonsense Qwen2.5-1.5B bottom_unfrozen r16 0.0005 2-6: RTX 4000 Ada (5) | NO |
| 0.4764 | tab:central | commonsense Qwen2.5-1.5B bottom r2 0.0002 2-6: RTX 4000 Ada (4), RTX A5000 (1); commonsense Mistral-7B-v0.3 top r2 0.001 2-6: RTX 4000 Ada (5) | NO |
| 0.4824 | tab:latescores | commonsense Qwen2.5-1.5B top r32 0.0005 2-6: GeForce RTX 3090 (1), RTX 4000 Ada (3), RTX A5000 (1); commonsense Qwen2.5-1.5B top_unfrozen r2 0.01 2-6: GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A4000 (1) | NO |
| 0.4840 | tab:central | format Qwen2.5-0.5B bottom r2 0.002 2-6: RTX 2000 Ada (2), RTX 4000 Ada (2), RTX A5000 (1); commonsense Qwen2.5-7B bottom r2 0.0005 2-6: RTX 4000 Ada (5); commonsense Qwen2.5-1.5B top_unfrozen r16 0.0001 2-6: RTX 4000 Ada (5) | NO |
| 0.4888 | tab:latescores | commonsense Qwen2.5-1.5B free r14 0.0005 2-6: RTX 4000 Ada (4), RTX A5000 (1); commonsense Qwen2.5-1.5B top r2 0.002 2-6: GeForce RTX 3090 (1), RTX 4000 Ada (3), RTX A5000 (1) | NO |
| 0.4960 | tab:latescores | commonsense Qwen2.5-1.5B top r16 0.0005 2-6: GeForce RTX 3090 (1), RTX 4000 Ada (3), RTX A4000 (1); commonsense Qwen2.5-1.5B free r1 0.0005 2-6: RTX 4000 Ada (4), RTX A4000 (1); commonsense Qwen2.5-1.5B random_ortho r32 0.0005 2-6: GeForce RTX 3090 (2), RTX 4000 Ada (3) | NO |
| 0.4984 | tab:latescores, tab:rank16 | commonsense Qwen2.5-1.5B bottom r16 0.001 2-6: GeForce RTX 3090 (3), RTX 4000 Ada (2); commonsense Qwen2.5-1.5B top r16 0.0005 2-6: RTX 4000 Ada (4), RTX A4000 (1) | NO |
| 0.5116 | tab:warm | commonsense Qwen2.5-1.5B bottom_unfrozen r1 0.01 2-6: RTX 4000 Ada (4), RTX A4000 (1); commonsense Qwen2.5-1.5B learned r2 0.005 2-6: RTX 4000 Ada (5) | NO |
| 0.5344 | tab:v2trend | commonsense Qwen2.5-1.5B dual_bottom r8 0.002 2-6: RTX 4000 Ada (5); commonsense Qwen2.5-1.5B dual_top r4 0.005 2-6: RTX 4000 Ada (5) | yes |
| 0.5456 | tab:answeronly, tab:central, tab:latescores, tab:placements | commonsense Qwen2.5-1.5B free r1 0.002 2-6: GeForce RTX 3090 (2), RTX 2000 Ada (1), RTX 4000 Ada (1), RTX A5000 (1); commonsense Qwen2.5-1.5B bottom_unfrozen r7 0.002 2-6: RTX 4000 Ada (5) | NO |
| 0.5464 | tab:v2budget | commonsense Qwen2.5-1.5B dual_random_ortho r2 0.01 2-6: RTX 4000 Ada (5); commonsense Qwen2.5-7B top r3 0.002 2-6: RTX 4000 Ada (5); commonsense Qwen2.5-7B top r4 0.005 2-6: RTX 4000 Ada (5) | yes |
| 0.5524 | tab:central, tab:v2trend | commonsense Qwen2.5-1.5B top_unfrozen r1 0.002 2-6: RTX 4000 Ada (5); commonsense Qwen2.5-1.5B free r2 0.005 2-6: RTX 4000 Ada (5) | yes |
| 0.5528 | tab:central, tab:mechreps, tab:placements | commonsense Qwen2.5-1.5B bottom_unfrozen r2 0.002 2-6: RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A5000 (2); commonsense Qwen2.5-1.5B bottom_unfrozen r16 0.001 2-6: RTX 4000 Ada (5); commonsense Qwen2.5-1.5B learned r2 0.005 2-6: RTX 4000 Ada (5) | NO |
| 0.5544 | tab:v2trend | commonsense Qwen2.5-1.5B dual_top r16 0.002 2-6: RTX 4000 Ada (5); commonsense Qwen2.5-1.5B free r4 0.002 2-6: RTX 4000 Ada (5) | yes |
| 0.5552 | tab:answeronly | commonsense Qwen2.5-1.5B top_unfrozen r7 0.001 2-6: RTX 4000 Ada (5); commonsense Qwen2.5-1.5B bottom_unfrozen r7 0.001 2-6: RTX 4000 Ada (5); commonsense Qwen2.5-1.5B dual_top r2 0.005 2-6: RTX 4000 Ada (5) | yes |
| 0.5572 | tab:answeronly | commonsense Qwen2.5-1.5B top_unfrozen r7 0.002 2-6: RTX 4000 Ada (5); commonsense Qwen2.5-1.5B free r1 0.002 2-6: RTX 4000 Ada (5) | yes |
| 0.5924 | tab:central | commonsense Mistral-7B-v0.3 top_unfrozen r1 0.002 2-6: RTX 4000 Ada (5); commonsense Mistral-7B-v0.3 bottom_unfrozen r1 0.002 2-6: RTX 4000 Ada (5) | yes |
| 0.5928 | tab:possweep | format Llama-3.2-1B band r2 0.005 2-8: RTX 2000 Ada (2), RTX 4000 Ada (5); format Qwen2.5-1.5B top r16 0.01 22-26: RTX 4000 Ada (5) | NO |
| 0.6257 | tab:central, tab:latescores | format Qwen2.5-1.5B top r2 0.01 22-26: RTX 4000 Ada (5); format Llama-3.2-1B random r2 0.002 2-6: GeForce RTX 3090 (1), RTX 4000 Ada (4); format Llama-3.2-1B bottom r2 0.005 2-6: RTX 4000 Ada (5) | NO |
| 0.6332 | tab:v2trend | format Llama-3.2-1B band r2 0.01 2-6: RTX 4000 Ada (4), RTX A5000 (1); commonsense Qwen2.5-7B top_unfrozen r16 0.0005 2-6: RTX 4000 Ada (5) | NO |
| 0.6534 | tab:placements | format Llama-3.2-1B random_ortho r2 0.01 2-6: GeForce RTX 3090 (1), RTX 4000 Ada (2), RTX A4000 (2); format Llama-3.2-1B bottom_unfrozen r8 0.005 2-6: RTX 4000 Ada (5) | NO |
| 0.6658 | tab:grad, tab:placements | format Llama-3.2-1B top_scalar r2 0.005 50-64: GeForce RTX 3090 (3), RTX 2000 Ada (1), RTX 4000 Ada (9), RTX A4000 (1), RTX A5000 (1); format Llama-3.2-1B top_scalar r2 0.005 65-104: GeForce RTX 3090 (18), RTX 2000 Ada (4), RTX 4000 Ada (13), RTX A4000 (5); format Llama-3.2-1B learned r2 0.005 2-6: RTX 4000 Ada (5) | NO |
| 0.6671 | tab:possweep | format Llama-3.2-1B band r2 0.01 2-8: GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (1), RTX A4000 (3), RTX A5000 (1); format Llama-3.2-1B random r4 0.002 2-6: GeForce RTX 3090 (1), RTX 4000 Ada (1), RTX A4000 (2), RTX A5000 (1) | NO |
| 0.6896 | tab:latescores | format Llama-3.2-1B top r4 0.01 2-6: GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A5000 (1); format Llama-3.2-1B random_ortho r2 0.01 2-6: RTX 2000 Ada (1), RTX 4000 Ada (3), RTX A5000 (1); format Llama-3.2-1B bottom_unfrozen r16 0.0001 2-6: RTX 4000 Ada (5) | NO |
| 0.7450 | tab:latescores | format Llama-3.2-1B bottom r8 0.002 42-46: RTX 4000 Ada (5); format Llama-3.2-1B top r8 0.002 42-46: GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (3) | NO |
| 0.7524 | tab:latescores | format Llama-3.2-1B bottom r16 0.0005 2-6: GeForce RTX 3090 (1), RTX 4000 Ada (4); format Qwen2.5-1.5B bottom r16 0.001 22-26: RTX 4000 Ada (5) | NO |
| 0.7684 | tab:central, tab:mechreps, tab:placements | format Llama-3.2-1B free r1 0.001 2-6: GeForce RTX 3090 (3), RTX 2000 Ada (1), RTX 4000 Ada (1); format Llama-3.2-1B dual_random_ortho r2 0.02 2-6: GeForce RTX 3090 (2), RTX 2000 Ada (2), RTX A5000 (1) | NO |
| 0.7811 | tab:central | format Qwen2.5-1.5B bottom_unfrozen r1 0.005 22-26: RTX 4000 Ada (5); format Llama-3.2-1B free r1 0.002 2-6: RTX 4000 Ada (5) | yes |
| 0.7818 | tab:central | format Llama-3.2-1B dual_bottom r2 0.01 2-6: GeForce RTX 3090 (1), RTX 2000 Ada (2), RTX 4000 Ada (1), RTX A4000 (1); format Llama-3.2-1B dual_bottom r2 0.01 120-124: RTX 4000 Ada (5) | NO |
| 0.7896 | tab:central | format Llama-3.2-1B bottom_unfrozen r2 0.005 2-6: RTX 2000 Ada (2), RTX 4000 Ada (1), RTX A4000 (1), RTX A5000 (1); format Llama-3.2-1B bottom_unfrozen r16 0.002 2-6: RTX 4000 Ada (5); format Qwen2.5-1.5B top_unfrozen r1 0.005 22-26: RTX 4000 Ada (5) | NO |
| 0.7948 | tab:latescores | commonsense Qwen2.5-1.5B top r16 0.0001 2-6: RTX 2000 Ada (2), RTX 4000 Ada (2), RTX A5000 (1); commonsense Qwen2.5-1.5B random_ortho r16 0.0001 2-6: RTX 2000 Ada (1), RTX 4000 Ada (4) | NO |
| 0.7984 | tab:latescores | format Llama-3.2-1B free r1 0.001 2-6: RTX 2000 Ada (1), RTX 4000 Ada (4); format Llama-3.2-1B top_unfrozen r16 0.0002 2-6: RTX 4000 Ada (5) | NO |
| 0.8078 | tab:latescores | format Llama-3.2-1B top_unfrozen r1 0.005 120-124: RTX 4000 Ada (5); format Qwen2.5-1.5B free r7 0.002 22-26: RTX 4000 Ada (5) | yes |
| 0.8081 | tab:mechreps, tab:placements | format Llama-3.2-1B bottom_unfrozen r2 0.002 120-124: RTX 4000 Ada (5); format Llama-3.2-1B bottom_unfrozen r2 0.002 2-6: RTX 4000 Ada (5); format Llama-3.2-1B learned r2 0.005 2-6: RTX 4000 Ada (5) | yes |

## Every cell

| table(s) | task | model | arm | rank | rate | seeds | value | PARTITION models (runs) | matching run sets |
|---|---|---|---|---|---|---|---|---|---|
| tab:answeronly | commonsense | Qwen2.5-1.5B | bottom | 2 | 0.001 | 2-6 | 0.4576 | RTX 4000 Ada (5) | 1 |
| tab:answeronly, tab:central, tab:latescores, tab:placements | commonsense | Qwen2.5-1.5B | bottom | 2 | 0.002 | 2-6 | 0.4220 | GeForce RTX 3090 (2), RTX 4000 Ada (3) | 1 |
| tab:answeronly | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 7 | 0.001 | 2-6 | 0.5552 | RTX 4000 Ada (5) | 1 |
| tab:answeronly, tab:central, tab:latescores, tab:placements | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 7 | 0.002 | 2-6 | 0.5456 | RTX 4000 Ada (5) | 1 |
| tab:answeronly, tab:central, tab:v2budget | commonsense | Qwen2.5-1.5B | dual_bottom | 2 | 0.01 | 2-6 | 0.5248 | RTX 4000 Ada (5) | 1 |
| tab:answeronly | commonsense | Qwen2.5-1.5B | dual_bottom | 2 | 0.01 | 2-6 | 0.5548 | RTX 4000 Ada (5) | 1 |
| tab:answeronly, tab:central, tab:v2budget | commonsense | Qwen2.5-1.5B | dual_top | 2 | 0.005 | 2-6 | 0.5308 | RTX 4000 Ada (5) | 1 |
| tab:answeronly | commonsense | Qwen2.5-1.5B | dual_top | 2 | 0.005 | 2-6 | 0.5552 | RTX 4000 Ada (5) | 1 |
| tab:answeronly, tab:central, tab:latescores, tab:placements | commonsense | Qwen2.5-1.5B | free | 1 | 0.002 | 2-6 | 0.5456 | GeForce RTX 3090 (2), RTX 2000 Ada (1), RTX 4000 Ada (1), RTX A5000 (1) | 1 |
| tab:answeronly | commonsense | Qwen2.5-1.5B | free | 1 | 0.002 | 2-6 | 0.5572 | RTX 4000 Ada (5) | 1 |
| tab:answeronly, tab:central, tab:latescores, tab:placements | commonsense | Qwen2.5-1.5B | random | 2 | 0.005 | 2-6 | 0.4652 | GeForce RTX 3090 (3), RTX 4000 Ada (1), RTX A4000 (1) | 1 |
| tab:answeronly, tab:central, tab:latescores, tab:placements | commonsense | Qwen2.5-1.5B | top | 2 | 0.005 | 2-6 | 0.4652 | RTX 4000 Ada (5) | 1 |
| tab:answeronly | commonsense | Qwen2.5-1.5B | top | 2 | 0.005 | 2-6 | 0.4836 | RTX 4000 Ada (5) | 1 |
| tab:answeronly | commonsense | Qwen2.5-1.5B | top_unfrozen | 7 | 0.001 | 2-6 | 0.5552 | RTX 4000 Ada (5) | 1 |
| tab:answeronly | commonsense | Qwen2.5-1.5B | top_unfrozen | 7 | 0.002 | 2-6 | 0.5572 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Qwen2.5-1.5B | bottom | 2 | 0.0002 | 2-6 | 0.4764 | RTX 4000 Ada (4), RTX A5000 (1) | 1 |
| tab:central | commonsense | Qwen2.5-7B | bottom | 2 | 0.0005 | 2-6 | 0.4840 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Mistral-7B-v0.3 | bottom | 2 | 0.001 | 2-6 | 0.4776 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 1 | 0.002 | 2-6 | 0.5496 | RTX 4000 Ada (4), RTX A4000 (1) | 1 |
| tab:central | commonsense | Qwen2.5-7B | bottom_unfrozen | 1 | 0.002 | 2-6 | 0.6264 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Mistral-7B-v0.3 | bottom_unfrozen | 1 | 0.002 | 2-6 | 0.5924 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Qwen2.5-7B | bottom_unfrozen | 2 | 0.001 | 2-6 | 0.6312 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Mistral-7B-v0.3 | bottom_unfrozen | 2 | 0.001 | 2-6 | 0.6048 | RTX 4000 Ada (5) | 1 |
| tab:central, tab:mechreps, tab:placements | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 2 | 0.002 | 2-6 | 0.5528 | RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A5000 (2) | 1 |
| tab:central, tab:mechreps, tab:placements | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 16 | 0.001 | 2-6 | 0.5528 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Mistral-7B-v0.3 | dual_bottom | 2 | 0.001 | 2-6 | 0.5860 | RTX 4000 Ada (5) | 1 |
| tab:central, tab:v2budget | commonsense | Qwen2.5-7B | dual_bottom | 2 | 0.002 | 2-6 | 0.6148 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Mistral-7B-v0.3 | dual_top | 2 | 0.0005 | 2-6 | 0.6076 | RTX 4000 Ada (5) | 1 |
| tab:central, tab:v2budget | commonsense | Qwen2.5-7B | dual_top | 2 | 0.002 | 2-6 | 0.6364 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Mistral-7B-v0.3 | free | 1 | 0.001 | 2-6 | 0.6012 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Qwen2.5-7B | free | 1 | 0.002 | 2-6 | 0.6284 | RTX 4000 Ada (5) | 1 |
| tab:central, tab:v2trend | commonsense | Qwen2.5-1.5B | free | 2 | 0.005 | 2-6 | 0.5524 | RTX 4000 Ada (5) | 1 |
| tab:central, tab:mechreps, tab:placements | commonsense | Qwen2.5-1.5B | learned | 2 | 0.005 | 2-6 | 0.5528 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Mistral-7B-v0.3 | top | 2 | 0.001 | 2-6 | 0.4764 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Qwen2.5-7B | top | 2 | 0.005 | 2-6 | 0.5316 | RTX 4000 Ada (5) | 1 |
| tab:central, tab:v2trend | commonsense | Qwen2.5-1.5B | top_unfrozen | 1 | 0.002 | 2-6 | 0.5524 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Qwen2.5-7B | top_unfrozen | 1 | 0.002 | 2-6 | 0.6232 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Mistral-7B-v0.3 | top_unfrozen | 1 | 0.002 | 2-6 | 0.5924 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Qwen2.5-7B | top_unfrozen | 2 | 0.001 | 2-6 | 0.6292 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Mistral-7B-v0.3 | top_unfrozen | 2 | 0.001 | 2-6 | 0.5988 | RTX 4000 Ada (5) | 1 |
| tab:central | commonsense | Qwen2.5-1.5B | top_unfrozen | 2 | 0.002 | 2-6 | 0.5576 | RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A4000 (1), RTX A5000 (1) | 1 |
| tab:central | commonsense | Qwen2.5-1.5B | top_unfrozen | 16 | 0.0001 | 2-6 | 0.4840 | RTX 4000 Ada (5) | 1 |
| tab:central | format | Qwen2.5-0.5B | bottom | 2 | 0.002 | 2-6 | 0.4840 | RTX 2000 Ada (2), RTX 4000 Ada (2), RTX A5000 (1) | 1 |
| tab:central, tab:placements | format | Llama-3.2-1B | bottom | 2 | 0.005 | 2-6 | 0.5922 | (none) (3), RTX 4000 Ada (1), RTX A5000 (1) | 1 |
| tab:central, tab:latescores | format | Qwen2.5-1.5B | bottom | 2 | 0.005 | 22-26 | 0.6391 | RTX 4000 Ada (5) | 1 |
| tab:central, tab:latescores | format | Llama-3.2-1B | bottom | 2 | 0.005 | 2-6 | 0.6257 | RTX 4000 Ada (5) | 1 |
| tab:central | format | Llama-3.2-1B | bottom_unfrozen | 1 | 0.005 | 2-6 | 0.7954 | GeForce RTX 3090 (1), RTX 4000 Ada (4) | 1 |
| tab:central | format | Qwen2.5-1.5B | bottom_unfrozen | 1 | 0.005 | 22-26 | 0.7811 | RTX 4000 Ada (5) | 1 |
| tab:central | format | Llama-3.2-1B | bottom_unfrozen | 2 | 0.002 | 2-6 | 0.8072 | RTX 2000 Ada (1), RTX 4000 Ada (4) | 1 |
| tab:central | format | Llama-3.2-1B | bottom_unfrozen | 2 | 0.005 | 2-6 | 0.7896 | RTX 2000 Ada (2), RTX 4000 Ada (1), RTX A4000 (1), RTX A5000 (1) | 1 |
| tab:central | format | Qwen2.5-1.5B | bottom_unfrozen | 2 | 0.005 | 22-26 | 0.7857 | RTX 4000 Ada (5) | 1 |
| tab:central | format | Llama-3.2-1B | bottom_unfrozen | 16 | 0.002 | 2-6 | 0.7896 | RTX 4000 Ada (5) | 1 |
| tab:central | format | Llama-3.2-1B | dual_bottom | 2 | 0.01 | 2-6 | 0.7818 | GeForce RTX 3090 (1), RTX 2000 Ada (2), RTX 4000 Ada (1), RTX A4000 (1) | 1 |
| tab:central | format | Llama-3.2-1B | dual_bottom | 2 | 0.01 | 120-124 | 0.7818 | RTX 4000 Ada (5) | 1 |
| tab:central, tab:mechreps, tab:placements | format | Llama-3.2-1B | dual_random_ortho | 2 | 0.02 | 2-6 | 0.7684 | GeForce RTX 3090 (2), RTX 2000 Ada (2), RTX A5000 (1) | 1 |
| tab:central | format | Llama-3.2-1B | dual_top | 2 | 0.01 | 2-6 | 0.7739 | RTX 4000 Ada (5) | 1 |
| tab:central, tab:mechreps, tab:placements | format | Llama-3.2-1B | free | 1 | 0.001 | 2-6 | 0.7684 | GeForce RTX 3090 (3), RTX 2000 Ada (1), RTX 4000 Ada (1) | 1 |
| tab:central | format | Llama-3.2-1B | free | 1 | 0.002 | 2-6 | 0.7811 | RTX 4000 Ada (5) | 1 |
| tab:central, tab:latescores | format | Qwen2.5-1.5B | free | 1 | 0.005 | 22-26 | 0.7879 | RTX 4000 Ada (5) | 1 |
| tab:central, tab:latescores | format | Llama-3.2-1B | random | 2 | 0.002 | 2-6 | 0.6257 | GeForce RTX 3090 (1), RTX 4000 Ada (4) | 1 |
| tab:central, tab:mechreps, tab:placements | format | Llama-3.2-1B | top | 2 | 0.01 | 2-6 | 0.6443 | (none) (3), RTX 4000 Ada (2) | 1 |
| tab:central, tab:latescores | format | Qwen2.5-1.5B | top | 2 | 0.01 | 22-26 | 0.6257 | RTX 4000 Ada (5) | 1 |
| tab:central | format | Llama-3.2-1B | top_unfrozen | 1 | 0.005 | 2-6 | 0.7928 | GeForce RTX 3090 (1), RTX 4000 Ada (3), RTX A4000 (1) | 1 |
| tab:central | format | Qwen2.5-1.5B | top_unfrozen | 1 | 0.005 | 22-26 | 0.7896 | RTX 4000 Ada (5) | 1 |
| tab:central | format | Llama-3.2-1B | top_unfrozen | 2 | 0.005 | 2-6 | 0.8000 | GeForce RTX 3090 (1), RTX 4000 Ada (4) | 1 |
| tab:central | format | Qwen2.5-1.5B | top_unfrozen | 2 | 0.005 | 22-26 | 0.7775 | RTX 4000 Ada (5) | 1 |
| tab:grad, tab:placements | commonsense | Qwen2.5-1.5B | learned | 2 | 0.002 | 2-6 | 0.4272 | RTX 4000 Ada (5) | 1 |
| tab:grad, tab:placements | format | Llama-3.2-1B | learned | 2 | 0.005 | 2-6 | 0.6658 | RTX 4000 Ada (5) | 1 |
| tab:grad, tab:mechreps | format | Llama-3.2-1B | top | 2 | 0.01 | 2-6 | 0.6430 | RTX 4000 Ada (5) | 1 |
| tab:grad, tab:placements | format | Llama-3.2-1B | top_scalar | 2 | 0.005 | 50-64 | 0.6658 | GeForce RTX 3090 (3), RTX 2000 Ada (1), RTX 4000 Ada (9), RTX A4000 (1), RTX A5000 (1) | 1 |
| tab:grad, tab:placements | format | Llama-3.2-1B | top_scalar | 2 | 0.005 | 65-104 | 0.6658 | GeForce RTX 3090 (18), RTX 2000 Ada (4), RTX 4000 Ada (13), RTX A4000 (5) | 1 |
| tab:gradacc, tab:mechreps, tab:warm | commonsense | Qwen2.5-1.5B | bottom | 2 | 0.002 | 2-6 | 0.4276 | RTX 4000 Ada (5) | 1 |
| tab:gradacc, tab:mechreps, tab:warm | commonsense | Qwen2.5-1.5B | free | 1 | 0.002 | 2-6 | 0.5484 | RTX 4000 Ada (5) | 1 |
| tab:gradacc | commonsense | Qwen2.5-1.5B | learned | 2 | 0.002 | 2-6 | 0.4304 | RTX 4000 Ada (5) | 1 |
| tab:gradacc, tab:mechreps, tab:warm | format | Llama-3.2-1B | free | 1 | 0.001 | 2-6 | 0.7687 | RTX 4000 Ada (5) | 1 |
| tab:gradacc | format | Llama-3.2-1B | learned | 2 | 0.005 | 2-6 | 0.6622 | RTX 4000 Ada (5) | 1 |
| tab:gradacc, tab:warm | format | Llama-3.2-1B | top_sigma | 2 | 0.005 | 50-64 | 0.6631 | GeForce RTX 3090 (6), RTX 4000 Ada (7), RTX A5000 (2) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | bottom | 2 | 0.002 | 2-6 | 0.4668 | GeForce RTX 3090 (1), RTX 4000 Ada (4) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | bottom | 16 | 0.0005 | 2-6 | 0.7792 | RTX 4000 Ada (3), RTX A4000 (2) | 1 |
| tab:latescores, tab:rank16 | commonsense | Qwen2.5-1.5B | bottom | 16 | 0.001 | 2-6 | 0.4984 | GeForce RTX 3090 (3), RTX 4000 Ada (2) | 1 |
| tab:latescores, tab:rank16 | commonsense | Qwen2.5-1.5B | bottom | 16 | 0.001 | 2-6 | 0.4452 | GeForce RTX 3090 (1), RTX 4000 Ada (2), RTX A4000 (1), RTX A5000 (1) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | bottom | 32 | 0.0002 | 2-6 | 0.4758 | RTX 4000 Ada (2), RTX A4000 (2), RTX A5000 (1) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 16 | 0.0005 | 2-6 | 0.4692 | RTX 4000 Ada (5) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | free | 1 | 0.0005 | 2-6 | 0.4960 | RTX 4000 Ada (4), RTX A4000 (1) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | free | 1 | 0.001 | 2-6 | 0.5044 | RTX 4000 Ada (4), RTX A5000 (1) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | free | 7 | 0.0001 | 2-6 | 0.7940 | RTX 2000 Ada (1), RTX 4000 Ada (4) | 1 |
| tab:latescores, tab:rank16 | commonsense | Qwen2.5-1.5B | free | 7 | 0.0005 | 2-6 | 0.4952 | RTX 4000 Ada (4), RTX A4000 (1) | 1 |
| tab:latescores, tab:rank16 | commonsense | Qwen2.5-1.5B | free | 7 | 0.002 | 2-6 | 0.5616 | RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A5000 (2) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | free | 14 | 0.0005 | 2-6 | 0.4888 | RTX 4000 Ada (4), RTX A5000 (1) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | random_ortho | 2 | 0.005 | 2-6 | 0.4692 | RTX 2000 Ada (1), RTX 4000 Ada (4) | 1 |
| tab:latescores, tab:placements | commonsense | Qwen2.5-1.5B | random_ortho | 2 | 0.02 | 2-6 | 0.4592 | RTX 4000 Ada (5) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | random_ortho | 16 | 0.0001 | 2-6 | 0.7948 | RTX 2000 Ada (1), RTX 4000 Ada (4) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | random_ortho | 16 | 0.0005 | 2-6 | 0.7970 | RTX 4000 Ada (5) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | random_ortho | 16 | 0.001 | 2-6 | 0.4860 | RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A4000 (1), RTX A5000 (1) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | random_ortho | 32 | 0.0005 | 2-6 | 0.4960 | GeForce RTX 3090 (2), RTX 4000 Ada (3) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | top | 2 | 0.002 | 2-6 | 0.4888 | GeForce RTX 3090 (1), RTX 4000 Ada (3), RTX A5000 (1) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | top | 2 | 0.005 | 2-6 | 0.4668 | RTX 2000 Ada (1), RTX 4000 Ada (3), RTX A4000 (1) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | top | 2 | 0.005 | 2-6 | 0.4842 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (3) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | top | 16 | 0.0001 | 2-6 | 0.7948 | RTX 2000 Ada (2), RTX 4000 Ada (2), RTX A5000 (1) | 1 |
| tab:latescores, tab:rank16 | commonsense | Qwen2.5-1.5B | top | 16 | 0.0005 | 2-6 | 0.4970 | GeForce RTX 3090 (1), RTX 4000 Ada (3), RTX A4000 (1) | 1 |
| tab:latescores, tab:rank16 | commonsense | Qwen2.5-1.5B | top | 16 | 0.0005 | 2-6 | 0.4984 | RTX 4000 Ada (4), RTX A4000 (1) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | top | 16 | 0.0005 | 2-6 | 0.4960 | GeForce RTX 3090 (1), RTX 4000 Ada (3), RTX A4000 (1) | 1 |
| tab:latescores, tab:rank16 | commonsense | Qwen2.5-1.5B | top | 16 | 0.002 | 2-6 | 0.5408 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (3) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | top | 32 | 0.0005 | 2-6 | 0.4824 | GeForce RTX 3090 (1), RTX 4000 Ada (3), RTX A5000 (1) | 1 |
| tab:latescores | commonsense | Qwen2.5-1.5B | top_unfrozen | 2 | 0.01 | 2-6 | 0.4824 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A4000 (1) | 1 |
| tab:latescores | format | Llama-3.2-1B | bottom | 2 | 0.005 | 2-6 | 0.6313 | RTX 4000 Ada (5) | 1 |
| tab:latescores | format | Llama-3.2-1B | bottom | 8 | 0.002 | 42-46 | 0.7450 | RTX 4000 Ada (5) | 1 |
| tab:latescores, tab:rank16 | format | Llama-3.2-1B | bottom | 16 | 0.0005 | 2-6 | 0.7554 | GeForce RTX 3090 (1), RTX 2000 Ada (2), RTX 4000 Ada (1), RTX A5000 (1) | 1 |
| tab:latescores | format | Llama-3.2-1B | bottom | 16 | 0.0005 | 2-6 | 0.7524 | GeForce RTX 3090 (1), RTX 4000 Ada (4) | 1 |
| tab:latescores | format | Qwen2.5-1.5B | bottom | 16 | 0.001 | 22-26 | 0.7524 | RTX 4000 Ada (5) | 1 |
| tab:latescores | format | Llama-3.2-1B | bottom_unfrozen | 16 | 0.0001 | 2-6 | 0.6896 | RTX 4000 Ada (5) | 1 |
| tab:latescores | format | Llama-3.2-1B | free | 1 | 0.001 | 2-6 | 0.7984 | RTX 2000 Ada (1), RTX 4000 Ada (4) | 1 |
| tab:latescores | format | Llama-3.2-1B | free | 4 | 0.002 | 42-46 | 0.8293 | GeForce RTX 3090 (1), RTX 4000 Ada (3), RTX A5000 (1) | 1 |
| tab:latescores | format | Qwen2.5-1.5B | free | 7 | 0.002 | 22-26 | 0.8078 | RTX 4000 Ada (5) | 1 |
| tab:latescores, tab:rank16 | format | Llama-3.2-1B | free | 8 | 0.002 | 2-6 | 0.8205 | GeForce RTX 3090 (2), RTX 4000 Ada (1), RTX A4000 (1), RTX A5000 (1) | 1 |
| tab:latescores | format | Llama-3.2-1B | random | 2 | 0.01 | 2-6 | 0.6935 | RTX 2000 Ada (3), RTX 4000 Ada (1), RTX A5000 (1) | 1 |
| tab:latescores | format | Llama-3.2-1B | random | 8 | 0.005 | 42-46 | 0.7550 | GeForce RTX 3090 (1), RTX 4000 Ada (4) | 1 |
| tab:latescores | format | Llama-3.2-1B | random | 16 | 0.001 | 2-6 | 0.7700 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A5000 (1) | 1 |
| tab:latescores | format | Llama-3.2-1B | random_ortho | 2 | 0.01 | 2-6 | 0.6896 | RTX 2000 Ada (1), RTX 4000 Ada (3), RTX A5000 (1) | 1 |
| tab:latescores | format | Qwen2.5-1.5B | random_ortho | 2 | 0.02 | 22-26 | 0.6238 | RTX 4000 Ada (5) | 1 |
| tab:latescores | format | Llama-3.2-1B | random_ortho | 8 | 0.005 | 42-46 | 0.7505 | GeForce RTX 3090 (1), RTX 2000 Ada (2), RTX 4000 Ada (1), RTX A4000 (1) | 1 |
| tab:latescores | format | Llama-3.2-1B | random_ortho | 16 | 0.001 | 2-6 | 0.7795 | RTX 2000 Ada (1), RTX 4000 Ada (3), RTX A4000 (1) | 1 |
| tab:latescores | format | Qwen2.5-1.5B | random_ortho | 16 | 0.005 | 22-26 | 0.7521 | RTX 4000 Ada (5) | 1 |
| tab:latescores | format | Llama-3.2-1B | top | 2 | 0.01 | 2-6 | 0.6967 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (3) | 1 |
| tab:latescores | format | Llama-3.2-1B | top | 4 | 0.01 | 2-6 | 0.6896 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (2), RTX A5000 (1) | 1 |
| tab:latescores | format | Llama-3.2-1B | top | 8 | 0.002 | 42-46 | 0.7450 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (3) | 1 |
| tab:latescores, tab:rank16 | format | Llama-3.2-1B | top | 16 | 0.002 | 2-6 | 0.7756 | RTX 4000 Ada (4), RTX A4000 (1) | 1 |
| tab:latescores | format | Qwen2.5-1.5B | top | 16 | 0.002 | 22-26 | 0.7430 | RTX 4000 Ada (5) | 1 |
| tab:latescores | format | Llama-3.2-1B | top_unfrozen | 1 | 0.005 | 120-124 | 0.8078 | RTX 4000 Ada (5) | 1 |
| tab:latescores | format | Llama-3.2-1B | top_unfrozen | 16 | 0.0002 | 2-6 | 0.7984 | RTX 4000 Ada (5) | 1 |
| tab:mechreps, tab:placements | commonsense | Qwen2.5-1.5B | learned | 2 | 0.002 | 2-6 | 0.5588 | RTX 4000 Ada (5) | 1 |
| tab:mechreps, tab:placements | format | Llama-3.2-1B | bottom_unfrozen | 2 | 0.002 | 120-124 | 0.8081 | RTX 4000 Ada (5) | 1 |
| tab:mechreps, tab:placements | format | Llama-3.2-1B | bottom_unfrozen | 2 | 0.002 | 2-6 | 0.8081 | RTX 4000 Ada (5) | 1 |
| tab:mechreps, tab:placements | format | Llama-3.2-1B | learned | 2 | 0.005 | 2-6 | 0.8081 | RTX 4000 Ada (5) | 1 |
| tab:placements | format | Llama-3.2-1B | bottom_unfrozen | 8 | 0.005 | 2-6 | 0.6534 | RTX 4000 Ada (5) | 1 |
| tab:placements | format | Llama-3.2-1B | random_ortho | 2 | 0.01 | 2-6 | 0.6534 | GeForce RTX 3090 (1), RTX 4000 Ada (2), RTX A4000 (2) | 1 |
| tab:possweep | format | Llama-3.2-1B | band | 2 | 0.005 | 2-8 | 0.5928 | RTX 2000 Ada (2), RTX 4000 Ada (5) | 1 |
| tab:possweep | format | Llama-3.2-1B | band | 2 | 0.01 | 2-8 | 0.6547 | RTX 2000 Ada (2), RTX 4000 Ada (1), RTX A4000 (2), RTX A5000 (2) | 1 |
| tab:possweep | format | Llama-3.2-1B | band | 2 | 0.01 | 2-8 | 0.6368 | RTX 4000 Ada (6), RTX A5000 (1) | 1 |
| tab:possweep | format | Llama-3.2-1B | band | 2 | 0.01 | 2-8 | 0.6768 | RTX 4000 Ada (6), RTX A5000 (1) | 1 |
| tab:possweep | format | Llama-3.2-1B | band | 2 | 0.01 | 2-8 | 0.6671 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (1), RTX A4000 (3), RTX A5000 (1) | 1 |
| tab:possweep | format | Llama-3.2-1B | random | 4 | 0.002 | 2-6 | 0.6671 | GeForce RTX 3090 (1), RTX 4000 Ada (1), RTX A4000 (2), RTX A5000 (1) | 1 |
| tab:possweep | format | Qwen2.5-1.5B | top | 16 | 0.01 | 22-26 | 0.5928 | RTX 4000 Ada (5) | 1 |
| tab:v2budget | commonsense | Qwen2.5-7B | bottom | 3 | 0.0005 | 2-6 | 0.4924 | RTX 4000 Ada (5) | 1 |
| tab:v2budget | commonsense | Qwen2.5-1.5B | bottom | 3 | 0.002 | 2-6 | 0.4328 | RTX 4000 Ada (5) | 1 |
| tab:v2budget | commonsense | Qwen2.5-7B | bottom | 4 | 0.001 | 2-6 | 0.5136 | RTX 4000 Ada (5) | 1 |
| tab:v2budget, tab:v2trend | commonsense | Qwen2.5-1.5B | bottom | 4 | 0.002 | 2-6 | 0.4524 | RTX 4000 Ada (5) | 1 |
| tab:v2budget | commonsense | Qwen2.5-1.5B | dual_random_ortho | 2 | 0.01 | 2-6 | 0.5464 | RTX 4000 Ada (5) | 1 |
| tab:v2budget, tab:v2trend | commonsense | Qwen2.5-1.5B | top | 2 | 0.005 | 2-6 | 0.4644 | GeForce RTX 3090 (1), RTX 2000 Ada (1), RTX 4000 Ada (3) | 1 |
| tab:v2budget | commonsense | Qwen2.5-7B | top | 3 | 0.002 | 2-6 | 0.5464 | RTX 4000 Ada (5) | 1 |
| tab:v2budget | commonsense | Qwen2.5-1.5B | top | 3 | 0.005 | 2-6 | 0.4680 | RTX 4000 Ada (5) | 1 |
| tab:v2budget, tab:v2trend | commonsense | Qwen2.5-1.5B | top | 4 | 0.005 | 2-6 | 0.4644 | RTX 4000 Ada (5) | 1 |
| tab:v2budget | commonsense | Qwen2.5-7B | top | 4 | 0.005 | 2-6 | 0.5464 | RTX 4000 Ada (5) | 1 |
| tab:v2budget, tab:v2trend | format | Qwen2.5-0.5B | band | 2 | 0.01 | 9-16 | 0.4644 | RTX 2000 Ada (1), RTX 4000 Ada (6), RTX A4000 (1) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | bottom | 8 | 0.001 | 2-6 | 0.4656 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-7B | bottom | 16 | 0.0005 | 2-6 | 0.5676 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 2 | 0.002 | 2-6 | 0.5520 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 4 | 0.002 | 2-6 | 0.5420 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-7B | bottom_unfrozen | 7 | 0.0005 | 2-6 | 0.6276 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 8 | 0.001 | 2-6 | 0.5556 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-7B | bottom_unfrozen | 16 | 0.0005 | 2-6 | 0.6200 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | dual_bottom | 4 | 0.002 | 2-6 | 0.5388 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | dual_bottom | 8 | 0.002 | 2-6 | 0.5344 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-7B | dual_bottom | 16 | 0.0005 | 2-6 | 0.6256 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | dual_top | 4 | 0.005 | 2-6 | 0.5344 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | dual_top | 8 | 0.002 | 2-6 | 0.5628 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-7B | dual_top | 16 | 0.001 | 2-6 | 0.6240 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | dual_top | 16 | 0.002 | 2-6 | 0.5544 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | free | 4 | 0.002 | 2-6 | 0.5544 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-7B | free | 7 | 0.0005 | 2-6 | 0.6320 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | top | 2 | 0.005 | 2-6 | 0.4656 | RTX 2000 Ada (1), RTX 4000 Ada (4) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | top | 8 | 0.005 | 2-6 | 0.4992 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-7B | top | 16 | 0.001 | 2-6 | 0.6020 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | top_unfrozen | 2 | 0.002 | 2-6 | 0.5540 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | top_unfrozen | 4 | 0.001 | 2-6 | 0.5504 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-7B | top_unfrozen | 7 | 0.0005 | 2-6 | 0.6248 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-1.5B | top_unfrozen | 8 | 0.001 | 2-6 | 0.5636 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | commonsense | Qwen2.5-7B | top_unfrozen | 16 | 0.0005 | 2-6 | 0.6332 | RTX 4000 Ada (5) | 1 |
| tab:v2trend | format | Llama-3.2-1B | band | 2 | 0.01 | 2-6 | 0.6332 | RTX 4000 Ada (4), RTX A5000 (1) | 1 |
| tab:warm | commonsense | Qwen2.5-1.5B | bottom_unfrozen | 1 | 0.01 | 2-6 | 0.5116 | RTX 4000 Ada (4), RTX A4000 (1) | 1 |
| tab:warm | commonsense | Qwen2.5-1.5B | learned | 2 | 0.005 | 2-6 | 0.5116 | RTX 4000 Ada (5) | 1 |
| tab:warm | format | Llama-3.2-1B | learned | 2 | 0.005 | 2-6 | 0.7221 | RTX 4000 Ada (5) | 1 |
