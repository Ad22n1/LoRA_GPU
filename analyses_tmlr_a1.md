# A1 — Holm over the 145 tests (post hoc, descriptive; definitions: ANALYSES_tmlr.md)

Tests: 145; established by the paper: 110 (of which 1 outside Holm, not a test).
**After Holm over 145 at 0.05: 46 stay established, 63 no longer established.**

Check of the transcription: 1 row(s) whose conversion does not reproduce the paper's own verdict at its own level:

- row 88 (PR NE; PR E): p = 0.0446 against its own level 0.025 — two tests in one row: cost (t > 3.495, NE) and bands (t > 2.776, E)

## Established results that no longer hold after Holm

- row 1 (PH; per comparison (tab:bandqwen)): p = 0.0106, Holm-adjusted 0.651
- row 4 (PR; per comparison (tab:bandqwen)): p = 0.00891, Holm-adjusted 0.57
- row 6 (PH E): p = 0.008, Holm-adjusted 0.552
- row 12 (PH; per comparison (tab:bandqwen)): p = 0.000601, Holm-adjusted 0.0595
- row 13 (PR E (marginal)): p = 0.0336, Holm-adjusted 1
- row 15 (PH E (vs top)): p = 0.0118, Holm-adjusted 0.695
- row 16 (PH E): p = 0.00531, Holm-adjusted 0.409
- row 19 (PH E): p = 0.00474, Holm-adjusted 0.37
- row 23 (PR E): p = 0.0067, Holm-adjusted 0.502
- row 24 (PR E): p = 0.000816, Holm-adjusted 0.0742
- row 25 (PR E): p = 0.000961, Holm-adjusted 0.0855
- row 27 (PH E): p = 0.00132, Holm-adjusted 0.115
- row 32 (PH E): p = 0.00064, Holm-adjusted 0.0621
- row 34 (PH E): p = 0.0255, Holm-adjusted 1
- row 38 (PH E): p = 0.0196, Holm-adjusted 0.86
- row 39 (PH E): p = 0.00904, Holm-adjusted 0.57
- row 40 (PH; E for random_ortho and free): p = 0.00834, Holm-adjusted 0.552
- row 41 (PH; E for random and free): p = 0.00254, Holm-adjusted 0.206
- row 43 (PH E): p = 0.00804, Holm-adjusted 0.552
- row 44 (PH E): p = 0.00804, Holm-adjusted 0.552
- row 46 (PH E): p = 0.0135, Holm-adjusted 0.734
- row 49 (PR E): p = 0.0294, Holm-adjusted 1
- row 57 (PR E): p = 0.0286, Holm-adjusted 1
- row 63 (PR E (equivalence)): p = 0.00359, Holm-adjusted 0.288
- row 69 (PR E (reversed)): p = 0.0125, Holm-adjusted 0.701
- row 70 (PH E): p = 0.00771, Holm-adjusted 0.539
- row 73 (PH E): p = 0.0174, Holm-adjusted 0.818
- row 75 (PR E): p = 0.00552, Holm-adjusted 0.419
- row 77 (PR E): p = 0.0146, Holm-adjusted 0.756
- row 78 (PR E): p = 0.000799, Holm-adjusted 0.0735
- row 79 (PR E): p = 0.000642, Holm-adjusted 0.0621
- row 80 (PR E): p = 0.00859, Holm-adjusted 0.558
- row 81 (PR E (difference)): p = 0.0441, Holm-adjusted 1
- row 85 (PR E / inconclusive): p = 0.0145, Holm-adjusted 0.756
- row 86 (PR E (top)): p = 0.0191, Holm-adjusted 0.86
- row 87 (PR E, fragile): p = 0.00197, Holm-adjusted 0.164
- row 88 (PR NE; PR E): p = 0.0446, Holm-adjusted 1
- row 91 (PR E): p = 0.000877, Holm-adjusted 0.0789
- row 92 (PR E): p = 0.00669, Holm-adjusted 0.502
- row 96 (PR E): p = 0.00413, Holm-adjusted 0.326
- row 99 (PR E): p = 0.00703, Holm-adjusted 0.506
- row 101 (PR E): p = 0.017, Holm-adjusted 0.814
- row 103 (PR NE / E): p = 0.0014, Holm-adjusted 0.119
- row 104 (PR E): p = 0.00118, Holm-adjusted 0.104
- row 105 (PR E (marginal)): p = 0.0403, Holm-adjusted 1
- row 106 (PR E): p = 0.0105, Holm-adjusted 0.651
- row 107 (PR E): p = 0.012, Holm-adjusted 0.697
- row 114 (PR E): p = 0.000671, Holm-adjusted 0.0631
- row 115 (PR E): p = 0.0265, Holm-adjusted 1
- row 118 (PR E): p = 0.00754, Holm-adjusted 0.536
- row 120 (PH E): p = 0.0114, Holm-adjusted 0.685
- row 126 (PR E): p = 0.000694, Holm-adjusted 0.0646
- row 128 (PR E): p = 0.00063, Holm-adjusted 0.0617
- row 130 (PR E): p = 0.0133, Holm-adjusted 0.734
- row 131 (PR E / inconclusive): p = 0.0122, Holm-adjusted 0.697
- row 132 (PR E): p = 0.00147, Holm-adjusted 0.124
- row 135 (PR E): p = 0.00211, Holm-adjusted 0.173
- row 140 (PR E): p = 0.00684, Holm-adjusted 0.502
- row 141 (PR E): p = 0.0156, Holm-adjusted 0.78
- row 143 (PR E (marginal)): p = 0.0388, Holm-adjusted 1
- row 146 (PR does not suffice): p = 0.0192, Holm-adjusted 0.86
- row 150 (PR E (gain) / inconclusive): p = 0.000655, Holm-adjusted 0.0623
- row 151 (PR E): p = 0.00135, Holm-adjusted 0.116

## Rows under the approximate (bootstrap or interval) rule with an adjusted p between 0.01 and 0.1

- row 32 (PH E): Holm-adjusted 0.0621
- row 33 (PH E): Holm-adjusted 0.032
- row 64 (PR E): Holm-adjusted 0.0303
- row 65 (PR E): Holm-adjusted 0.0166
- row 74 (PR E): Holm-adjusted 0.0488
- row 114 (PR E): Holm-adjusted 0.0631

## Every row

| row | paper's status | class | p | Holm-adjusted | after Holm |
|---|---|---|---|---|---|
| 1 | PH; per comparison (tab:bandqwen) | t | 0.0106 | 0.651 | no longer established |
| 2 | PH NE | p | 0.016 | 0.784 | not established |
| 3 | PH NE | t | 0.0259 | 1 | not established |
| 4 | PR; per comparison (tab:bandqwen) | t | 0.00891 | 0.57 | no longer established |
| 5 | PR NE | t | 0.0143 | 0.755 | not established |
| 6 | PH E | p | 0.008 | 0.552 | no longer established |
| 7 | PH NE | p | 0.035 | 1 | not established |
| 10 | PH NE | t | 0.148 | 1 | not established |
| 12 | PH; per comparison (tab:bandqwen) | t | 0.000601 | 0.0595 | no longer established |
| 13 | PR E (marginal) | t | 0.0336 | 1 | no longer established |
| 14 | PR NE | t | 0.233 | 1 | not established |
| 15 | PH E (vs top) | t | 0.0118 | 0.695 | no longer established |
| 16 | PH E | ci | 0.00531 | 0.409 | no longer established |
| 17 | PH NE | t | 0.0196 | 0.86 | not established |
| 18 | PH NE | t | 0.325 | 1 | not established |
| 19 | PH E | t | 0.00474 | 0.37 | no longer established |
| 20 | PH E | t | 6.8e-05 | 0.00829 | stays established |
| 21 | PR NE | t | 0.0728 | 1 | not established |
| 22 | PR E | t | 0.000175 | 0.0203 | stays established |
| 23 | PR E | t | 0.0067 | 0.502 | no longer established |
| 24 | PR E | t | 0.000816 | 0.0742 | no longer established |
| 25 | PR E | t | 0.000961 | 0.0855 | no longer established |
| 26 | PR NE | t | 0.0991 | 1 | not established |
| 27 | PH E | t | 0.00132 | 0.115 | no longer established |
| 28 | PR E | none | 1 | 1 | outside Holm (not a test) |
| 29 | PR E | t | 0.000424 | 0.0429 | stays established |
| 30 | PR E | t | 0.000379 | 0.039 | stays established |
| 31 | PR NE | ci | 0.131 | 1 | not established |
| 32 | PH E | ci | 0.00064 | 0.0621 | no longer established |
| 33 | PH E | ci | 0.000302 | 0.032 | stays established |
| 34 | PH E | ci | 0.0255 | 1 | no longer established |
| 35 | PH NE | ci | 0.962 | 1 | not established |
| 36 | PR E | ci | 1.46e-05 | 0.0019 | stays established |
| 37 | PH NE | t | 0.0626 | 1 | not established |
| 38 | PH E | t | 0.0196 | 0.86 | no longer established |
| 39 | PH E | t | 0.00904 | 0.57 | no longer established |
| 40 | PH; E for random_ortho and free | t | 0.00834 | 0.552 | no longer established |
| 41 | PH; E for random and free | t | 0.00254 | 0.206 | no longer established |
| 42 | PH NE | t | 0.0947 | 1 | not established |
| 43 | PH E | t | 0.00804 | 0.552 | no longer established |
| 44 | PH E | t | 0.00804 | 0.552 | no longer established |
| 45 | PH NE | p | 0.15 | 1 | not established |
| 46 | PH E | t | 0.0135 | 0.734 | no longer established |
| 47 | PH E | t | 0.000215 | 0.0246 | stays established |
| 48 | PH NE | t | 0.801 | 1 | not established |
| 49 | PR E | t | 0.0294 | 1 | no longer established |
| 50 | PR NE | t | 0.0916 | 1 | not established |
| 51 | PR NE | t | 0.0483 | 1 | not established |
| 52 | PR NE | ci | 0.229 | 1 | not established |
| 53 | PR E | ci | 1.46e-06 | 0.000203 | stays established |
| 54 | PR NE | t | 0.0186 | 0.856 | not established |
| 55 | PR E | ci | 1.24e-05 | 0.00164 | stays established |
| 56 | PR NE | t | 1 | 1 | not established |
| 57 | PR E | ci | 0.0286 | 1 | no longer established |
| 58 | PR NE | t | 1 | 1 | not established |
| 59 | PR E (reversed) | ci | 3.75e-08 | 5.4e-06 | stays established |
| 60 | PR E (reversed) | ci | 7.22e-07 | 0.000101 | stays established |
| 62 | PR NE | tost | 0.0855 | 1 | not established |
| 63 | PR E (equivalence) | tost | 0.00359 | 0.288 | no longer established |
| 64 | PR E | ci | 0.000283 | 0.0303 | stays established |
| 65 | PR E | ci | 0.00014 | 0.0166 | stays established |
| 66 | PR E | t | 4.36e-05 | 0.00554 | stays established |
| 67 | PR E | t | 6.57e-06 | 0.00088 | stays established |
| 68 | PR NE | ci | 0.773 | 1 | not established |
| 69 | PR E (reversed) | ci | 0.0125 | 0.701 | no longer established |
| 70 | PH E | t | 0.00771 | 0.539 | no longer established |
| 71 | PR E | t | 0.000357 | 0.0371 | stays established |
| 72 | PR E | t | 3.56e-06 | 0.000487 | stays established |
| 73 | PH E | t | 0.0174 | 0.818 | no longer established |
| 74 | PR E | ci | 0.000488 | 0.0488 | stays established |
| 75 | PR E | ci | 0.00552 | 0.419 | no longer established |
| 76 | PR E | t | 0.000215 | 0.0246 | stays established |
| 77 | PR E | t | 0.0146 | 0.756 | no longer established |
| 78 | PR E | t | 0.000799 | 0.0735 | no longer established |
| 79 | PR E | t | 0.000642 | 0.0621 | no longer established |
| 80 | PR E | t | 0.00859 | 0.558 | no longer established |
| 81 | PR E (difference) | t | 0.0441 | 1 | no longer established |
| 82 | PR E | ci | 0 | 0 | stays established |
| 83 | PR NE | ci | 0.0986 | 1 | not established |
| 85 | PR E / inconclusive | ci | 0.0145 | 0.756 | no longer established |
| 86 | PR E (top) | t | 0.0191 | 0.86 | no longer established |
| 87 | PR E, fragile | t | 0.00197 | 0.164 | no longer established |
| 88 | PR NE; PR E | t | 0.0446 | 1 | no longer established |
| 89 | PR E | t | 0.000269 | 0.0291 | stays established |
| 90 | PR E | t | 0.000154 | 0.018 | stays established |
| 91 | PR E | t | 0.000877 | 0.0789 | no longer established |
| 92 | PR E | ci | 0.00669 | 0.502 | no longer established |
| 93 | PR E | t | 0.000233 | 0.0261 | stays established |
| 94 | PR below floor | none | 1 | 1 | outside Holm (not a test) |
| 95 | PR E | t | 0.000254 | 0.0277 | stays established |
| 96 | PR E | t | 0.00413 | 0.326 | no longer established |
| 97 | PR not interpretable / E | t | 4.71e-06 | 0.000641 | stays established |
| 98 | PR not read | none | 1 | 1 | outside Holm (not a test) |
| 99 | PR E | t | 0.00703 | 0.506 | no longer established |
| 100 | PR inconclusive | ci | 0.0658 | 1 | not established |
| 101 | PR E | t | 0.017 | 0.814 | no longer established |
| 102 | PR NE / E | t | 0.000242 | 0.0268 | stays established |
| 103 | PR NE / E | t | 0.0014 | 0.119 | no longer established |
| 104 | PR E | t | 0.00118 | 0.104 | no longer established |
| 105 | PR E (marginal) | t | 0.0403 | 1 | no longer established |
| 106 | PR E | ci | 0.0105 | 0.651 | no longer established |
| 107 | PR E | ci | 0.012 | 0.697 | no longer established |
| 108 | PR E | t | 0.000324 | 0.034 | stays established |
| 109 | PR E | t | 5.09e-05 | 0.00626 | stays established |
| 110 | PR NE | t | 0.744 | 1 | not established |
| 111 | PR not conclusive | none | 1 | 1 | outside Holm (not a test) |
| 112 | PR E | t | 9.26e-06 | 0.00123 | stays established |
| 113 | PR E | t | 2.15e-07 | 3.07e-05 | stays established |
| 114 | PR E | ci | 0.000671 | 0.0631 | no longer established |
| 115 | PR E | t | 0.0265 | 1 | no longer established |
| 116 | PR E | t | 6.86e-05 | 0.0083 | stays established |
| 117 | PR E | t | 1.4e-05 | 0.00183 | stays established |
| 118 | PR E | t | 0.00754 | 0.536 | no longer established |
| 120 | PH E | t | 0.0114 | 0.685 | no longer established |
| 121 | PH E | cis | 1.51e-05 | 0.00195 | stays established |
| 122 | PH E | ci | 5.33e-06 | 0.000719 | stays established |
| 123 | PH E | cis | 3.47e-07 | 4.93e-05 | stays established |
| 124 | PH E | cis | 3.39e-06 | 0.000467 | stays established |
| 125 | PH E / NE / E | cis | 3.81e-07 | 5.37e-05 | stays established |
| 126 | PR E | conj | 0.000694 | 0.0646 | no longer established |
| 127 | PR E | conj | 4.55e-05 | 0.00569 | stays established |
| 128 | PR E | t | 0.00063 | 0.0617 | no longer established |
| 129 | PR E | t | 2.27e-05 | 0.00291 | stays established |
| 130 | PR E | ci | 0.0133 | 0.734 | no longer established |
| 131 | PR E / inconclusive | ci | 0.0122 | 0.697 | no longer established |
| 132 | PR E | t | 0.00147 | 0.124 | no longer established |
| 133 | PR E | t | 4.42e-05 | 0.00557 | stays established |
| 134 | PR E | t | 0.000251 | 0.0276 | stays established |
| 135 | PR E | t | 0.00211 | 0.173 | no longer established |
| 136 | PR below floor | none | 1 | 1 | outside Holm (not a test) |
| 137 | PR E | t | 0.00038 | 0.039 | stays established |
| 138 | PR NE | t | 0.663 | 1 | not established |
| 139 | PR inconclusive | ci | 0.0289 | 1 | not established |
| 140 | PR E | t | 0.00684 | 0.502 | no longer established |
| 141 | PR E | t | 0.0156 | 0.78 | no longer established |
| 142 | PR NE | t | 0.133 | 1 | not established |
| 143 | PR E (marginal) | t | 0.0388 | 1 | no longer established |
| 144 | PR E | conj | 0.000193 | 0.0222 | stays established |
| 145 | PR does not suffice | ci | 7.81e-05 | 0.00929 | stays established |
| 146 | PR does not suffice | ci | 0.0192 | 0.86 | no longer established |
| 147 | PR does not suffice | ci | 7.4e-05 | 0.00888 | stays established |
| 148 | PR inconclusive | ci2 | 0.058 | 1 | not established |
| 149 | PR E (gain) / part of the gap | t | 4.81e-05 | 0.00597 | stays established |
| 150 | PR E (gain) / inconclusive | t | 0.000655 | 0.0623 | no longer established |
| 151 | PR E | t | 0.00135 | 0.116 | no longer established |
