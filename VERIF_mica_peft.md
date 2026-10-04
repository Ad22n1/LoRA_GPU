# Equivalence numerique : bras `bottom` contre PEFT 0.21.0 `init_lora_weights="mica"` (sans entrainement)

Modele Qwen/Qwen2.5-1.5B, couches [0, 14] (14 modules : q, k, v, o, gate, up, down), r = 2, alpha / r = 1, copies fp32 des poids ;
un lot aleatoire fixe (graine 0) ; perte lineaire en la sortie. Script : `scripts/test_mica_peft.py` ; valeurs completes : `test_mica_peft.json`.
Les vecteurs singuliers sont definis au signe pres : B est compare brut, apres alignement des signes, et comme sous-espace.

| comparaison (ecart absolu maximal sur les modules) | B brut | B aligne | projecteurs BBᵀ | sorties, A = 0 | sorties, meme A | gradient sur A |
|---|---|---|---|---|---|---|
| PEFT contre notre `bottom`, SVD fp32 fraiche | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| notre `bottom` : cache fp16 contre SVD fp32 fraiche | — | 2.60e-04 | 2.68e-04 | — | 1.14e-03 | 9.00e-03 |
| PEFT contre le bras des runs (cache fp16) | 1.34e+00 | 2.60e-04 | 2.68e-04 | — | — | — |

- Parametres entrainables : identiques (A seul, r x d_in par module ; B gele dans les deux).
- Ecarts relatifs dus au cache : sorties 7.7e-04 de l'apport de l'adaptateur ; gradients sur A 7.8e-04 de leur valeur maximale.
- Signes : le cache retourne 13 colonne(s) de B sur 28 par rapport a la SVD fp32 fraiche ; sans effet sur le sous-espace.
- **Mathematiquement, les deux constructions coincident** : oui (B et gradient a la precision fp32, une fois les signes alignes).
- **Le cache fp16** decale B de 2.6e-04 au plus (sous-espace : 2.7e-04) : a la precision fp16.
