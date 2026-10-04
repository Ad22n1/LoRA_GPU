# Équivalence numérique étendue : bras `bottom` (cache SVD fp16 des runs) contre PEFT 0.21.0 `init_lora_weights="mica"` (SVD fp32)

Statut : **post hoc, descriptif** ; sans entraînement ; aucun résultat publié modifié. Prolonge le test du commit `<commit>`.
Tous les modules adaptés : Qwen2.5-1.5B (196), Llama-3.2-1B (112) ; rangs 2 et 16 ; α/r = 1 ; poids chargés dans la précision des runs, copiés en fp32. Valeurs par module : `test_mica_peft_all.json`.

## Définitions

- **B de PEFT** : `mica_init` de PEFT 0.21.0, appelé une fois par module au rang 16 (SVD fp32 sur le processeur) ; le B de rang 2 en est
  les deux dernières colonnes, ce qui est exactement ce que `mica_init` produit au rang 2 (même SVD, mêmes colonnes).
- **B du cache** : notre bras `bottom` construit comme dans les runs, depuis le cache SVD stocké en fp16 (colonnes m − r à m − 1 de U).
- **Alignement des signes** : chaque colonne du B du cache est multipliée par ±1 pour que son produit scalaire avec la colonne
  correspondante du B de PEFT soit positif ou nul. Les vecteurs singuliers ne sont définis qu'au signe près.
- **Écart maximal sur B** : max_ij |B_cache,aligné − B_PEFT|_ij.
- **Distance entre sous-espaces** : le sinus du plus grand angle principal θ_max entre les espaces engendrés par les deux B, calculé
  comme sin θ_max = ‖(I − B_PEFT B_PEFTᵀ) Q_cache‖₂, où Q_cache est une base orthonormale (QR) de l'espace engendré par le B du cache.
  Le B du cache, arrondi en fp16, n'est orthonormal qu'à ~10⁻⁴ près : on l'orthonormalise avant de mesurer l'angle, et l'on n'utilise
  pas √(1 − cos²θ), qui transformerait une erreur de norme de 10⁻⁴ en un sinus apparent de 1,4·10⁻². Elle ne dépend ni des signes ni
  d'une rotation à l'intérieur de la bande ; elle vaut 0 pour deux sous-espaces identiques et 1 si l'un contient une direction orthogonale à
  l'autre. Comme le bras gèle B et entraîne A, seule cette distance change ce que le bras peut représenter.
- **Écart relatif des sorties** : pour un même lot aléatoire x (2 × 8 positions) et une même matrice A non nulle (A_cache = signes ⊙ A_PEFT),
  max |y_cache − y_PEFT| / max |y_PEFT|, où y = x Aᵀ Bᵀ est l'apport de l'adaptateur (α/r = 1).
- **Écart relatif des gradients sur A**, à A = 0, pour la perte L = Σ w ⊙ y avec w aléatoire fixe : ∂L/∂A = Bᵀ M, M = Σ_positions wᵀ x ;
  max |(∂L/∂A)_cache,aligné − (∂L/∂A)_PEFT| / max |(∂L/∂A)_PEFT|.
- **Écarts de valeurs singulières** (σ décroissantes, en fp32, indices 0 à m − 1 ; la bande mineure de rang r est m − r … m − 1) :
  écart à la frontière (σ_{m−r−1} − σ_{m−r}) / σ_{m−r} ; plus petit écart interne min_k (σ_k − σ_{k+1}) / σ_{k+1} sur la bande.
- **Signalement** : un module dont la distance entre sous-espaces dépasse 0.01 ; le cache y sélectionnerait un autre sous-espace,
  et non seulement un sous-espace moins précis.
- Note : l'écart sur B, les sorties et les gradients se comparent colonne par colonne ; ils grandissent si les valeurs singulières
  internes à la bande sont presque égales (rotation dans la bande), alors que la distance entre sous-espaces, elle, n'en dépend pas.

## Résultats par modèle, rang et type de module (médiane / maximum sur les couches)

| modèle | r | type | écart max sur B | distance sin θ_max | sorties (relatif) | gradients (relatif) | signes retournés |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 2 | q | 8.2e-05 / 1.6e-04 | 2.3e-04 / 8.4e-04 | 4.3e-04 / 1.1e-03 | 3.2e-04 / 1.0e-03 | 23 / 56 |
| Qwen2.5-1.5B | 2 | k | 1.3e-04 / 2.6e-04 | 2.1e-04 / 6.3e-04 | 2.5e-04 / 5.8e-04 | 2.2e-04 / 4.5e-04 | 29 / 56 |
| Qwen2.5-1.5B | 2 | v | 1.2e-04 / 2.5e-04 | 2.1e-04 / 2.7e-04 | 2.3e-04 / 4.1e-04 | 2.0e-04 / 4.3e-04 | 28 / 56 |
| Qwen2.5-1.5B | 2 | o | 6.3e-05 / 1.1e-04 | 2.4e-04 / 7.7e-04 | 4.1e-04 / 1.0e-03 | 4.0e-04 / 8.7e-04 | 27 / 56 |
| Qwen2.5-1.5B | 2 | gate | 4.1e-05 / 2.5e-04 | 2.3e-04 / 5.9e-04 | 3.4e-04 / 2.0e-03 | 2.7e-04 / 1.6e-03 | 31 / 56 |
| Qwen2.5-1.5B | 2 | up | 3.3e-05 / 1.6e-04 | 2.1e-04 / 6.2e-04 | 3.0e-04 / 1.1e-03 | 2.5e-04 / 6.7e-04 | 27 / 56 |
| Qwen2.5-1.5B | 2 | down | 1.8e-04 / 4.2e-04 | 2.5e-04 / 9.0e-04 | 3.8e-04 / 8.1e-04 | 3.6e-04 / 1.0e-03 | 32 / 56 |
| Qwen2.5-1.5B | 16 | q | 1.2e-04 / 1.9e-04 | 2.6e-04 / 6.7e-04 | 4.5e-04 / 6.7e-04 | 4.6e-04 / 8.8e-04 | 220 / 448 |
| Qwen2.5-1.5B | 16 | k | 1.7e-04 / 1.6e-03 | 2.6e-04 / 5.8e-04 | 3.1e-04 / 3.5e-03 | 3.2e-04 / 5.0e-03 | 221 / 448 |
| Qwen2.5-1.5B | 16 | v | 1.7e-04 / 1.1e-03 | 2.8e-04 / 1.3e-03 | 3.5e-04 / 2.1e-03 | 5.6e-04 / 2.4e-03 | 214 / 448 |
| Qwen2.5-1.5B | 16 | o | 9.7e-05 / 1.5e-04 | 2.4e-04 / 7.5e-04 | 4.0e-04 / 5.9e-04 | 4.2e-04 / 9.1e-04 | 226 / 448 |
| Qwen2.5-1.5B | 16 | gate | 2.7e-04 / 1.8e-03 | 6.1e-04 / 4.3e-03 | 7.7e-04 / 5.6e-03 | 1.3e-03 / 1.0e-02 | 225 / 448 |
| Qwen2.5-1.5B | 16 | up | 2.2e-04 / 1.3e-03 | 5.4e-04 / 5.5e-03 | 7.9e-04 / 4.3e-03 | 1.5e-03 / 5.6e-03 | 225 / 448 |
| Qwen2.5-1.5B | 16 | down | 8.3e-04 / 6.7e-03 | 8.4e-04 / 3.8e-03 | 1.6e-03 / 1.3e-02 | 2.2e-03 / 2.3e-02 | 219 / 448 |
| Llama-3.2-1B | 2 | q | 9.9e-05 / 3.4e-04 | 2.9e-04 / 5.5e-04 | 4.5e-04 / 1.2e-03 | 3.9e-04 / 1.7e-03 | 22 / 32 |
| Llama-3.2-1B | 2 | k | 1.5e-04 / 2.6e-04 | 2.5e-04 / 3.3e-04 | 2.8e-04 / 4.0e-04 | 2.3e-04 / 3.0e-04 | 16 / 32 |
| Llama-3.2-1B | 2 | v | 1.0e-04 / 2.5e-04 | 2.3e-04 / 3.4e-04 | 2.3e-04 / 4.1e-04 | 2.3e-04 / 4.9e-04 | 18 / 32 |
| Llama-3.2-1B | 2 | o | 6.6e-05 / 1.7e-04 | 2.5e-04 / 4.8e-04 | 4.8e-04 / 7.0e-04 | 3.7e-04 / 7.1e-04 | 15 / 32 |
| Llama-3.2-1B | 2 | gate | 8.8e-05 / 9.6e-04 | 2.3e-04 / 1.1e-03 | 3.7e-04 / 6.6e-03 | 3.3e-04 / 5.1e-03 | 13 / 32 |
| Llama-3.2-1B | 2 | up | 4.6e-05 / 1.2e-04 | 2.3e-04 / 1.2e-03 | 3.5e-04 / 1.7e-03 | 3.0e-04 / 1.6e-03 | 16 / 32 |
| Llama-3.2-1B | 2 | down | 2.1e-04 / 1.2e-03 | 2.4e-04 / 1.5e-03 | 5.7e-04 / 1.5e-03 | 5.1e-04 / 2.1e-03 | 22 / 32 |
| Llama-3.2-1B | 16 | q | 2.1e-04 / 1.5e-02 | 3.4e-04 / 6.0e-04 | 5.6e-04 / 1.8e-02 | 6.3e-04 / 3.5e-02 | 122 / 256 |
| Llama-3.2-1B | 16 | k | 2.4e-04 / 4.8e-04 | 2.8e-04 / 3.6e-04 | 3.1e-04 / 8.1e-04 | 4.1e-04 / 1.2e-03 | 121 / 256 |
| Llama-3.2-1B | 16 | v | 4.0e-04 / 9.4e-04 | 3.2e-04 / 3.2e-03 | 6.7e-04 / 1.8e-03 | 9.6e-04 / 2.5e-03 | 120 / 256 |
| Llama-3.2-1B | 16 | o | 1.1e-04 / 3.9e-04 | 2.7e-04 / 4.7e-04 | 5.1e-04 / 1.3e-03 | 4.7e-04 / 1.2e-03 | 110 / 256 |
| Llama-3.2-1B | 16 | gate | 5.4e-04 / 2.6e-03 | 8.3e-04 / 1.9e-03 | 1.2e-03 / 5.6e-03 | 2.5e-03 / 1.2e-02 | 120 / 256 |
| Llama-3.2-1B | 16 | up | 3.8e-04 / 2.4e-03 | 8.1e-04 / 5.8e-03 | 1.1e-03 / 5.2e-03 | 2.2e-03 / 7.5e-03 | 126 / 256 |
| Llama-3.2-1B | 16 | down | 6.3e-04 / 2.0e-03 | 9.1e-04 / 2.2e-03 | 1.3e-03 / 6.1e-03 | 1.9e-03 / 8.2e-03 | 126 / 256 |

## Modules signalés (distance entre sous-espaces > 0.01) : 0 sur 616 comparaisons

Aucun : à tous les modules et aux deux rangs, le cache sélectionne le même sous-espace que PEFT.

## Llama-3.2-1B, modules q et o : écarts de valeurs singulières à la bande mineure, couche par couche

| type | couche | r = 2 : écart frontière | r = 2 : écart interne | r = 2 : sin θ_max | r = 16 : écart frontière | r = 16 : écart interne min | r = 16 : sin θ_max |
|---|---|---|---|---|---|---|---|
| q | 0 | 3.42e+00 | 8.80e+00 | 3.6e-04 | 9.80e-02 | 4.78e-03 | 4.5e-04 |
| q | 1 | 4.22e-01 | 7.60e+00 | 2.9e-04 | 6.04e-02 | 4.13e-02 | 3.1e-04 |
| q | 2 | 1.44e+00 | 1.04e+01 | 2.8e-04 | 4.05e-02 | 4.35e-02 | 2.7e-04 |
| q | 3 | 5.06e-01 | 9.10e+00 | 2.3e-04 | 4.31e-02 | 1.89e-02 | 5.9e-04 |
| q | 4 | 6.44e-01 | 5.00e-01 | 2.9e-04 | 1.00e-01 | 3.98e-02 | 2.7e-04 |
| q | 5 | 3.11e-01 | 2.01e+00 | 3.1e-04 | 1.08e-01 | 2.48e-02 | 3.7e-04 |
| q | 6 | 4.65e-01 | 3.69e-01 | 5.5e-04 | 7.80e-02 | 2.18e-02 | 2.7e-04 |
| q | 7 | 7.13e-01 | 3.10e+00 | 2.8e-04 | 8.26e-02 | 2.39e-02 | 2.5e-04 |
| q | 8 | 1.16e+00 | 4.86e+01 | 2.7e-04 | 7.98e-02 | 5.98e-02 | 2.7e-04 |
| q | 9 | 9.23e-01 | 2.17e+01 | 2.9e-04 | 2.98e-02 | 8.71e-03 | 6.0e-04 |
| q | 10 | 7.32e-01 | 1.87e-01 | 4.2e-04 | 1.05e-01 | 2.15e-02 | 3.0e-04 |
| q | 11 | 1.21e+00 | 1.26e+01 | 2.2e-04 | 3.58e-02 | 2.02e-02 | 4.4e-04 |
| q | 12 | 6.80e-01 | 1.35e+00 | 2.7e-04 | 3.41e-02 | 2.39e-02 | 5.2e-04 |
| q | 13 | 8.36e-01 | 3.78e+00 | 2.4e-04 | 1.15e-01 | 3.22e-02 | 2.7e-04 |
| q | 14 | 8.03e-01 | 4.70e+00 | 3.1e-04 | 6.50e-02 | 2.51e-02 | 5.2e-04 |
| q | 15 | 1.97e+00 | 9.56e-01 | 2.4e-04 | 4.49e-02 | 4.39e-02 | 4.9e-04 |
| o | 0 | 7.57e-01 | 3.38e+00 | 2.4e-04 | 5.52e-02 | 3.54e-02 | 2.9e-04 |
| o | 1 | 2.47e-01 | 1.61e+00 | 2.5e-04 | 5.08e-02 | 3.76e-02 | 2.5e-04 |
| o | 2 | 1.94e-01 | 5.52e-01 | 2.5e-04 | 5.02e-02 | 5.12e-02 | 3.6e-04 |
| o | 3 | 1.50e+00 | 7.90e-01 | 3.2e-04 | 7.85e-02 | 1.66e-02 | 2.6e-04 |
| o | 4 | 6.99e-01 | 9.47e-01 | 2.6e-04 | 1.20e-01 | 2.51e-02 | 2.4e-04 |
| o | 5 | 3.05e-01 | 4.07e+00 | 4.7e-04 | 7.15e-02 | 5.67e-02 | 2.8e-04 |
| o | 6 | 1.20e+00 | 1.34e+01 | 2.4e-04 | 3.74e-02 | 1.92e-02 | 2.8e-04 |
| o | 7 | 3.23e-01 | 8.26e-01 | 4.8e-04 | 1.24e-01 | 1.86e-02 | 3.2e-04 |
| o | 8 | 5.01e-01 | 2.02e+01 | 2.5e-04 | 1.20e-01 | 1.82e-02 | 2.8e-04 |
| o | 9 | 6.92e-01 | 1.41e+00 | 2.4e-04 | 8.78e-02 | 5.58e-02 | 2.7e-04 |
| o | 10 | 2.94e-01 | 1.39e+01 | 4.1e-04 | 1.14e-01 | 7.61e-02 | 2.3e-04 |
| o | 11 | 6.21e-01 | 1.36e+01 | 2.3e-04 | 9.08e-02 | 3.70e-02 | 3.1e-04 |
| o | 12 | 1.14e+00 | 2.14e+00 | 2.5e-04 | 6.94e-02 | 7.01e-02 | 2.5e-04 |
| o | 13 | 8.87e-01 | 3.70e+00 | 2.3e-04 | 6.82e-02 | 2.61e-02 | 2.4e-04 |
| o | 14 | 1.78e+00 | 8.64e+00 | 2.3e-04 | 4.22e-02 | 2.47e-02 | 4.7e-04 |
| o | 15 | 1.22e+00 | 6.61e-01 | 2.3e-04 | 7.60e-02 | 2.02e-02 | 2.5e-04 |

- Paramètres entraînables de PEFT : A seul, B gelé, à tous les modules : oui.
- Orthonormalité du B du cache (arrondi fp16) : max |BᵀB − I| = 1.0e-03 sur tous les modules.
- Durée : 478 s sur le processeur.
