# Protocole pré-enregistré — `bottom_sigma` : Σ inverse-t-il le sens du déplacement du taux ?

Écrit et committé avant le premier run (24/09), avec le code du mode et son test.

## Pourquoi
Sans Σ, la bande mineure choisit un taux plus bas que la bande dominante (Llama, rang 2 : `bottom` 5·10⁻³, `top` 10⁻²).
Avec Σ dans le facteur figé, LoRA-XS trouve l'inverse. `bottom_sigma` (B = U √σ sur la bande mineure, figé) teste ce
renversement dans notre propre montage, contre `top_sigma` (la même construction sur la bande dominante).

## Une seule prédiction
**`bottom_sigma` choisit un taux strictement plus élevé que `top_sigma`**, dans la même population.

## Montage
Llama-3.2-1B, tâche de format, rang 2, écrêtage 1,0, α = r, sélection sur les graines 0 et 1 (validation), meilleur score.
- **Grille étendue largement au-dessus de 10⁻²** : 5·10⁻³ à 1 (huit taux). Les √σ de la bande mineure sont minuscules, et
  le taux retenu pourrait sinon sortir par le haut. Une sélection au bord entraîne un prolongement d'un cran.
- **Modules dégénérés — décision fixée ici : analyse AVEC et SANS eux.** Sur Llama, `q` et `o` ont σ_min/σ_max ≈ 10⁻⁶,
  sous le seuil du critère spectral (10⁻⁴) ; leur pas y serait quasi nul.
  - avec : sept modules — `bottom_sigma` contre `top_sigma`, dont la sélection existe déjà (5·10⁻³) ;
  - sans : cinq modules non dégénérés — `bottom_sigma` ET `top_sigma` balayés dans cette population.

## Les issues
1. **Confirmée** : plus élevé dans les deux populations.
2. **Partielle** : dans une seule.
3. **Réfutée** : dans aucune (taux égal ou plus bas).

## Engagements
Rapporté quelle que soit l'issue. Le papier dira dans quel montage le sens du déplacement s'inverse, et dans lequel non.
