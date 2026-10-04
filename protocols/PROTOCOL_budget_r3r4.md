# Protocole pré-enregistré — B gelé aux rangs 3 et 4 contre A gelé au rang 2 (le budget)

Écrit et committé AVANT tout run ; sa date est celle de son commit.
**Arrêt des lancements : mardi 29/09 à 20 h.** Ce qui n'est pas parti à cette heure n'entre pas dans le rapport.

## Pourquoi
Geler A entraîne r·d_out paramètres par module, geler B r·d_in. Sur OpenBookQA au rang 2, geler A fait mieux que geler B, mais avec plus de
paramètres. Le contrôle sur modules carrés est resté non concluant (effondrement de `top` sur `q` et `o`). Ici, on garde les sept modules et on
donne à B gelé **plus** de paramètres qu'à A gelé au rang 2 : si A gelé reste devant, le budget ne peut plus expliquer son avantage.

## Budgets exacts (par couche ; 28 couches pour chaque modèle)
| modèle | A gelé, r = 2 | B gelé, r = 3 | B gelé, r = 4 |
|---|---|---|---|
| Qwen2.5-1.5B | 46 080 | 54 528 (**+18,3 %**) | 72 704 (**+57,8 %**) |
| Qwen2.5-7B | 99 328 | 121 344 (**+22,2 %**) | 161 792 (**+62,9 %**) |

## Montage
- OpenBookQA, **Qwen2.5-1.5B** et **Qwen2.5-7B** ; configuration reprise des runs de référence eux-mêmes (A gelé au rang 2), seuls changent le
  mode, le rang (α = r), le taux et la graine.
- **Nouveaux bras** : `top` et `bottom` (B gelé sur les 3 ou 4 premiers, ou derniers, vecteurs singuliers à gauche), aux rangs 3 et 4.
- **Taux** : six par bras (1,5B : 5·10⁻⁴, 10⁻³, 2·10⁻³, 5·10⁻³, 10⁻², 2·10⁻² ; 7B : 2·10⁻⁴, 5·10⁻⁴, 10⁻³, 2·10⁻³, 5·10⁻³, 10⁻²), sélection en validation sur les graines 0 et 1, rapport sur les graines 2
  à 6 au seul taux retenu ; si l'optimum tombe au bord, **un cran de prolongement**, puis le taux retenu est rapporté comme limite s'il reste au bord.
- **Références** : `dual_top` et `dual_bottom` (A gelé, r = 2) aux taux publiés, graines 2 à 6. **Si une référence n'a pas tourné entièrement
  sur RTX 4000 Ada, elle est réentraînée à l'identique sur ce modèle** (graines 2 à 6, `~/lora-runs-suivi`), et c'est la version réentraînée qui
  sert aux tests ; l'autre est rapportée.
- **Une seule carte : RTX 4000 Ada** ; pour le 7B, les machines où un run de ce modèle s'est déjà terminé.
- **Facteurs non sauvegardés** : le quota personnel (30 Go, dont 28 déjà utilisés) ne le permet pas. Le lanceur vérifie ce quota avant chaque étape.
- **Environ 136 runs** (8 bras × 17) **+ jusqu'à 10 réentraînements**, soit **~32 PARTITION-heures** (le 7B en fait l'essentiel).

## Test, écrit d'avance
**A gelé au rang 2 moins B gelé au rang 3**, pour chaque bande (`top`, `bottom`) et chaque modèle, apparié par graine, **t > 3,495** (deux bandes).
Rapporté sans test : A gelé au rang 2 moins B gelé au rang 4 ; les scores et taux retenus ; les modèles de carte.

## Lecture fixée d'avance
- Si A gelé au rang 2 reste devant B gelé au rang 3 **sur les deux bandes**, sur un modèle, alors qu'il a moins de paramètres : **le budget
  n'explique pas l'avantage de geler A** sur ce modèle.
- Si B gelé au rang 3 **dépasse** A gelé au rang 2 de façon établie sur une bande : l'avantage de geler A est **attribué au budget sur cette bande**.
- Sinon, l'avantage est rapporté comme **en partie lié au budget** sur ce modèle.

## Règles communes
Une relance à l'identique autorisée pour un échec machine (mémoire, disque, carte occupée), sur les mêmes machines éprouvées, et c'est tout.
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés hors le cran de prolongement.
