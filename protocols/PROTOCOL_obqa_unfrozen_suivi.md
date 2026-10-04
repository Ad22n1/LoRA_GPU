# Second additif à `PROTOCOL_obqa_unfrozen.md` — suivi : réévaluation sur un seul modèle de carte

Écrit le 26/09 et committé AVANT tout run de ce suivi, APRÈS la lecture des résultats de la campagne principale (verdicts connus :
test (3), rétrécissement de l'écart entre bandes, ÉTABLI, +3,84, t = 4,12).

## Pourquoi
Le contrôle à PARTITION égal du premier additif n'est pas calculable pour le test (3) : aucune graine n'a ses quatre runs évalués sur le
même modèle de carte. La règle du premier additif ne déclenchait le suivi que si le contrôle CHANGEAIT le verdict ; un contrôle
incalculable n'en change aucun. Ce suivi va donc au-delà de la règle, et il est rapporté comme tel : le verdict pré-enregistré
reste celui de la campagne principale ; le suivi dit s'il résiste à l'évaluation sur un seul modèle de carte.

## Ce qui est fait
1. **Réentraînement des dix références gelées du rang 2**, avec leurs facteurs sauvegardés (elles ne les avaient pas) : `top` à
   5·10⁻³ et `bottom` à 2·10⁻³, graines 2 à 6, configuration identique, dans un dossier séparé (`~/lora-runs-suivi`), hors de la
   population principale.
2. **Réévaluation, dans un seul job, sur une seule machine**, dont le modèle de carte est vérifié au démarrage (le job s'arrête sinon),
   des trente checkpoints comparés : ces dix références réentraînées, et `top_unfrozen` et `bottom_unfrozen` aux rangs 1 et 2, à leur
   taux retenu (2·10⁻³), graines 2 à 6, rechargés de leurs facteurs finaux (le chemin de rechargement du point 6).
3. **Les tests (1), (2) et (3) refaits** sur ces seules évaluations, avec les mêmes seuils (G fixé : 8,04 et 12,36 ; 3,495 ; 2,776).

## Rapporté
Les verdicts du suivi à côté de ceux de la campagne ; l'écart entre les références réentraînées et les références d'origine (dans
le processus d'entraînement), qui mesure la reproductibilité de l'entraînement lui-même ; l'écart entre l'évaluation sur la carte
unique et l'évaluation d'origine, run par run.

## Engagements
Rapporté quelle que soit l'issue. Si le verdict du test (3) change, le papier donne les deux et le dit.
