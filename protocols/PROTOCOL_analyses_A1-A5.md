# Analyses post hoc, sans entraînement — définitions et scripts committés avant exécution

Écrit et committé AVANT toute exécution, avec les scripts (`scripts/analyses_v4.py`, qui réutilise la recherche de cellules de
`scripts/analyses_v3.py`) ; sa date est celle de son commit.

## Statut
Les données de ces analyses existent déjà : elles sont donc **post hoc**. Committer leurs définitions avant de les exécuter fixe ce qui
sera calculé et rapporté, pas une confirmation. Aucune lecture n'est fixée d'avance ; toutes sont **descriptives** (statut `PH · D`), et
aucune affirmation confirmatoire du papier ne reposera sur elles.

## Règles communes
- Scripts, graines de tirage et chemins des données committés avec ce fichier ; sorties enregistrées en JSON dans le dépôt
  (`analyses_v4.json`).
- Chaque analyse est rapportée quelle que soit son issue, y compris nulle.
- Si une donnée d'entrée manque (facteurs ou prédictions non sauvegardés), l'analyse le dit, rapporte ce qui reste calculable, et ne
  remplace pas la donnée manquante par une autre sans le déclarer.
- Médiane sur les couches par type de module (q, k, v, o, gate, up, down) ; la distribution complète est sauvegardée.

## A1 — Angles principaux entre directions
**Questions.** Les directions du gradient initial tombent-elles dans la bande mineure ? Les directions apprises par LoRA sont-elles
stables d'une graine à l'autre ? À quelle distance `grad` est-il de `learned` ?
**Entrées.** Les bases de `grad` (64 séquences) sur les deux paires ; celles de `grad256`, `grad1024`, `gradall` produites par la campagne
du gradient accumulé. Les bandes dominante et mineure de rang 2 (colonnes de U, cache SVD utilisé par les bras). Les bases de `learned` :
graines de direction 0 et 7 sur OpenBookQA, 0 sur la tâche de format. Les B sauvegardés du LoRA libre au rang 1, graines 2 à 6 (phase 1).
**Mesure.** Pour deux sous-espaces de bases orthonormales Q₁ et Q₂, les cosinus des angles principaux sont les valeurs singulières de
Q₁ᵀQ₂. On rapporte le plus grand angle principal et le recouvrement moyen, la moyenne des cos².
**Référence nulle.** La même mesure contre 100 bases orthonormales aléatoires de même dimension dans ℝ^{d_out} (graine 2000) ; on rapporte
chaque recouvrement divisé par sa moyenne nulle.
**Comparaisons.** 1. `grad` contre la bande dominante, la bande mineure et `learned`, par paire et type de module. 2. `learned` graine 0
contre `learned` graine 7 (OpenBookQA). 3. Les B du LoRA libre au rang 1, deux à deux sur leurs cinq graines. 4. L'énergie de la base de
`grad` par dixième du spectre de W₀, comme le profil de la phase 1.

## A2 — Normes de mise à jour par type de module, A gelé contre B gelé
**Question.** L'avantage des bras à A gelé se loge-t-il dans `gate` et `up`, où d_out > d_in et où ils détiennent plus de paramètres ?
**Entrées.** Les normes par module ‖(α/r)BA‖_F déjà enregistrées pour les runs rapportés (graines 2 à 6), au rang 2 : `top`, `bottom`,
`dual_top`, `dual_bottom` et le LoRA libre au rang 1, sur la tâche de format avec Llama-3.2-1B et sur OpenBookQA avec Qwen2.5-1.5B.
**Mesures.** Par type de module : la norme (médiane sur les couches, moyenne sur les graines) ; le rapport A gelé / B gelé par bande ; la
part de chaque type de module dans la node des normes au carré ; à côté, le nombre de paramètres entraînés par type de module et par bras.
**Limite déclarée.** Une norme de mise à jour ne mesure pas la contribution au score ; l'analyse localise, elle ne sépare pas le côté du
facteur de la forme du module.

## A3 — Équivalence entre bandes spectrales et base aléatoire (TOST)
**Question.** « Aucune bande spectrale n'est montrée meilleure que `random_ortho` » est une non-détection. Quelle marge les données
excluent-elles ?
**Populations.** Au rang 2, B gelé, chaque population où `random_ortho` a tourné, à ses graines rapportées : tâche de format avec
Llama-3.2-1B, OpenBookQA avec Qwen2.5-1.5B, tâche de format avec Qwen2.5-1.5B, ARC-Challenge, et la tâche de format avec Qwen2.5-0.5B.
**Mesures.** 1. Par bande : `top` − `random_ortho` et `bottom` − `random_ortho`, appariés par graine, intervalle à 90 % (t) ; la plus petite
marge symétrique exclue, max(|borne basse|, |borne haute|). 2. La meilleure bande spectrale − `random_ortho`, par bootstrap apparié sur les
graines (10 000 tirages, graine 3000), la meilleure bande étant recalculée à chaque tirage ; même marge.
**Rapporté.** La marge par population et par bande, sans seuil fixé d'avance. Les populations à cinq graines donnent des marges larges ; on
le dit.

## A4 — Le bras mineur qui reste au niveau du modèle de base (OpenBookQA)
**Question.** Sur OpenBookQA avec Qwen2.5-1.5B, `bottom` à B gelé obtient 0,4220, exactement le score du modèle de base. N'apprend-il
rien, ou apprend-il quelque chose qui s'annule au score ?
**Entrées.** Pour les graines 2 à 6 : les prédictions question par question de `bottom` et du modèle de base sur les 500 questions de test.
Si elles n'ont pas été sauvegardées, dire si les facteurs l'ont été ; dans ce cas seulement, réévaluer les checkpoints sur la carte du run
d'origine et le déclarer.
**Mesures.** Par graine : le nombre de questions dont la prédiction change par rapport au modèle de base, séparé en juste → faux et faux →
juste ; la variation moyenne de l'écart de vraisemblance entre la bonne option et la meilleure mauvaise ; la norme de la mise à jour de
`bottom` comparée à celle de `top`, et la perturbation relative.

## A5 — Bootstrap sur les graines et les questions
**Question.** Les écarts de 2 à 5 points entre bandes tiennent-ils quand on propage aussi l'incertitude due à l'échantillon de questions,
et pas seulement celle des graines ?
**Entrées.** La justesse question par question de chaque run rapporté, au rang 2, pour `top`, `bottom`, `dual_top`, `dual_bottom` : tâche de
format avec Llama-3.2-1B (614 questions) et OpenBookQA avec Qwen2.5-1.5B (500 questions), graines 2 à 6. D'abord, dire pour chaque run si
ces prédictions existent ; si certaines manquent, restreindre l'analyse et le déclarer.
**Méthode.** Bootstrap à deux niveaux, apparié : on tire avec remise les graines, puis les questions, les mêmes pour tous les bras comparés ;
10 000 tirages, graine 4000.
**Statistiques.** L'écart entre bandes avec B gelé (`bottom` − `top`), avec A gelé (`dual_bottom` − `dual_top`), et leur différence
(l'interaction du tableau facteur × bande) ; intervalles à 95 % (percentiles), à côté des intervalles sur les graines seules.
**Rapporté.** Si les intervalles s'élargissent, et de combien ; si un résultat établi sur les graines seules ne l'est plus, on le dit.

---

## Précisions déclarées avant exécution (d'après l'inventaire des données, 29/09)
- **A4** : sur OpenBookQA, les runs rapportés n'ont gardé **ni les prédictions question par question, ni leurs facteurs** (seulement
  l'exactitude et les normes de mise à jour). Selon la règle ci-dessus, la comparaison question par question et la réévaluation sont
  impossibles ; A4 rapporte ce qui reste calculable : la norme de la mise à jour de `bottom` et de `top`, et la perturbation relative
  ‖ΔW‖ / ‖W₀‖_F (‖W₀‖_F tiré du cache SVD).
- **A5** : pour la même raison, **OpenBookQA est exclu** ; l'analyse porte sur la tâche de format seule. La justesse question par question
  y est recalculée à partir des générations enregistrées (`generations_test.jsonl`), avec la fonction de notation du code (`check_alien`,
  champ `parsed`), et **vérifiée run par run** contre `format_parsed` enregistré ; au moindre écart, l'analyse s'arrête.
- **A1** : la comparaison deux à deux des B du LoRA libre au rang 1 porte sur les paires dont les facteurs ont été sauvegardés (phase 1 :
  OpenBookQA) ; l'analyse dit pour quelle paire ils manquent.
- **Cellules** : retrouvées par leur valeur publiée ; si plusieurs taux la redonnent, celui de la sélection (le meilleur en validation).
