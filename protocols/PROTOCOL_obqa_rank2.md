# Protocole pré-enregistré — OpenBookQA au rang 2, Qwen2.5-1.5B

Écrit et committé avant le premier run (24/09).

## Pourquoi
Au rang 16 sur OpenBookQA, l'inversion ne s'est pas reproduite ; sur la tâche de format, au rang 16, elle se dessine.
Ce croisement complète le plan tâche × rang : si l'inversion apparaît sur OpenBookQA au rang 2, l'effet de rang est
confirmé des deux côtés ; si elle n'apparaît pas, OpenBookQA ne la porte à aucun rang.

## Montage
Qwen2.5-1.5B, OpenBookQA (empreinte des données `dc05f4d847f9`), sept modules, écrêtage 1,0, α = r, 600 000 tokens,
évaluation par vraisemblance des options. `top`, `bottom`, `random`, `random_ortho` au rang 2, **bras aléatoires salés**
comme aux rangs 2 et 4 (différence déclarée avec la campagne du rang 16, dont `random_ortho` n'était pas salé).
Référence `free` au rang 1 : **+13,4 %** de paramètres (1 154 048 contre 1 017 856) — les dimensions de Qwen ne permettent
pas l'écart de +7,5 % de Llama ; déclaré.
Sélection : sept taux de 5·10⁻⁴ à 5·10⁻², graines 0 et 1, validation ; bord → prolongement d'un cran. Rapport : graines 2 à 6.

## Règles, fixées d'avance (celles du rang 16)
- **Plancher** : `free` doit dépasser le modèle de base (0,422) d'au moins 5 points, sinon « non informatif ».
- **Taux communs** : celui de `bottom` et celui du meilleur autre bras gelé ; confondus → inversion non testable.
- **Tests** : chaque moitié de l'inversion, t > 2,776 (bilatéral), contre le meilleur concurrent recalculé et contre chaque
  concurrent fixé ; le test d'interaction bras × taux, seuil corrigé de Bonferroni.
- **Gel contre choix** aux taux propres, intervalle bootstrap apparié, rapporté quelle que soit sa direction.
- **Prédiction du déplacement du taux** : `bottom` choisit un taux strictement plus bas que `top`, `random` et
  `random_ortho` — confirmée (les trois), partielle, ou réfutée (aucun).

## Engagements
Rapporté quelle que soit l'issue ; aucun taux choisi après coup ; aucune graine ajoutée.
