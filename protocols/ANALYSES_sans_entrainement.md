# Analyses sans entraînement — définitions fixées avant exécution (`scripts/analyses_v2.py`)

Committées AVANT d'être exécutées. Chaque cellule est retrouvée par sa valeur publiée (à 10⁻⁴ près) ; le script s'arrête sinon, ou sur un doublon.
Intervalles : bootstrap apparié sur les graines, 10 000 tirages, graine de tirage 0. Aucun seuil de décision : ce sont des analyses descriptives,
sauf le t de l'analyse 2, rapporté avec le seuil habituel pour deux comparaisons (3,495).

2. **A gelé (rang 2) contre le meilleur B gelé (rang 4)**, tâche de format, Llama, graines 2 à 6. Le B gelé au rang 4 retenu est le meilleur des
   quatre bras principaux **à son meilleur taux** (choix qui l'avantage). Budgets par couche : 47 104 contre 81 920 (+73,9 % pour B gelé).
3. **Décomposition facteur × bande au rang 2**, sur les cinq populations qui ont les quatre bras (format Llama, graines 2-6 et 120-124 ;
   OpenBookQA avec Qwen2.5-1.5B, Qwen2.5-7B, Mistral-7B) : effet du facteur, effet de la bande, interaction.
4. **I = G − U = libre − B entraîné au rang 1**, et U/G, sur les cinq paires de la comparaison à budget égal.
5. **Courbes de perte** (journaux d'entraînement), tâche de format et OpenBookQA : perte à 25, 50, 75 et 100 % des pas, et baisse sur le
   dernier quart, pour distinguer un bras plus lent d'un bras plafonné. Fichiers : `analyses_loss_curves.csv` et `.png`.
6. **Pas et époques** : le jeu d'entraînement reconstruit par la fonction du code (`build_dataset`), comme à l'entraînement.
1. **Où écrit le LoRA libre** : voir `PROTOCOL_free_writes.md`.
