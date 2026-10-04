# Analyse pré-enregistrée — où écrit le LoRA libre (sans entraînement)

Le script d'analyse (`scripts/analyses_v2.py --writes`) est écrit et committé AVANT d'être exécuté ; sa date est celle de son commit.
Aucun résultat n'est regardé avant. Un autotest sur une matrice synthétique (`--selftest`) doit réussir avant l'analyse des vrais facteurs.

## Question
Le LoRA libre au rang 1, qui atteint la meilleure exactitude à budget égal, écrit-il dans la bande dominante, dans la bande mineure, ou ailleurs ?

## Sur quoi
Les facteurs appris du LoRA libre au rang 1, graines 2 à 6, au dernier pas :
- **tâche de format avec Llama-3.2-1B** : les runs existants, **au taux publié (10⁻³) si les cinq graines ont leurs facteurs, sinon à 2·10⁻³**,
  le seul autre taux où elles les ont (règle appliquée par le script, et le taux utilisé est affiché) ;
- **OpenBookQA avec Qwen2.5-1.5B** : aucun run n'a sauvegardé ses facteurs. **Réentraînement déclaré** : les cinq runs, au taux publié (2·10⁻³),
  à l'identique du run de référence, facteurs sauvegardés au dernier pas, épinglés sur RTX 4000 Ada, dans un dossier à part (`~/lora-runs-fw`).
  Leur exactitude est rapportée à côté de celle des runs d'origine.

## Mesures (pour chaque module de chaque couche)
- **Colonnes de B contre les bandes de U** (vecteurs singuliers à gauche de W₀) : la part de l'énergie de B, ‖P B‖²_F / ‖B‖²_F, dans la bande
  dominante (les k premiers), la bande mineure (les k derniers, les mêmes indices que le bras `bottom` et que MiCA), et une base orthonormée aléatoire de même
  taille (moyenne sur 100 tirages) ; **référence nulle k/d_out**.
- **Lignes de A contre les bandes de V** (vecteurs singuliers à droite) : les mêmes parts ; référence nulle k/d_in.
- **Les angles principaux** entre l'espace de B (resp. de A) et chaque bande.
- **Largeurs de bande** : **k = 2** (principale : les bandes des bras gelés à budget égal) ; k = 16 en complément.

## Agrégation et lecture
Moyenne sur les couches, par type de module, puis sur les graines ; intervalle à 95 % sur les cinq graines. Rapporté sans test : les parts
d'énergie divisées par la référence nulle, bande par bande, avec leurs intervalles. Lecture descriptive, sans seuil : une part au-dessus de
la référence nulle indique que le LoRA libre écrit préférentiellement dans cette bande.
