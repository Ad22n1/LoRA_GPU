# Protocole pré-enregistré — le rang 16 à grande échelle (OpenBookQA, Qwen2.5-7B)

Écrit et committé AVANT tout run ; sa date est celle de son commit.
**Arrêt des lancements : mardi 29/09 à 20 h.** Ce qui n'est pas parti à cette heure n'entre pas dans le rapport.

## Pourquoi
Au rang 16, l'isolation et les bras à A gelé n'existent que sur Qwen2.5-1.5B. Ce protocole les porte sur Qwen2.5-7B.

## Budgets exacts (par couche)
B gelé au rang 16 : 647 168 ; **LoRA libre de budget égal : r = 7**, 630 784 (**−2,5 %** ; r = 8 : +11,4 %) ; A gelé au rang 16 : +22,8 %.

## Étape 0 : la sonde mémoire, avec sa règle
Un run de `top_unfrozen` au rang 16 (A et B entraînés, le bras le plus gourmand), au taux le plus haut de la grille, graine 0, sur une machine
éprouvée pour ce modèle. **Règle** : pic mémoire ≤ 18,5 Go et durée ≤ 45 minutes → la campagne part ; sinon, **elle s'arrête, sans repli**.
La sonde et son verdict sont committés avant tout autre run.

## Montage
- **Bras** : ceux du protocole des rangs 4 et 8, au rang 16 : `top`, `bottom` ; `top_unfrozen`, `bottom_unfrozen` aux rangs 7 et 16 ;
  `dual_top`, `dual_bottom` ; `free` au rang 7. **9 bras.**
- Configuration reprise d'un run de référence d'OpenBookQA sur Qwen2.5-7B ; seuls changent le mode, le rang (α = r), le taux et la graine.
- **Grille plus basse** qu'au rang 2, comme au rang 16 sur ARC : cinq taux, 10⁻⁴ à 2·10⁻³, **un cran de prolongement** (5·10⁻⁵ ou 5·10⁻³).
- Sélection sur les graines 0 et 1, rapport sur les graines 2 à 6 ; **deux vagues**, G committé entre les deux, comme aux rangs 4 et 8.
- Machines éprouvées pour Qwen2.5-7B (RTX 4000 Ada). **Environ 135 runs, 50 à 65 PARTITION-heures** (un run de 7B sur OpenBookQA prend 20 à 28 minutes).
- **Facteurs non sauvegardés** : le quota personnel (30 Go, dont 28 déjà utilisés) ne le permet pas. Le lanceur vérifie ce quota avant chaque étape.
  La sonde est aussi un run de sélection (graine 0) : elle n'est pas refaite.

## Tests et lecture
Les quatre tests du protocole des rangs 4 et 8, avec les mêmes seuils, planchers et orientation de l'écart entre bandes ; U/G et I = G − U avec
leurs intervalles. Rapporté quelle que soit l'issue ; une relance pour échec machine.

## Additif du 28/09, committé avant toute relance supplémentaire
Deux runs de sélection ont échoué deux fois, avant d'avoir produit le moindre résultat, par manque de mémoire causé par d'autres processus
présents sur la carte (sessions d'autres utilisateurs des salles partagées ; jusqu'à six processus et plus de 7 Go occupés) : `top` au rang 16
à 10⁻³ (graine 1) et `top_unfrozen` au rang 7 à 10⁻⁴ (graine 0). Un run de cette campagne demande environ 16,5 Go sur 19,5 : il tient seul sur
la carte, pas avec des voisins. La relance unique prévue a été consommée par la première de ces pannes. **Une seconde relance, à l'identique,
est autorisée pour ces deux runs seulement**, parce que leurs journaux prouvent une panne de machine et qu'ils n'ont produit aucun résultat ;
aucun autre run n'est relancé à ce titre. Si l'un d'eux échoue encore, la campagne s'arrête à la sélection et elle est rapportée comme incomplète.
