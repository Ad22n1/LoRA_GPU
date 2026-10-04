# Protocole pré-enregistré — le régime des méthodes visées : rang 16, benchmark public

Écrit et committé **avant** le lancement.

## Pourquoi
Tous nos résultats sont mesurés aux rangs 1 à 4 sur des tâches que nous avons construites. Les
méthodes comparées (PiCa, MiCA, PiSSA, MiLoRA) travaillent au rang 16 et au-delà, sur des benchmarks
publics. C'est l'objection qu'un relecteur formulera en premier, et cette campagne y répond ou la
confirme.

## Ce qui est mesuré, et où
- **Tâche** : OpenBookQA, 4 957 items d'entraînement, 400 de validation, 500 de test, quatre options,
  hasard à 25 %. C'est l'une des huit tâches de la suite de sens commun que PiCa rapporte.
  Différence assumée avec ces papiers : ils entraînent sur Commonsense170K et évaluent sur les huit
  tâches ; nous entraînons sur le train d'OpenBookQA et évaluons sur son test. Le papier le dira.
- **Modèle** : Qwen2.5-1.5B. Le 3B n'est pas retenu : en bf16 il pèse environ 6 Go, contre 2,2 Go
  libres sur le quota. Il ira dans la réplication de la version 2.
- **Rang 16**, sept modules, écrêtage 1,0. **Évaluation par vraisemblance des options**, la même que
  la tâche de connaissance : aucune génération, donc aucun format à apprendre. 500 items de test,
  soit une erreur binomiale d'environ deux points par run, qu'il faut garder en tête.
- **Référence `free` au rang 7** : $8\,078\,336$ paramètres contre $8\,142\,848$ pour un bras
  contraint au rang 16, soit **−0,8 %**. C'est le meilleur appariement disponible, et il est bien
  meilleur que les +7,5 % des rangs bas ; la référence est ici légèrement **plus petite** que les
  bras contraints, donc le biais résiduel joue contre elle, pas contre eux. Le rang 8 donnerait
  +13,4 % et n'est pas retenu.

## Ce qui est lancé
1. **Balayages** sur les graines de sélection 0 et 1, sept taux de 1e-4 à **2e-2** (la grille monte à
   2e-2 d'emblée, pour ne pas retomber sur des sélections en bordure comme aux rangs bas) :
   `top`, `bottom`, `random_ortho` au rang 16 (42 runs) et `free` au rang 7 (14 runs).
   Toute sélection au bord entraîne un prolongement d'un cran et une nouvelle sélection.
2. **Les deux taux communs**, 2e-3 et 1e-2, graines 2 à 6, pour les trois bras contraints (30 runs).
3. **Les taux propres**, graines 2 à 6, pour les quatre bras (20 runs), lancés après la sélection.

Les facteurs ne sont pas sauvegardés : au rang 16 ils satureraient le quota. L'oubli n'est pas
mesuré ici.

## Ce qui sera conclu, fixé d'avance
**Les deux taux communs sont définis par une règle, pas choisis.** Ce sont le taux que la sélection
retient pour `bottom` et celui qu'elle retient pour le meilleur des autres bras, sur les graines 0 et
1. Sans cette règle, un relecteur objecterait à juste titre qu'on peut toujours trouver deux taux qui
inversent un ordre.

**L'inversion.** Un critère sur l'ordre des moyennes ne suffit pas ici : sur 500 items de test,
l'erreur binomiale d'un run est d'environ deux points, comparable aux effets attendus, et « premier
puis dernier » peut sortir du bruit. L'inversion est donc déclarée reproduite si, sur les graines 2 à
6 appariées, **chacune des deux moitiés passe son test apparié** au seuil bilatéral de $2{,}776$ :
l'avance de `bottom` au taux bas et son retard au taux haut. Rappel de ce qui est arrivé à $1{,}5$
milliard au rang 2 : une seule des deux moitiés passait.
- **Si elle se reproduit** : le premier résultat du papier vaut dans le régime des méthodes visées,
  et le papier le dira ainsi.
**Test de validité, avant toute conclusion.** Notre tâche de connaissance est restée près du hasard
jusqu'au rang 64 ; si OpenBookQA fait de même ici, une non-reproduction ne prouverait rien. Le bras
`free` doit donc dépasser le modèle de base d'au moins **5 points** d'exactitude, marge fixée ici.
S'il ne le fait pas, l'issue est **« non informatif »**, et non « échec » : la campagne ne dit alors
rien sur l'inversion, et le papier l'écrira ainsi.

**Issue partielle.** Si une seule moitié passe son test, c'est elle qui est rapportée et l'autre est
dite non établie, comme sur Llama et sur Qwen2.5-1.5B, où ce n'était pas la même moitié dans les
deux cas. Le papier dira alors que ce que le taux commun établit au rang 16 est cette moitié-là.

- **Si elle ne se reproduit pas** — ce qui est possible : à 1,5B au rang 2, la moitié « déficit »
  n'était déjà pas établie ($t = 2{,}14$) — **le papier l'écrira sans détour** : l'inversion est
  mesurée aux rangs 1 à 4 et ne se reproduit pas au rang 16 sur cette tâche, et le premier résultat
  sera restreint aux rangs bas dans le titre, le résumé et la contribution 1.
- **Si une seule moitié est établie**, c'est celle-là qui est rapportée, et l'autre est dite non
  établie, comme pour Qwen2.5-1.5B au rang 2.

**Gel contre choix.** Avec la référence `free` à −0,8 %, le gel est `free` (r=7) moins le meilleur
bras contraint, et le choix l'étendue entre bras contraints, aux taux propres. Les deux sont
rapportés avec leur intervalle bootstrap apparié, et la comparaison au rang 16 est ajoutée au
tableau des populations, quelle que soit sa direction.

## Engagements
- Les trois cas de l'inversion sont écrits tels qu'ils sortent, y compris celui qui restreint le
  premier résultat.
- Aucune graine ni aucun taux ajouté après coup, hors le prolongement de grille prévu.
- La place disque est vérifiée avant le lancement ; si elle manque, la campagne est réduite en
  nombre de graines, jamais en nombre de bras.
