# Protocole pré-enregistré — `top_unfrozen` : isoler le gel

Écrit et committé **avant** le lancement.

## Pourquoi
Le « coût du gel » du papier est l'écart entre `free` et le meilleur bras gelé. Ces deux bras diffèrent
par quatre choses : le gel de B, l'initialisation (A Kaiming et B = 0 contre B = U et A = 0), le rang
(1 contre 2) et le budget (+7,5 %). L'écart mesuré n'identifie donc pas le gel.
`top_unfrozen` ne change qu'une chose par rapport à `top` : **B est entraîné**. Il part de la même base
U, avec A = 0, donc le modèle est inchangé au pas zéro.

## L'appariement de budget, fixé d'avance
Entraîner B double le budget à rang égal. `top_unfrozen` est donc lancé **au rang 1**, où il entraîne
r·(d_in + d_out), soit le budget de `free` au rang 1 : le même appariement, à +7,5 %, que la référence
du papier contre un bras contraint au rang 2. Toute autre comparaison retomberait dans le défaut que
ce bras doit corriger.

## Ce qui est lancé
1. **Balayage** : `top_unfrozen`, Llama-3.2-1B, rang 1, les 7 modules, écrêtage 1,0, tâche format,
   sept taux de 1e-4 à 2e-2, graines 0 et 1 (14 runs) ; la grille monte à 2e-2, comme celle des balayages de position, pour ne pas recréer une sélection en bordure. Si l'optimum tombe au bord, la grille est
   prolongée d'un cran et la sélection refaite ; la sélection retenue est déclarée intérieure ou non.
2. **Graines rapportées** : 2 à 6, au taux retenu (5 runs par rang), lancées après la sélection. Ce sont
   les graines des cellules `top` et `free` existantes, pour que les trois comparaisons soient appariées.

## Trois comparaisons, parce qu'aucune n'isole tout
À budget fixé, entraîner B oblige à diviser le rang par deux : on ne peut donc pas isoler en même temps
le gel, le rang et le budget. Les trois comparaisons se complètent :

| comparaison | ce qui est commun | ce qui diffère |
|---|---|---|
| `top_unfrozen` r=1 − `free` r=1 | rang, budget | l'initialisation seule (B = U₁ contre B = 0) |
| `top_unfrozen` r=1 − `top` r=2 | budget (+7,5 %) | le gel **et** le rang |
| `top_unfrozen` r=2 − `top` r=2 | rang, initialisation | le gel seul, au prix d'un budget de 2,15× |

La troisième est la seule où le gel est seul en cause ; son budget n'est pas apparié, et le papier le dira.
Le rang 2 demande son propre balayage de taux, aux mêmes conditions que le rang 1.

## Les comparaisons, fixées d'avance
Soit G = `free` (r=1) − `top` (r=2) = **12,41 points** sur les graines 2 à 6, et
U = `top_unfrozen` (r=1) − `top` (r=2), mesuré sur les mêmes graines, test t apparié bilatéral.
(Le 11,5 du papier est l'écart entre `free` et le **meilleur** bras gelé, `random_ortho` ; ici la
comparaison se fait contre `top`, dont `top_unfrozen` partage la base.)
- **U ≥ 0,7·G** : le gel explique l'essentiel de l'écart. Le papier le dira, et nommera ce bras comme
  le contrôle qui identifie le gel.
- **U ≤ 0,3·G** : l'essentiel de l'écart vient de l'initialisation, de la base ou du rang, pas du gel.
  Le papier le dira et **reformulera la contribution 3 en conséquence**.
- **Entre les deux** : partagé entre les deux causes, rapporté comme tel, sans attribution.
En second, sans revendication : `top_unfrozen` − `free` (r=1), et les comptes de paramètres des trois
bras vérifiés contre leurs valeurs analytiques.

## Situer ce bras
`top_unfrozen` est proche du bras `init_top` de l'annexe PiSSA : même base U, mais sans √Σ et sans
soustraire la bande de W₀. C'est `top` avec B rendu entraînable, rien d'autre.

## La dérive du sous-espace, à mesurer sans quoi les comparaisons ne disent rien
Avec A = 0, le gradient de B est nul au premier pas, puis Adam déplace chaque entrée de B d'environ
le taux à chaque pas, alors que les entrées de U_r valent environ 1/√d_out ≈ 0,02. À 10⁻², B peut
donc quitter la bande dominante en quelques dizaines de pas. Si c'est le cas, ce bras est un LoRA
libre avec une autre initialisation, et la première comparaison ne mesure plus rien de spectral.
**Les angles principaux entre le B final et U_r sont donc calculés en fin de run et rapportés.**
Sans eux, les trois comparaisons sont ininterprétables ; avec eux, on sait laquelle dit quelque chose.

## Engagements
- Les deux comparaisons sont rapportées quel que soit leur sens.
- Aucune graine ni aucun taux n'est ajouté après coup, hors le prolongement de grille prévu ci-dessus.
- Avec A = 0, le gradient de B est nul au premier pas : B ne bouge qu'une fois A non nul. C'est attendu,
  et le papier le dira.
