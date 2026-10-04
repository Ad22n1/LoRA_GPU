# Protocole — Σ ou le gel : le bras `top_sigma`

Écrit et committé **avant** le lancement.

## La question
Le taux optimal se déplace avec la position de la bande, dans le sens opposé à celui rapporté pour
PiSSA : chez nous `bottom` (bande mineure) veut un taux plus bas que `top` ; chez PiSSA, c'est
l'extrémité dominante qui veut le taux plus bas. Nos bras diffèrent de PiSSA par deux choses : ils
gèlent un facteur, et ils n'ont pas Σ (B = U). Le bras `top_sigma` garde le gel et rend Σ :
B = U_r √Σ_r sur la bande du haut, gelé, A entraîné depuis zéro, même budget que `top`.

## Ce qui est lancé
1. **Balayage** : `top_sigma`, Llama-3.2-1B, rang 2, les 7 modules, écrêtage 1,0, tâche format, dix taux
   de 2e-5 à 2e-2, graines de sélection 0 et 1 (20 runs).
2. **Graines rapportées** : cinq graines, 2 à 6, au taux retenu (5 runs), lancées seulement après la
   sélection, sans autre modification.

## La règle de sélection, fixée d'avance
Le taux retenu maximise le score de validation moyen sur les graines 0 et 1, comme pour tous les bras.
Il doit être **à l'intérieur de la grille** ; s'il tombe sur un bord, la grille est prolongée d'un
cran dans ce sens et la sélection refaite. Si les deux premiers taux sont à moins d'une erreur-type
binomiale l'un de l'autre, l'égalité est rapportée et les deux taux sont lancés sur les graines 2 à 6.

## Ce qu'on conclura, fixé d'avance
Taux retenus existants : `top` 1e-2 ; `bottom` 2e-3 ou 5e-3 (égalité dans le bruit).
- **(A) taux de `top_sigma` ≤ 1e-3**, c'est-à-dire sous celui de `bottom` quel que soit le départage :
  rendre Σ suffit à retrouver le sens de PiSSA. Le papier dira que **Σ inverse le déplacement**.
- **(B) taux de `top_sigma` ≥ 1e-2**, celui de `top` : Σ ne déplace pas le taux ; le papier dira que
  **le déplacement ne vient pas de Σ**, ce qui désigne le gel comme la différence restante.
- **(C) entre les deux** : Σ déplace le taux en partie ; rapporté tel quel, **sans attribution**.

## Ce qui est su d'avance, et sera écrit
Avec Σ, les colonnes de B ont une norme √σ, plus grande en haut du spectre : chaque pas est plus grand
à taux égal. Un taux optimal plus bas (cas A ou C) est donc en partie attendu, par un effet d'échelle
que le §15 mesure déjà pour PiSSA. Le papier le dira à côté du résultat.

## Engagements
- Le résultat change au plus le résumé, la conclusion et un paragraphe du §7, et rien d'autre.
- Les scores des graines 2 à 6 sont rapportés, mais ne servent pas à la décision.
- Aucune graine ni aucun taux n'est ajouté après coup, sauf le prolongement de grille prévu ci-dessus.
