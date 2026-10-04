# Additif au protocole d'OpenBookQA au rang 2 — un départage exact, réglé avant la seconde étape

Écrit et committé le 25/09, après la sélection et **avant tout run sur les graines rapportées**.

## Ce qui a été constaté
La règle fixe le second taux commun sur « le meilleur autre bras gelé ». En validation (graines 0 et 1), deux bras sont
**exactement à égalité** : `random` à 5·10⁻³ et `random_ortho` à 2·10⁻², tous deux à **0,411250** (`top` suit à 0,410000).
La règle ne prévoyait pas de départage ; le script de lecture avait retenu `random` par le seul ordre de sa liste.

## Ce qui est décidé, sans rien avoir vu des graines rapportées
Aucun des deux n'est préféré. **Les deux paires de taux communs sont testées et rapportées** :
- paire A : 2·10⁻³ (`bottom`) et 5·10⁻³ (`random`) ;
- paire B : 2·10⁻³ (`bottom`) et 2·10⁻² (`random_ortho`).
Les seuils sont corrigés pour ces deux tests (Bonferroni) : chaque moitié de l'inversion, **t > 3,495** (au node de 2,776) ;
le test d'interaction, trois concurrents fois deux paires, **t > 4,851**. L'inversion est déclarée établie si elle l'est
pour au moins une paire, et le papier rapporte les deux.

## Runs ajoutés
`top`, `bottom` et `random` à 2·10⁻² sur les graines 2 à 6 (15 runs), pour que la paire B soit complète ; `random_ortho` à
2·10⁻² y est déjà (taux propre).
