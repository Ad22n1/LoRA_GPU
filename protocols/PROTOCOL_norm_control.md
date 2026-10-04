# Protocole pré-enregistré — position ou norme ? Contrôle à norme égale (point 6)

Écrit et committé le 25/09, avant tout run.

## Question
Le retard de `bottom` au rang 2 tient-il quand les quatre bras ont exactement la même norme de mise à jour ?

## Ce qui existe, et pourquoi on ne le prolonge pas tel quel
Deux cibles ont tourné (`norm_matched_low_1b`, `norm_matched_1b`) : 1,5 et 4,5, graines 2-6. Aucune ne laisse les quatre bras
apprendre (à 1,5, seul `bottom` apprend ; à 4,5, tous sauf lui). Leurs taux ont été remplis par le lanceur : 10⁻² pour `top`,
`random` et `random_ortho`, mais **2·10⁻³ pour `bottom`**, son taux d'avant le départage. Elles restent un résultat à part, décrit
comme tel. Une grille `magnitude_1b` (cibles 0,2 à 1) n'a laissé aucun run : jamais lancée, ou entièrement échouée — déclaré.

## Montage
Llama-3.2-1B, format, rang 2, sept modules, écrêtage 1,0, α = r, bras aléatoires salés. Chaque bras à son taux retenu ACTUEL, écrit
explicitement : **10⁻² pour `top`, `random`, `random_ortho` ; 5·10⁻³ pour `bottom`**. **Graines fraîches 105 à 109** (la zone des
cibles a été choisie d'après la figure « score contre norme », tracée sur les graines 2 à 6 ; d'autres graines évitent la circularité).

**Chaque couple (bras, graine) est entraîné UNE fois, sans norme cible** (20 runs). La remise à l'échelle (`rescale_update_`)
s'applique à la mise à jour finale, après l'entraînement : l'entraînement ne dépend donc pas de la cible. Chaque run est ensuite
rechargé (`scripts/norm_rescale_eval.py`) et évalué :
1. **à sa propre norme, sans remise à l'échelle** — cette évaluation doit reproduire le score enregistré en fin d'entraînement, à
   trois items près sur 614 (test) et sur 500 (validation) ; sinon le rechargement est faux et rien d'autre n'est mesuré. Elle sert
   aussi de **référence** : le retard de `bottom` sur les graines 105-109 à norme libre est rapporté à côté du test ;
2. **à chacune des sept normes cibles, fixées ici : 1,5 / 2,0 / 2,5 / 3,0 / 3,5 / 4,0 / 4,5**.

**Cible utilisable** (définie ici) : les quatre bras y dépassent 0,3 de score moyen.

## Test, issues écrites d'avance
Statistique : pour chaque graine, l'écart moyen, sur les cibles utilisables, entre chacun des trois autres bras et `bottom`.
Test apparié sur 5 graines, **t > 3,96** (bilatéral, 4 d.l., Bonferroni sur trois comparaisons).
1. **Aucune cible utilisable** → contrôle inutilisable, rapporté tel quel ; conclusion du papier inchangée.
2. **`bottom` derrière chacun des trois bras** → l'effet de position ne s'explique pas par la norme.
3. **Derrière un ou deux bras seulement** → partiel ; rapporté bras par bras, sans conclure pour la position en général.
4. **Aucun écart établi** → l'écart entre positions est compatible avec un effet de norme ; rapporté avec la référence à norme libre.

## Analyse secondaire, descriptive, écrite d'avance
La courbe du score en fonction de la norme pour chaque bras, et l'écart de `bottom` cible par cible, sans test.

## Interprétation, fixée ici
Ce contrôle compare **les directions apprises** par chaque bras, **à amplitude égale** ; il ne teste pas un entraînement à norme
égale. Le papier le dira en ces termes.

## Engagements
Rapporté quelle que soit l'issue ; aucune cible ni graine ajoutée.
