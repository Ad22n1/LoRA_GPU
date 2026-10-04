# Protocole pré-enregistré — la perte sur la réponse seule

Écrit et committé AVANT tout run ; sa date est celle de son commit.
**Arrêt des lancements : jeudi 01/10 a 08 h 00 (2026-10-01 08:00).** Ce qui n'est pas parti à cette heure n'entre pas dans le rapport.

## Question
Sur OpenBookQA, la perte couvre les tokens de la question autant que ceux de la réponse. L'asymétrie entre facteurs (geler A coûte beaucoup
moins que geler B) dépend-elle de cette supervision de la question ?

## Paire
OpenBookQA avec Qwen2.5-1.5B, rang 2 seulement.

## Données et perte
Même format « Question: … Answer: … », mêmes séquences empaquetées de 384 tokens, même ordre, même nombre de pas d'optimisation (49) que les
runs de référence. Seuls les tokens de la réponse sont supervisés ; ceux de la question sont masqués dans la perte.
*Mise en œuvre, précisée avant commit* : les blocs sont ceux que produit le chargeur, inchangés ; seules les étiquettes changent. Sont
supervisés les tokens qui suivent « Answer: », jusqu'au retour à la ligne qui termine l'exemple, non supervisé ; une réponse coupée entre
deux blocs reste supervisée des deux côtés. Le nombre de pas est fixé à 49 dans la configuration (`max_steps`).
Déclaré : avec la perte sur la réponse seule, le nombre de tokens supervisés est bien plus faible qu'avec la perte complète. On garde les
mêmes pas et les mêmes séquences, et non le budget de 600 000 tokens supervisés, qui demanderait des dizaines d'époques. Le nombre de tokens
supervisés de chaque run est rapporté.
Test du nouveau chargeur, committé avant tout run (`scripts/test_answer_only.py`, exécuté sur le serveur par le déploiement) : blocs
identiques à la référence et dans le même ordre, étiquettes masquées ou égales à leur token, une réponse supervisée par marque « Answer: »,
réponses affichées en clair, option refusée hors d'OpenBookQA et d'ARC, aucun changement pour les autres modes (identifiants inchangés).

## Bras
`top`, `bottom` (B gelé), `dual_top`, `dual_bottom` (A gelé), LoRA libre au rang 1, tous sous la perte sur la réponse seule.

## Taux et graines
Cinq taux par bras : 5·10⁻⁴, 10⁻³, 2·10⁻³, 5·10⁻³, 10⁻². Sélection en validation sur les graines 0 et 1, rapport sur les graines 2 à 6.
Un cran de prolongement si l'optimum tombe au bord de la grille. Évaluation inchangée (vraisemblance moyenne par token de chaque option).

## Tests et lecture, fixés d'avance
Par bande : A gelé − B gelé (`dual_top` − `top`, `dual_bottom` − `bottom`), apparié par graine, t > 3,495 (deux bandes).
- Si A gelé bat B gelé sur les deux bandes : l'asymétrie ne dépend pas de la perte sur la question.
- Sinon : elle en dépend au moins en partie.

## Rapportés sans test
- G par bande dans ce régime : LoRA libre au rang 1 − B gelé.
- La comparaison avec la perte complète : G (8,04 et 12,36) et A gelé − B gelé (6,56 et 10,28).
- Chaque bras − le modèle de base (0,422).

## Règles communes
- Une seule carte (RTX 4000 Ada) pour tous les runs de la campagne.
- Une relance pour échec machine. Les runs abandonnés faute de mémoire libre sont relancés hors relance, au plus trois fois. Couvre-feu
  respecté : un run interrompu est repris la nuit suivante, sans compter comme relance.
- Rapporté quelle que soit l'issue. Aucune graine ni aucun taux ajoutés hors le cran de prolongement.
- Tout écart est consigné dans un addendum daté, committé avant le run qu'il concerne.
