# Second additif à `PROTOCOL_scale.md` — onze runs de sélection relancés à l'identique

Écrit et committé AVANT la relance ; sa date est celle de son commit.

Sur les 70 runs de l'étape de sélection (Qwen2.5-7B, graines 0 et 1), **59 se sont terminés, 10 ont échoué et 1 n'a pas démarré**.
Les dix échecs ont trois causes, toutes liées à la machine et non au bras ni au taux : mémoire PARTITION épuisée (3 : certaines machines
ont moins de mémoire libre que celle de la sonde), environnement Python cassé (2 : l'environnement vit dans `/tmp`, propre à chaque
machine), disque local plein (5 : le cache du modèle, 15 Go, est sur `/Data`, propre à chaque machine). Aucun de ces runs n'a produit
de score : leur issue est inconnue.

**Ces onze runs sont relancés à l'identique** — mêmes bras, mêmes taux, mêmes graines ; rien n'est ajouté — **uniquement sur les
machines où un run du 7B s'est déjà terminé** (`scripts/nodes_for_7b.py`), qui ont montré assez de mémoire, un environnement sain et
la place pour le modèle. Les 59 runs terminés ne sont pas touchés. La sélection se fait ensuite, comme prévu, sur les 70 runs complets. **Les 35 runs du rapport tournent, eux aussi, sur ces seules machines éprouvées**, pour la même raison.
Si un run échoue encore pour une cause de machine, il est relancé une fois de la même façon ; au-delà, son bras est déclaré incomplet et
la campagne est rapportée sans lui.
