# Analyse A6, post hoc et descriptive — où tombent les bases du préchauffage, et ce qu'était chaque coupe

Écrit et committé AVANT toute exécution, avec ses scripts (`scripts/analyses_v5.py`, `scripts/submit_grad32.py`) ; sa date est celle de
son commit. Statut : **post hoc · descriptif** (`PH · D`). Aucune lecture n'est fixée d'avance ; aucune affirmation confirmatoire du papier
ne reposera sur elle. Sorties : `analyses_v5.json`. Aucun entraînement.

## Partie 1 — Les coupes du préchauffage
Pour OpenBookQA (Qwen2.5-1.5B) et la tâche de format (Llama-3.2-1B), aux coupes k = 5, 10 et 25 % du run de direction du préchauffage
(`PROTOCOL_warm.md` ; pas 3, 5, 13 sur 49 et 14, 28, 69 sur 274) : le nombre de pas d'optimisation, de tokens supervisés, d'époques, et le
taux d'apprentissage — celui de la dernière mise à jour avant la coupe, et celui que l'ordonnancement atteint juste après.
**Méthode.** Rejeu exact de l'entraînement, sans modèle : même configuration (celle du run de direction), même chargeur mélangé à la même
graine, même accumulation (une mise à jour toutes les `grad_accum` micro-lots), même formule d'ordonnancement (importée de `train.py`).
**Contrôle.** À chaque pas journalisé par le run de direction, le nombre de tokens rejoué doit égaler le nombre journalisé ; au moindre
écart, l'analyse s'arrête.

## Partie 2 — Énergie des bases dans les sous-espaces du gradient initial et dans les bandes de W₀
**Entrées.** Les bases sauvegardées du préchauffage (`warm5`, `warm10`, `warm25`, sur les deux paires) et la base de `learned` : phase 2,
graine de direction 0, sur OpenBookQA ; réplique B, graine de direction 0, sur la tâche de format. Le gradient initial : celui de `grad`
(mêmes 64 séquences, graine 1000, modèle de base, sans mise à jour), recalculé en gardant ses **32 premiers vecteurs singuliers à gauche**
(`scripts/submit_grad32.py` ; deux jobs PARTITION de quelques minutes ; ce n'est pas un entraînement). Les bandes dominante et mineure de rang 2 de
W₀ (colonnes de U du cache SVD utilisé par les bras).
**Mesure.** Pour une base orthonormale Q de rang 2 et un sous-espace de base orthonormale P, l'énergie ‖PᵀQ‖²_F / 2 ; pour le top-k du
gradient, k = 2, 8 et 32 ; pour chaque bande de rang 2. Médiane sur les couches par type de module.
**Référence aléatoire.** L'énergie attendue d'une base aléatoire de rang 2 dans un sous-espace de dimension k de ℝ^{d_out} : k / d_out ;
donc 2 / d_out pour les bandes et pour k = 2, 8 / d_out et 32 / d_out pour k = 8 et 32. On rapporte l'énergie et son rapport à cette
référence.
**Contrôle.** Les 2 premières des 32 directions doivent redonner les bases de `grad` (recouvrement ≈ 1) ; leur empreinte SHA-256 est
rapportée.

## Règles communes
Rapporté quelle que soit l'issue ; une donnée manquante est dite, jamais remplacée en silence.
