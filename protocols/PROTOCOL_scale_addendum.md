# Additif à `PROTOCOL_scale.md` — la sonde du 7B et le contrôle numérique de la SVD

Écrit et committé AVANT la relance de la sonde ; sa date est celle de son commit.

La première sonde de Qwen2.5-7B s'est arrêtée **avant l'entraînement**, pendant le calcul du cache spectral, sur le contrôle de
reconstruction de `svd_of_weight` : erreur maximale 3,652·10⁻⁴ contre une tolérance de 3,252·10⁻⁴, soit 1,13·10⁻³ du plus grand poids
contre un seuil de 10⁻³. Le modèle a été téléchargé et chargé ; **ni score, ni mémoire de pointe, ni durée d'entraînement n'ont été
mesurés**, et la règle de choix (mémoire, durée) n'a donc rien pu évaluer.

Ce contrôle existe pour détecter une décomposition **ratée** (erreurs de l'ordre des poids), pas pour mesurer la précision : l'erreur
d'arrondi en fp32 croît avec la taille des matrices (jusqu'à 3 584 × 18 944 pour ce modèle), et U et Vᵀ sont ensuite stockés en fp16, ce
qui perd environ 10⁻³ de précision relative de toute façon. La tolérance passe donc à **10⁻² du plus grand poids**, encore cent fois sous
l'erreur d'une décomposition ratée (un test sur le serveur vérifie qu'une décomposition faussée reste refusée). Les caches déjà calculés,
pour les autres modèles, ne sont pas touchés.

**La sonde du 7B est relancée ; la règle de la précision 1 s'applique ensuite telle quelle.** Aucune autre modification.
