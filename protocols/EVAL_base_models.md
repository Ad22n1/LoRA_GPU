# Évaluations du modèle de base (sans entraînement) — lignes « modèle de base » du tableau central

Écrit et committé avant les runs. Aucun test : ce sont des mesures de référence, rapportées telles quelles.
Arrêt des lancements fixé au **lundi 28/09 à 20 h**.

- Mode `baseline` du code : le modèle non adapté est évalué, puis le run s'arrête (aucun pas d'entraînement).
- **Quatre configurations**, chacune construite sur la base de sa campagne : tâche de format sur Llama-3.2-1B (`sigma_residual_C2.yaml`) et
  sur Qwen2.5-1.5B (`scale_qwen15.yaml`) ; OpenBookQA sur Qwen2.5-7B et sur Mistral-7B-v0.3 (`obqa_r2_common.yaml`, modèle remplacé).
- **Graines neuves 150 et 151** (une évaluation ancienne existe sur la graine 0 ; des graines neuves garantissent de nouveaux runs, épinglés) :
  l'évaluation ne dépend d'aucun tirage, les deux doivent donner le même score (contrôle du déterminisme).
- **8 runs**, épinglés sur RTX 4000 Ada. Rapporté : le score de chaque configuration, et son écart entre les deux graines.
