# Speakli Quality Framework

Framework d'évaluation modulaire conçu pour mesurer la fiabilité du pipeline de transcription (STT) et de structuration clinique (LLM) de la solution Speakli.

## Objectif
L'outil permet de quantifier la précision de l'extraction de données médicales à partir de flux vocaux, en identifiant les écarts entre la vérité terrain (Ground Truth) et les prédictions du modèle.

## Métriques Implémentées

### Qualité de Transcription (STT)
* **Word Error Rate (WER)** : Mesure la distance de Levenshtein au niveau des mots pour évaluer la performance brute du moteur de reconnaissance vocale.
* **Resident Similarity** : Calcul de similarité textuelle sur l'identité du patient pour quantifier les risques d'erreurs d'attribution (Fuzzy Matching).

### Qualité de Structuration (JSON)
* **Token Overlap (Recall)** : Mesure le taux de récupération des concepts clés pour les transmissions narratives et ciblées, assurant l'exhaustivité de l'information.
* **Extraction Quality** : Évaluation de la complétude et de l'exactitude des constantes vitales après normalisation numérique.

### Sécurité Clinique
* **Clinical Safety Scanner** : Détection automatique des hallucinations critiques (termes médicaux sensibles inventés par le modèle tels que "chute" ou "stade d'escarre").
* **Value Mismatch Tracking** : Identification des erreurs de saisie numérique sur les constantes vitales (écarts de valeurs significatifs).

## Installation et Utilisation

### Dépendances
Le framework nécessite Python 3.8+ et les bibliothèques listées dans `requirements.txt`.

```bash
pip install -r requirements.txt


### Exécution
Pour lancer l'analyse complète du dataset par défaut :
```bash
python evaluate.py