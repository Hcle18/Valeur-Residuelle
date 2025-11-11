# Proof of Concept Valeur Résiduelle - Documentation

!!! info "À propos du projet"
    Projet de développement d'un modèle de machine learning pour l'estimation des prix de véhicules d'occasion et de leur courbe de décote, avec une démo de l'application web intégrée.

## Vue d'ensemble

Ce projet se déroule en **deux phases principales** :

### Phase 1 : Développement du modèle ML
- **Objectif** : Créer un modèle de machine learning pour estimer le prix de vente des véhicules d'occasion
- **Données** : Scraping d'annonces des voitures d'occasion sur autohero.com pour la **Valeur Résiduelle** et scraping fiches techniques de Caradisiac pour obtenir le **Prix neuf**
- **Critères** : Véhicules avec mise en circulation >= 2017, kilométrage <= 100 000 km
- **Métrique clé** : Taux de décote = VR/Prix neuf

### Phase 2 : Application web
- **Objectif** : Développer une application web avec Dash
- **Fonctionnalité** : Calculateur de valeur résiduelle intégrant le modèle ML
- **Interface** : Interface utilisateur pour les estimations

## Structure du projet

```
Valeur-Residuelle/
├── data/                    # Données brutes et traitées
├── notebooks/               # Notebooks Jupyter d'analyse
├── src/                     # Code source principal
├── models/                  # Modèles entraînés et pipelines
├── app.py                   # Application web principale
└── docs/                    # Documentation
```

## Technologies utilisées

- **Machine Learning** : scikit-learn, XGBoost, CatBoost
- **Traitement des données** : pandas, numpy
- **Visualisation** : matplotlib, seaborn, plotly
- **Application web** : Dash
- **Scraping** : Selenium

## Démarrage rapide

Pour commencer avec le projet :

1. **Installation** : Voir [Installation](dev/installation.md)
2. **Exploration des données** : Consultez [EDA](data/eda.md)
3. **Modèles** : Découvrez la [Modélisation](models/modeling.md)
4. **Application** : Testez l'[Interface](app/interface.md)

## Navigation

### 📊 Données
- [Sources de données](data/sources.md) - Origine et collecte des données
- [Preprocessing](data/preprocessing.md) - Nettoyage et transformation
- [Exploration](data/eda.md) - Analyse exploratoire

### 🤖 Modèles
- [Modélisation](models/modeling.md) - Développement des modèles
- [Évaluation](models/evaluation.md) - Métriques et validation
- [Performances](models/performance.md) - Résultats et comparaisons

### 🌐 Application
- [Interface](app/interface.md) - Interface utilisateur
- [API](app/api.md) - Documentation de l'API
- [Déploiement](app/deployment.md) - Guide de déploiement

---

*Documentation générée avec MkDocs Material pour le projet Valeur Résiduelle - NEXIALOG*