# Vue d'ensemble du projet

## Contexte et objectifs

Le projet **Valeur Résiduelle** vise à développer une solution complète pour l'estimation automatisée des prix de véhicules d'occasion et de leur courbe de décote.

### Problématique

- **Besoin** : Estimation précise de la valeur résiduelle des véhicules
- **Défis** : Multitude de facteurs influençant le prix (marque, modèle, année, kilométrage, état, etc.)
- **Solution** : Modèle de machine learning basé sur des données réelles du marché

## Architecture du projet

```mermaid
graph TD
    A[Collecte de données] --> B[Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Entraînement du modèle]
    D --> E[Évaluation]
    E --> F[Application Web]
    
    A1[autohero.com] --> A
    A2[caradisiac.com] --> A
    
    D --> D1[XGBoost]
    D --> D2[CatBoost]
    D --> D3[Random Forest]
    
    F --> F1[Interface utilisateur]
    F --> F2[API REST]
```

## Méthodologie

### 1. Collecte des données
- **Source 1** : Annonces véhicules d'occasion (autohero.com)
- **Source 2** : Fiches techniques et prix neufs (caradisiac.com)
- **Critères de sélection** :
  - Année de mise en circulation ≥ 2017
  - Kilométrage ≤ 100 000 km

### 2. Preprocessing et feature engineering
- Nettoyage des données manquantes
- Standardisation des formats
- Création de variables dérivées
- Encodage des variables catégorielles

### 3. Modélisation
- **Variable cible** : Taux de décote = Prix occasion / Prix neuf
- **Algorithmes testés** : XGBoost, CatBoost, Random Forest
- **Validation** : Cross-validation et jeu de test

### 4. Déploiement
- Application web interactive avec Dash
- API REST pour intégration
- Interface utilisateur intuitive

## Livrables

### Phase 1 - Modèle ML
- [x] Dataset nettoyé et préprocessé
- [x] Modèles entraînés et évalués
- [x] Pipeline de preprocessing
- [x] Métriques de performance

### Phase 2 - Application
- [x] Interface web fonctionnelle
- [x] API REST documentée
- [ ] Tests automatisés
- [ ] Documentation complète

## Équipe et responsabilités

| Rôle | Responsabilité |
|------|----------------|
| Data Scientist | Modélisation et feature engineering |
| Data Engineer | Pipeline de données et preprocessing |
| Développeur Web | Application et interface utilisateur |
| DevOps | Déploiement et infrastructure |

## Planning

```mermaid
gantt
    title Planning Projet Valeur Résiduelle
    dateFormat  YYYY-MM-DD
    section Phase 1
    Collecte données    :2024-01-01, 2024-02-01
    Preprocessing       :2024-02-01, 2024-02-15
    Modélisation        :2024-02-15, 2024-03-15
    Évaluation          :2024-03-15, 2024-03-31
    section Phase 2
    Développement app   :2024-04-01, 2024-05-15
    Tests et validation :2024-05-15, 2024-05-31
    Déploiement         :2024-06-01, 2024-06-15
```