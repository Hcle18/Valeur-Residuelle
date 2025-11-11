# Sources de données

## Vue d'ensemble

Le projet utilise **deux sources principales** de données pour construire un dataset complet permettant le calcul de la valeur résiduelle des véhicules.

## Source 1 : Annonces véhicules d'occasion

### [autohero.com](https://autohero.com)
- **Type** : Site de vente de véhicules d'occasion
- **Méthode** : Web scraping avec Selenium
- **Données collectées** :
  - Prix de vente
  - Marque et modèle
  - Année de mise en circulation
  - Kilométrage
  - Carburant
  - Transmission
  - Puissance
  - État du véhicule

### Critères de sélection
```python
# Filtres appliqués lors du scraping
ANNEE_MIN = 2017
KILOMETRAGE_MAX = 100000

# Justification des critères
```

!!! info "Justification des critères"
    - **Année ≥ 2017** : Véhicules récents 
    - **Kilométrage ≤ 100 000 km** : Véhicules en bon état général
    - **Focus sur la qualité** : Données plus fiables et représentatives

### Structure des données brutes

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `prix` | float | Prix affiché (€) | 25000.0 |
| `marque` | string | Marque du véhicule | "BMW" |
| `modele` | string | Modèle du véhicule | "Série 3" |
| `annee` | int | Année de mise en circulation | 2019 |
| `kilometrage` | int | Kilométrage (km) | 45000 |
| `carburant` | string | Type de carburant | "Essence" |
| `transmission` | string | Type de transmission | "Automatique" |
| `puissance` | int | Puissance (CV) | 150 |

## Source 2 : Fiches techniques et prix neufs

### [caradisiac.com](https://www.caradisiac.com/fiches-techniques/)
- **Type** : Site de fiches techniques automobiles
- **Méthode** : Web scraping ciblé avec Selenium
- **Données collectées** :
  - Prix catalogue neuf
  - Caractéristiques techniques détaillées
  - Équipements de série
  - Consommation officielle

### Données de référence

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `prix_neuf` | float | Prix catalogue neuf (€) | 35000.0 |
| `consommation` | float | Consommation mixte (L/100km) | 6.2 |
| `emissions_co2` | int | Émissions CO2 (g/km) | 142 |
| `cylindree` | int | Cylindrée (cm³) | 1998 |
| `nb_places` | int | Nombre de places | 5 |

## Processus de collecte

### 1. Scraping des annonces
```python
# Configuration Selenium
driver_options = webdriver.ChromeOptions()
driver_options.add_argument('--headless')
driver_options.add_argument('--no-sandbox')

# Exemple de script de scraping
def scrape_autohero():
    # Initialisation du driver
    # Navigation et collecte
    # Sauvegarde en CSV
    pass
```

### 2. Collecte des prix neufs
```python
def scrape_prix_neuf():
    # Mapping marque/modèle
    # Requêtes vers fiches techniques
    # Extraction prix et caractéristiques
    pass
```

## Qualité des données

### Statistiques de collecte

| Métrique | Valeur |
|----------|--------|
| **Nombre d'annonces collectées** | 2352  |
| **Période de collecte** | 2025-04-09 |
| **Couverture marques** | 34 marques |
| **Couverture modèles** | 221 modèles |

### Problèmes identifiés

!!! warning "Points d'attention"
    - **Doublons** : Même véhicule sur plusieurs annonces
    - **Données manquantes** : Certains champs non renseignés
    - **Incohérences** : Prix annoncé > Prix neuf sur quelques annonces

### Solutions mises en place

```python
# Déduplication
df.drop_duplicates(subset=['marque', 'modele', 'annee', 'kilometrage'], keep='first')

# Validation des données
def validate_data(df):
    # Prix cohérents
    df = df[(df['prix'] > 5000) & (df['prix'] < 100000)]
    # Kilométrage réaliste
    df = df[df['kilometrage'] < (2024 - df['annee']) * 25000]
    return df
```

## Organisation des fichiers

```
data/
├── raw_data/
│   ├── autohero.csv                 # Données brutes autohero
│   ├── prix_neuf_voitures_*.csv    # Données prix neufs par batch
│   └── *_error.csv                 # Logs d'erreurs
├── processed_data/
│   ├── preprocessed_data.csv       # Dataset final nettoyé
│   └── modeles_voitures.csv        # Référentiel modèles
```

## Mise à jour des données (à implémenter en cible)

### Fréquence
- **Données d'occasion** : Scraping trimestriel
- **Prix neufs** : Mise à jour trimestrielle
- **Validation** : Contrôles automatisés

### Pipeline automatisé
1. Déclenchement programmé
2. Scraping avec gestion d'erreurs
3. Validation et nettoyage
4. Intégration au dataset principal
5. Ré-entraînement du modèle si nécessaire