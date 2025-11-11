# Installation et configuration

## Prérequis

### Système
- **Python** : Version 3.9 ou supérieure
- **Git** : Pour le clonage du repository
- **Chrome/Chromium** : Requis pour le web scraping

### Outils recommandés
- **IDE** : Visual Studio Code avec extensions Python
- **Terminal** : PowerShell (Windows) ou bash (Linux/macOS)
- **Gestionnaire de paquets** : pip (inclus avec Python)

## Installation

### 1. Cloner le repository

```bash
git clone https://github.com/Hcle18/Valeur-Residuelle.git
cd Valeur-Residuelle
```

### 2. Créer un environnement virtuel

=== "Windows (PowerShell)"
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

=== "Linux/macOS"
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configuration des variables d'environnement

Créer un fichier `.env` à la racine du projet :

```env
# Configuration base de données
DATABASE_URL=sqlite:///instance/car_data.db

# Configuration scraping
CHROME_DRIVER_PATH=/path/to/chromedriver
SCRAPING_DELAY=2

# Configuration application
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
```

## Vérification de l'installation

### Test des imports Python

```python
# Test des imports principaux
import pandas as pd
import numpy as np
import sklearn
import xgboost
import dash

print("✅ Tous les modules sont installés correctement")
```

### Test de l'application

```bash
# Lancer l'application de test
python app.py
```

L'application devrait être accessible sur `http://localhost:8050`

## Structure du projet après installation

```
Valeur-Residuelle/
├── app.py                          # Application principale
├── requirements.txt                # Dépendances Python
├── mkdocs.yml                      # Configuration documentation
├── .env                            # Variables d'environnement (à créer)
├── venv/                          # Environnement virtuel (créé)
├── data/                          # Données du projet
├── src/                           # Code source
├── models/                        # Modèles entraînés
├── notebooks/                     # Notebooks Jupyter
├── docs/                          # Documentation
└── tests/                         # Tests unitaires
```

## Configuration des outils de développement

### Visual Studio Code

Extensions recommandées :

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "yzhang.markdown-all-in-one",
    "james-yu.latex-workshop"
  ]
}
```

### Jupyter

Pour utiliser les notebooks :

```bash
# Installation si nécessaire
pip install jupyter

# Lancement
jupyter notebook
```

## Données d'exemple

### Télécharger les données de test

```bash
# Créer les dossiers de données
mkdir -p data/raw_data data/processed_data data/outil_data

# Les données d'exemple sont incluses dans le repository
# Vérifier la présence des fichiers
ls data/outil_data/sample_app_car_data.csv
```

### Initialisation de la base de données

```python
# Script d'initialisation
from src.app.database import init_db

# Créer les tables
init_db()
print("✅ Base de données initialisée")
```

## Dépannage

### Problèmes courants

#### 1. Erreur d'import de modules

```bash
# Vérifier l'environnement virtuel
which python  # doit pointer vers venv/bin/python

# Réinstaller les dépendances
pip install --upgrade -r requirements.txt
```

#### 2. Problème avec ChromeDriver

```bash
# Installation automatique
pip install webdriver-manager

# Ou téléchargement manuel depuis
# https://chromedriver.chromium.org/
```

#### 3. Erreur de permissions

=== "Windows"
    ```powershell
    # Modifier la politique d'exécution
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```

=== "Linux/macOS"
    ```bash
    # Vérifier les permissions
    chmod +x venv/bin/activate
    ```

### Logs et debug

Activation du mode debug :

```python
# Dans app.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Ou via variable d'environnement
export FLASK_ENV=development
```

## Mise à jour

### Mettre à jour les dépendances

```bash
# Sauvegarder l'état actuel
pip freeze > requirements_backup.txt

# Mettre à jour
pip install --upgrade -r requirements.txt

# En cas de problème, revenir à l'état précédent
pip install -r requirements_backup.txt
```

### Mise à jour du code

```bash
# Récupérer les dernières modifications
git pull origin main

# Installer les nouvelles dépendances si nécessaire
pip install -r requirements.txt
```

## Support

### Ressources utiles

- **Documentation officielle** : Cette documentation MkDocs
- **Repository GitHub** : [Valeur-Residuelle](https://github.com/Hcle18/Valeur-Residuelle)
- **Issues GitHub** : Pour signaler des bugs ou demander des fonctionnalités

### Contact

Pour toute question technique :
- Créer une issue sur GitHub
- Contacter l'équipe NEXIALOG