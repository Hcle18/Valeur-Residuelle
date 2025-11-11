# Modélisation

## Vue d'ensemble

La phase de modélisation vise à prédire la **valeur résiduelle** des véhicules d'occasion en utilisant des algorithmes de machine learning avancés.

### Objectif
Prédire le **taux de décote** : $\text{Taux de décote} = \frac{\text{Prix occasion}}{\text{Prix neuf}}$

## Préparation des données

### Variables d'entrée (features)

| Catégorie | Variables | Type | Exemple |
|-----------|-----------|------|---------|
| **Véhicule** | marque, modele, annee | Catégoriel/Numérique | BMW, X3, 2019 |
| **Caractéristiques** | puissance, cylindree, carburant | Numérique/Catégoriel | 150 CV, 1998 cm³, Diesel |
| **Usage** | kilometrage, age_vehicule | Numérique | 45000 km, 3 ans |
| **Marché** | segment, gamme | Catégoriel | SUV, Premium |

### Variables dérivées (feature engineering)

```python
# Exemples de features engineered
def create_features(df):
    # Age du véhicule
    df['age_vehicule'] = 2024 - df['annee']
    
    # Kilométrage par an
    df['km_par_an'] = df['kilometrage'] / df['age_vehicule']
    
    # Ratio puissance/poids (approximé)
    df['puissance_specifique'] = df['puissance'] / df['cylindree'] * 1000
    
    # Catégorie de kilométrage
    df['categorie_km'] = pd.cut(df['kilometrage'], 
                               bins=[0, 20000, 50000, 80000, float('inf')],
                               labels=['Faible', 'Moyen', 'Élevé', 'Très élevé'])
    
    # Segment véhicule (basé sur prix neuf)
    df['segment'] = pd.cut(df['prix_neuf'],
                          bins=[0, 20000, 35000, 50000, float('inf')],
                          labels=['Économique', 'Compact', 'Familial', 'Premium'])
    
    return df
```

## Algorithmes testés

### 1. XGBoost (Extreme Gradient Boosting)

**Avantages** :
- Excellent pour les données tabulaires
- Gestion native des valeurs manquantes
- Robuste au surapprentissage
- Interprétabilité via SHAP

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score

# Configuration XGBoost
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Entraînement
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train)
```

### 2. CatBoost

**Avantages** :
- Traitement automatique des variables catégorielles
- Peu de préprocessing requis
- Performances compétitives
- Gestion du sur-apprentissage

```python
from catboost import CatBoostRegressor

# Variables catégorielles
cat_features = ['marque', 'modele', 'carburant', 'transmission', 'segment']

# Configuration CatBoost
catboost_params = {
    'iterations': 1000,
    'depth': 6,
    'learning_rate': 0.1,
    'cat_features': cat_features,
    'verbose': False,
    'random_seed': 42
}

# Entraînement
catboost_model = CatBoostRegressor(**catboost_params)
catboost_model.fit(X_train, y_train)
```

### 3. Random Forest

**Avantages** :
- Robuste et stable
- Peu sensible aux hyperparamètres
- Importance des features native
- Bon baseline

```python
from sklearn.ensemble import RandomForestRegressor

# Configuration Random Forest
rf_params = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

# Entraînement
rf_model = RandomForestRegressor(**rf_params)
rf_model.fit(X_train, y_train)
```

## Pipeline de preprocessing

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Définition des transformateurs
numeric_features = ['kilometrage', 'puissance', 'cylindree', 'age_vehicule']
categorical_features = ['marque', 'modele', 'carburant', 'transmission']

# Pipeline pour variables numériques
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline pour variables catégorielles
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('encoder', LabelEncoder())
])

# Combinaison des transformateurs
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Pipeline complet
complete_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(**xgb_params))
])
```

## Validation et sélection du modèle

### Stratégie de validation

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Validation temporelle (important pour les données de prix)
tscv = TimeSeriesSplit(n_splits=5)

# Métriques d'évaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X, y, cv=tscv):
    """Évaluation complète d'un modèle"""
    
    # Cross-validation
    mae_scores = cross_val_score(model, X, y, cv=cv, 
                                scoring='neg_mean_absolute_error')
    rmse_scores = cross_val_score(model, X, y, cv=cv, 
                                 scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(model, X, y, cv=cv, 
                               scoring='r2')
    
    return {
        'MAE': -mae_scores.mean(),
        'RMSE': np.sqrt(-rmse_scores.mean()),
        'R²': r2_scores.mean(),
        'MAE_std': mae_scores.std(),
        'RMSE_std': np.sqrt(rmse_scores.std()),
        'R²_std': r2_scores.std()
    }
```

## Hyperparameter tuning

### Optimisation bayésienne avec Optuna

```python
import optuna

def objective(trial):
    """Fonction objectif pour l'optimisation des hyperparamètres"""
    
    # Hyperparamètres à optimiser
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
    }
    
    # Entraînement avec validation croisée
    model = xgb.XGBRegressor(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                            scoring='neg_mean_absolute_error')
    
    return -scores.mean()

# Optimisation
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f"Meilleurs hyperparamètres: {study.best_params}")
```

## Analyse des résultats

### Importance des features

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Importance XGBoost
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualisation
plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(15), 
            x='importance', y='feature')
plt.title('Top 15 - Importance des Features (XGBoost)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
```

### Analyse SHAP

```python
import shap

# Valeurs SHAP pour interprétabilité
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Graphiques SHAP
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)
```

## Modèle final sélectionné

Après évaluation, **XGBoost** a été retenu comme modèle de production :

### Hyperparamètres optimaux
```python
final_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 1500,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42
}
```

### Sauvegarde du modèle

```python
import joblib

# Sauvegarde du pipeline complet
joblib.dump(complete_pipeline, 'models/xg_boost_cst_reg.joblib')

# Sauvegarde des transformateurs
joblib.dump(preprocessor, 'models/transform_pipeline.joblib')

print("✅ Modèle sauvegardé avec succès")
```

### Utilisation en production

```python
# Chargement du modèle
model = joblib.load('models/xg_boost_cst_reg.joblib')

# Prédiction
def predict_residual_value(car_features):
    """Prédiction de la valeur résiduelle"""
    prediction = model.predict([car_features])
    return prediction[0]

# Exemple d'utilisation
car_example = {
    'marque': 'BMW',
    'modele': 'X3',
    'annee': 2020,
    'kilometrage': 35000,
    'puissance': 190,
    'carburant': 'Diesel'
}

taux_decote = predict_residual_value(car_example)
print(f"Taux de décote prédit: {taux_decote:.3f}")
```