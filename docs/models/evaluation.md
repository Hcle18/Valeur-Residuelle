# Évaluation des modèles

## Vue d'ensemble

L'évaluation des modèles est cruciale pour s'assurer de la qualité et de la fiabilité des prédictions de valeur résiduelle. Cette section détaille les métriques, méthodologies et résultats d'évaluation.

## Métriques d'évaluation

### Métriques principales

| Métrique | Formule | Interprétation | Usage |
|----------|---------|----------------|-------|
| **MAE** | $\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y_i}\|$ | Erreur absolue moyenne | Erreur typique en € |
| **RMSE** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2}$ | Pénalise les grosses erreurs | Sensibilité aux outliers |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Variance expliquée (0-1) | Qualité globale |
| **MAPE** | $\frac{1}{n}\sum_{i=1}^{n}\frac{\|y_i - \hat{y_i}\|}{y_i} \times 100$ | Erreur relative (%) | Erreur en pourcentage |

### Métriques métier spécifiques

```python
def business_metrics(y_true, y_pred, prix_neuf):
    """Métriques spécifiques au domaine automobile"""
    
    # Erreur en euros
    prix_occasion_true = y_true * prix_neuf
    prix_occasion_pred = y_pred * prix_neuf
    
    mae_euros = np.mean(np.abs(prix_occasion_true - prix_occasion_pred))
    
    # Pourcentage de prédictions dans une marge acceptable (±5%)
    marge_acceptable = 0.05
    predictions_acceptables = np.mean(
        np.abs(y_true - y_pred) <= marge_acceptable
    )
    
    # Biais (sous-estimation vs sur-estimation)
    biais = np.mean(y_pred - y_true)
    
    return {
        'mae_euros': mae_euros,
        'predictions_acceptables_5pc': predictions_acceptables * 100,
        'biais_taux_decote': biais
    }
```

## Stratégie de validation

### 1. Division temporelle des données

!!! warning "Importance de la validation temporelle"
    Les prix des véhicules évoluent dans le temps (inflation, nouveaux modèles, crises). Une validation temporelle est essentielle.

```python
def temporal_split(df, test_size=0.2):
    """Division temporelle basée sur la date de collecte"""
    
    # Tri par date de collecte
    df_sorted = df.sort_values('date_collecte')
    
    # Point de coupure
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_data = df_sorted.iloc[:split_idx]
    test_data = df_sorted.iloc[split_idx:]
    
    return train_data, test_data
```

### 2. Validation croisée temporelle

```python
from sklearn.model_selection import TimeSeriesSplit

# Configuration de la validation croisée temporelle
tscv = TimeSeriesSplit(n_splits=5)

def temporal_cross_validation(model, X, y):
    """Validation croisée avec respect de l'ordre temporel"""
    
    scores = {
        'mae': [],
        'rmse': [],
        'r2': [],
        'mape': []
    }
    
    for train_idx, val_idx in tscv.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Entraînement
        model.fit(X_train_fold, y_train_fold)
        
        # Prédiction
        y_pred = model.predict(X_val_fold)
        
        # Métriques
        scores['mae'].append(mean_absolute_error(y_val_fold, y_pred))
        scores['rmse'].append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))
        scores['r2'].append(r2_score(y_val_fold, y_pred))
        scores['mape'].append(mean_absolute_percentage_error(y_val_fold, y_pred))
    
    return {metric: (np.mean(values), np.std(values)) 
            for metric, values in scores.items()}
```

## Résultats des modèles

### Comparaison des algorithmes

| Modèle | MAE | RMSE | R² | MAPE (%) | Temps (s) |
|--------|-----|------|----|---------:|----------:|
| **XGBoost** | **0.087** | **0.124** | **0.892** | **12.3** | 45 |
| CatBoost | 0.091 | 0.128 | 0.885 | 13.1 | 38 |
| Random Forest | 0.095 | 0.135 | 0.871 | 14.2 | 22 |
| Linear Regression | 0.142 | 0.198 | 0.712 | 19.8 | 2 |

!!! success "Modèle retenu"
    **XGBoost** offre les meilleures performances avec un R² de 0.892 et une MAE de 0.087.

### Performance par segment

```python
def evaluate_by_segment(model, X_test, y_test, segment_col):
    """Évaluation des performances par segment de véhicule"""
    
    results = {}
    
    for segment in X_test[segment_col].unique():
        mask = X_test[segment_col] == segment
        X_segment = X_test[mask]
        y_segment = y_test[mask]
        
        if len(y_segment) > 10:  # Minimum d'échantillons
            y_pred_segment = model.predict(X_segment)
            
            results[segment] = {
                'n_samples': len(y_segment),
                'mae': mean_absolute_error(y_segment, y_pred_segment),
                'rmse': np.sqrt(mean_squared_error(y_segment, y_pred_segment)),
                'r2': r2_score(y_segment, y_pred_segment)
            }
    
    return pd.DataFrame(results).T
```

### Résultats par segment de véhicule

| Segment | Échantillons | MAE | RMSE | R² |
|---------|-------------:|-----|------|----:|
| **Économique** | 3,245 | 0.082 | 0.118 | 0.901 |
| **Compact** | 5,678 | 0.085 | 0.121 | 0.895 |
| **Familial** | 4,123 | 0.089 | 0.127 | 0.888 |
| **Premium** | 2,156 | 0.095 | 0.138 | 0.875 |

!!! note "Observations"
    - Meilleures performances sur les véhicules économiques et compacts
    - Légère dégradation sur le segment premium (plus de variabilité)

## Analyse des erreurs

### Distribution des erreurs

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_error_distribution(y_true, y_pred):
    """Analyse de la distribution des erreurs"""
    
    errors = y_pred - y_true
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribution des erreurs
    axes[0,0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(0, color='red', linestyle='--', label='Erreur nulle')
    axes[0,0].set_title('Distribution des erreurs')
    axes[0,0].set_xlabel('Erreur (prédit - réel)')
    axes[0,0].legend()
    
    # Q-Q plot pour normalité
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot (normalité des erreurs)')
    
    # Résidus vs prédictions
    axes[1,0].scatter(y_pred, errors, alpha=0.5)
    axes[1,0].axhline(0, color='red', linestyle='--')
    axes[1,0].set_xlabel('Valeurs prédites')
    axes[1,0].set_ylabel('Résidus')
    axes[1,0].set_title('Résidus vs Prédictions')
    
    # Valeurs réelles vs prédites
    axes[1,1].scatter(y_true, y_pred, alpha=0.5)
    axes[1,1].plot([y_true.min(), y_true.max()], 
                   [y_true.min(), y_true.max()], 
                   'r--', linewidth=2)
    axes[1,1].set_xlabel('Valeurs réelles')
    axes[1,1].set_ylabel('Valeurs prédites')
    axes[1,1].set_title('Réel vs Prédit')
    
    plt.tight_layout()
    plt.show()
```

### Analyse des cas extrêmes

```python
def analyze_extreme_errors(X_test, y_test, y_pred, threshold=0.15):
    """Analyse des prédictions avec de grosses erreurs"""
    
    errors = np.abs(y_pred - y_test)
    extreme_mask = errors > threshold
    
    extreme_cases = X_test[extreme_mask].copy()
    extreme_cases['error'] = errors[extreme_mask]
    extreme_cases['y_true'] = y_test[extreme_mask]
    extreme_cases['y_pred'] = y_pred[extreme_mask]
    
    print(f"Cas avec erreur > {threshold}: {extreme_mask.sum()} ({extreme_mask.mean()*100:.1f}%)")
    
    # Analyse par caractéristiques
    print("\nCaractéristiques des cas extrêmes:")
    for col in ['marque', 'segment', 'carburant']:
        if col in extreme_cases.columns:
            print(f"\n{col}:")
            print(extreme_cases[col].value_counts().head())
    
    return extreme_cases
```

## Tests de robustesse

### 1. Stabilité temporelle

```python
def temporal_stability_test(model, X, y, date_col, window_months=6):
    """Test de stabilité des performances dans le temps"""
    
    results = []
    
    # Fenêtres glissantes
    dates = pd.to_datetime(X[date_col])
    start_date = dates.min()
    end_date = dates.max()
    
    current_date = start_date
    while current_date < end_date - pd.DateOffset(months=window_months):
        
        # Fenêtre de test
        test_start = current_date
        test_end = current_date + pd.DateOffset(months=window_months)
        
        test_mask = (dates >= test_start) & (dates < test_end)
        
        if test_mask.sum() > 50:  # Minimum d'échantillons
            X_window = X[test_mask]
            y_window = y[test_mask]
            
            y_pred_window = model.predict(X_window)
            
            results.append({
                'date': test_start,
                'n_samples': len(y_window),
                'mae': mean_absolute_error(y_window, y_pred_window),
                'r2': r2_score(y_window, y_pred_window)
            })
        
        current_date += pd.DateOffset(months=1)
    
    return pd.DataFrame(results)
```

### 2. Test de dérive des données

```python
from scipy.stats import ks_2samp

def data_drift_test(X_train, X_test, threshold=0.05):
    """Détection de dérive dans les distributions des features"""
    
    drift_results = {}
    
    for col in X_train.select_dtypes(include=[np.number]).columns:
        # Test de Kolmogorov-Smirnov
        statistic, p_value = ks_2samp(X_train[col], X_test[col])
        
        drift_results[col] = {
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': p_value < threshold
        }
    
    drift_df = pd.DataFrame(drift_results).T
    
    print(f"Features avec dérive détectée: {drift_df['drift_detected'].sum()}")
    print("\nTop 5 dérives:")
    print(drift_df.sort_values('statistic', ascending=False).head())
    
    return drift_df
```

## Validation métier

### Cohérence économique

```python
def economic_coherence_check(model, test_cases):
    """Vérification de la cohérence économique des prédictions"""
    
    coherence_tests = []
    
    # Test 1: Plus de kilométrage = décote plus forte
    base_case = test_cases.iloc[0].copy()
    
    for km in [20000, 50000, 80000, 120000]:
        case = base_case.copy()
        case['kilometrage'] = km
        prediction = model.predict([case])[0]
        coherence_tests.append({
            'test': 'kilométrage_impact',
            'kilometrage': km,
            'taux_decote': prediction
        })
    
    # Test 2: Plus vieux = décote plus forte
    for age in [1, 3, 5, 7]:
        case = base_case.copy()
        case['age_vehicule'] = age
        case['annee'] = 2024 - age
        prediction = model.predict([case])[0]
        coherence_tests.append({
            'test': 'age_impact',
            'age': age,
            'taux_decote': prediction
        })
    
    return pd.DataFrame(coherence_tests)
```

## Rapport d'évaluation final

### Résumé des performances

!!! success "Performances du modèle XGBoost"
    - **R² = 0.892** : Explique 89.2% de la variance
    - **MAE = 0.087** : Erreur moyenne de 8.7% sur le taux de décote
    - **Erreur moyenne en €** : ~2,300€ sur le prix de vente
    - **Prédictions acceptables (±5%)** : 73% des cas

### Points forts
- ✅ Excellentes performances globales
- ✅ Stabilité temporelle validée
- ✅ Cohérence économique respectée
- ✅ Robustesse aux valeurs aberrantes

### Points d'amélioration
- 🔄 Performance moindre sur le segment premium
- 🔄 Sensibilité aux modèles rares
- 🔄 Besoin de mise à jour régulière des données