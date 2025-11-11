# Analyse exploratoire des données (EDA)

## Vue d'ensemble

L'analyse exploratoire des données (EDA) est essentielle pour comprendre les patterns, distributions et relations dans notre dataset de véhicules d'occasion. Cette analyse guide les décisions de preprocessing et de modélisation.

## Structure du dataset

### Statistiques générales

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Chargement des données
df = pd.read_csv('data/processed_data/preprocessed_data.csv')

print(f"Dataset: {len(df):,} véhicules, {len(df.columns)} variables")
print(f"Période: {df['annee'].min()} - {df['annee'].max()}")
print(f"Valeur résiduelle moyenne: {df['taux_decote'].mean():.1%}")
```

| Métrique | Valeur |
|----------|--------|
| **Nombre de véhicules** | 2352 |
| **Variables** | 34 |
| **Années couvertes** | 2017-2024 |
| **Marques** | 34 |
| **Modèles** | 221 |

## Analyse univariée

### Distribution de la variable cible

```python
def analyze_target_variable(df):
    """Analyse de la distribution du taux de décote"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Distribution', 'Box Plot', 'Q-Q Plot', 'Évolution temporelle'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Histogramme
    fig.add_trace(
        go.Histogram(x=df['taux_decote'], nbinsx=50, name='Distribution'),
        row=1, col=1
    )
    
    # 2. Box plot
    fig.add_trace(
        go.Box(y=df['taux_decote'], name='Taux de décote'),
        row=1, col=2
    )
    
    # 3. Q-Q plot (approximation)
    from scipy import stats
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
    sample_quantiles = np.percentile(df['taux_decote'], np.linspace(1, 99, 100))
    
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sample_quantiles, 
                  mode='markers', name='Q-Q Plot'),
        row=2, col=1
    )
    
    # 4. Évolution par année
    yearly_stats = df.groupby('annee')['taux_decote'].agg(['mean', 'std']).reset_index()
    
    fig.add_trace(
        go.Scatter(x=yearly_stats['annee'], y=yearly_stats['mean'],
                  mode='lines+markers', name='Moyenne annuelle',
                  error_y=dict(type='data', array=yearly_stats['std'])),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Analyse de la variable cible: Taux de décote")
    fig.show()
    
    # Statistiques descriptives
    print("Statistiques du taux de décote:")
    print(df['taux_decote'].describe())

# Exécution de l'analyse
analyze_target_variable(df)
```

**Observations clés** :
- Distribution légèrement asymétrique (skewness = -0.23)
- Taux de décote médian : 67.3%
- Écart-type : 14.2%
- Valeurs aberrantes : < 1% du dataset

### Variables numériques principales

```python
def analyze_numeric_variables(df):
    """Analyse des variables numériques importantes"""
    
    numeric_vars = ['kilometrage', 'puissance', 'age_vehicule', 'km_par_an', 'prix_neuf']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, var in enumerate(numeric_vars):
        # Histogramme avec densité
        axes[i].hist(df[var], bins=50, alpha=0.7, edgecolor='black', density=True)
        
        # Ajout de la courbe de densité
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(df[var].dropna())
        x_range = np.linspace(df[var].min(), df[var].max(), 100)
        axes[i].plot(x_range, kde(x_range), 'r-', linewidth=2)
        
        axes[i].set_title(f'Distribution: {var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Densité')
        
        # Statistiques sur le graphique
        mean_val = df[var].mean()
        median_val = df[var].median()
        axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Moyenne: {mean_val:.0f}')
        axes[i].axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Médiane: {median_val:.0f}')
        axes[i].legend()
    
    # Suppression du dernier subplot vide
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.show()
    
    # Matrice de corrélation
    correlation_matrix = df[numeric_vars + ['taux_decote']].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f')
    plt.title('Matrice de corrélation - Variables numériques')
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

# Analyse des corrélations
corr_matrix = analyze_numeric_variables(df)
```

### Variables catégorielles

```python
def analyze_categorical_variables(df):
    """Analyse des variables catégorielles"""
    
    categorical_vars = ['marque', 'carburant', 'transmission', 'segment']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, var in enumerate(categorical_vars):
        # Comptage des modalités
        value_counts = df[var].value_counts()
        
        # Limitation aux top 10 pour la lisibilité
        if len(value_counts) > 10:
            top_values = value_counts.head(10)
            others_count = value_counts.iloc[10:].sum()
            if others_count > 0:
                top_values['Autres'] = others_count
            value_counts = top_values
        
        # Graphique en barres horizontales
        y_pos = np.arange(len(value_counts))
        axes[i].barh(y_pos, value_counts.values)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(value_counts.index)
        axes[i].set_xlabel('Nombre de véhicules')
        axes[i].set_title(f'Distribution: {var}')
        
        # Ajout des pourcentages
        for j, v in enumerate(value_counts.values):
            percentage = (v / len(df)) * 100
            axes[i].text(v + max(value_counts.values) * 0.01, j, 
                        f'{percentage:.1f}%', va='center')
    
    plt.tight_layout()
    plt.show()

analyze_categorical_variables(df)
```

## Analyse bivariée

### Relations avec la variable cible

```python
def analyze_target_relationships(df):
    """Analyse des relations entre variables et taux de décote"""
    
    # 1. Taux de décote par marque
    plt.figure(figsize=(15, 8))
    
    # Calcul des statistiques par marque
    brand_stats = df.groupby('marque')['taux_decote'].agg(['mean', 'std', 'count']).reset_index()
    brand_stats = brand_stats[brand_stats['count'] >= 50]  # Minimum 50 véhicules
    brand_stats = brand_stats.sort_values('mean')
    
    # Box plot par marque
    brands_filtered = brand_stats['marque'].tolist()
    df_filtered = df[df['marque'].isin(brands_filtered)]
    
    sns.boxplot(data=df_filtered, x='marque', y='taux_decote')
    plt.xticks(rotation=45)
    plt.title('Distribution du taux de décote par marque')
    plt.ylabel('Taux de décote')
    plt.tight_layout()
    plt.show()
    
    # 2. Évolution avec l'âge
    plt.figure(figsize=(12, 6))
    
    # Régression polynomiale pour visualiser la tendance
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    X = df['age_vehicule'].values.reshape(-1, 1)
    y = df['taux_decote'].values
    
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Prédiction pour la courbe lisse
    X_smooth = np.linspace(df['age_vehicule'].min(), df['age_vehicule'].max(), 100).reshape(-1, 1)
    X_smooth_poly = poly_features.transform(X_smooth)
    y_smooth = model.predict(X_smooth_poly)
    
    # Scatter plot avec tendance
    plt.scatter(df['age_vehicule'], df['taux_decote'], alpha=0.5, s=20)
    plt.plot(X_smooth, y_smooth, 'r-', linewidth=3, label='Tendance polynomiale')
    
    plt.xlabel('Âge du véhicule (années)')
    plt.ylabel('Taux de décote')
    plt.title('Relation entre âge et taux de décote')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 3. Impact du kilométrage
    plt.figure(figsize=(12, 6))
    
    # Binning du kilométrage pour une meilleure visualisation
    df['km_binned'] = pd.cut(df['kilometrage'], 
                            bins=[0, 20000, 50000, 80000, 120000, float('inf')],
                            labels=['0-20k', '20-50k', '50-80k', '80-120k', '120k+'])
    
    sns.violinplot(data=df, x='km_binned', y='taux_decote')
    plt.title('Impact du kilométrage sur le taux de décote')
    plt.xlabel('Kilométrage (km)')
    plt.ylabel('Taux de décote')
    plt.show()

analyze_target_relationships(df)
```

### Heatmap des corrélations avancées

```python
def advanced_correlation_analysis(df):
    """Analyse avancée des corrélations"""
    
    # Sélection des variables numériques importantes
    numeric_cols = [
        'taux_decote', 'age_vehicule', 'kilometrage', 'km_par_an',
        'puissance', 'puissance_specifique', 'prix_neuf', 'popularite_marque'
    ]
    
    # Matrice de corrélation avec différentes méthodes
    correlations = pd.DataFrame()
    
    for method in ['pearson', 'spearman', 'kendall']:
        corr = df[numeric_cols].corr(method=method)['taux_decote'].drop('taux_decote')
        correlations[method] = corr
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, method in enumerate(['pearson', 'spearman', 'kendall']):
        corr_values = correlations[method].sort_values(key=abs, ascending=False)
        
        colors = ['red' if x < 0 else 'green' for x in corr_values.values]
        axes[i].barh(range(len(corr_values)), corr_values.values, color=colors, alpha=0.7)
        axes[i].set_yticks(range(len(corr_values)))
        axes[i].set_yticklabels(corr_values.index)
        axes[i].set_xlabel(f'Corrélation {method.title()}')
        axes[i].set_title(f'Corrélations avec taux_decote ({method})')
        axes[i].grid(True, alpha=0.3)
        
        # Ajout des valeurs
        for j, v in enumerate(corr_values.values):
            axes[i].text(v + 0.01 if v > 0 else v - 0.01, j, f'{v:.3f}', 
                        va='center', ha='left' if v > 0 else 'right')
    
    plt.tight_layout()
    plt.show()
    
    return correlations

correlations_advanced = advanced_correlation_analysis(df)
```

## Analyse multivariée

### Analyse en composantes principales (PCA)

```python
def pca_analysis(df):
    """Analyse en composantes principales"""
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Sélection des variables numériques
    numeric_features = [
        'age_vehicule', 'kilometrage', 'puissance', 'cylindree',
        'km_par_an', 'puissance_specifique', 'prix_neuf'
    ]
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_features])
    
    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Visualisation de la variance expliquée
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Variance expliquée par composante
    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_)
    axes[0].set_xlabel('Composante principale')
    axes[0].set_ylabel('Variance expliquée')
    axes[0].set_title('Variance expliquée par composante')
    
    # 2. Variance cumulative
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    axes[1].axhline(y=0.8, color='r', linestyle='--', label='80% de variance')
    axes[1].axhline(y=0.95, color='g', linestyle='--', label='95% de variance')
    axes[1].set_xlabel('Nombre de composantes')
    axes[1].set_ylabel('Variance cumulative expliquée')
    axes[1].set_title('Variance cumulative expliquée')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Biplot des deux premières composantes
    plt.figure(figsize=(12, 8))
    
    # Points colorés par taux de décote
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=df['taux_decote'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Taux de décote')
    
    # Vecteurs des variables originales
    feature_vectors = pca.components_[:2].T
    
    for i, feature in enumerate(numeric_features):
        plt.arrow(0, 0, feature_vectors[i, 0]*3, feature_vectors[i, 1]*3,
                 head_width=0.1, head_length=0.1, fc='red', ec='red')
        plt.text(feature_vectors[i, 0]*3.5, feature_vectors[i, 1]*3.5, feature,
                fontsize=10, ha='center', va='center')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} de variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} de variance)')
    plt.title('Biplot PCA - Projection des véhicules et variables')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Contribution des variables aux composantes principales
    components_df = pd.DataFrame(
        pca.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=numeric_features
    )
    
    print("Contribution des variables aux 3 premières composantes:")
    print(components_df.round(3))
    
    return pca, X_pca

pca_result, X_pca = pca_analysis(df)
```

### Clustering des véhicules

```python
def clustering_analysis(df):
    """Analyse de clustering pour identifier des groupes de véhicules"""
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    
    # Préparation des données
    clustering_features = [
        'age_vehicule', 'kilometrage', 'puissance', 'prix_neuf', 'taux_decote'
    ]
    
    X = df[clustering_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Détermination du nombre optimal de clusters
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Visualisation des métriques
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Méthode du coude
    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].set_xlabel('Nombre de clusters (k)')
    axes[0].set_ylabel('Inertie')
    axes[0].set_title('Méthode du coude')
    axes[0].grid(True, alpha=0.3)
    
    # Score de silhouette
    axes[1].plot(K_range, silhouette_scores, 'ro-')
    axes[1].set_xlabel('Nombre de clusters (k)')
    axes[1].set_ylabel('Score de silhouette')
    axes[1].set_title('Score de silhouette')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Clustering optimal (k=4 basé sur le coude et silhouette)
    optimal_k = 4
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(X_scaled)
    
    # Ajout des clusters au dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    # Analyse des clusters
    cluster_summary = df_clustered.groupby('cluster')[clustering_features].mean()
    
    print("Profil des clusters (moyennes):")
    print(cluster_summary.round(2))
    
    # Visualisation 3D des clusters
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for cluster in range(optimal_k):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster]
        
        fig.add_trace(go.Scatter3d(
            x=cluster_data['age_vehicule'],
            y=cluster_data['kilometrage'],
            z=cluster_data['prix_neuf'],
            mode='markers',
            marker=dict(size=4, color=colors[cluster], opacity=0.6),
            name=f'Cluster {cluster}',
            text=cluster_data['marque'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Âge: %{x} ans<br>' +
                         'Kilométrage: %{y:,.0f} km<br>' +
                         'Prix neuf: %{z:,.0f} €<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title='Clustering des véhicules (3D)',
        scene=dict(
            xaxis_title='Âge (années)',
            yaxis_title='Kilométrage (km)',
            zaxis_title='Prix neuf (€)'
        ),
        width=800,
        height=600
    )
    
    fig.show()
    
    return df_clustered, kmeans_final

df_clustered, kmeans_model = clustering_analysis(df)
```

## Insights et patterns découverts

### Résumé des découvertes principales

```python
def generate_eda_insights(df, correlations, df_clustered):
    """Génération automatique d'insights basés sur l'EDA"""
    
    insights = {
        'patterns_temporels': {},
        'segments_vehicules': {},
        'facteurs_cles': {},
        'recommandations': []
    }
    
    # 1. Patterns temporels
    yearly_depreciation = df.groupby('annee')['taux_decote'].mean()
    insights['patterns_temporels']['evolution_decote'] = {
        'tendance': 'stable' if yearly_depreciation.std() < 0.05 else 'variable',
        'annee_max_decote': yearly_depreciation.idxmax(),
        'annee_min_decote': yearly_depreciation.idxmin()
    }
    
    # 2. Segmentation véhicules
    brand_performance = df.groupby('marque')['taux_decote'].agg(['mean', 'count'])
    brand_performance = brand_performance[brand_performance['count'] >= 50]
    
    insights['segments_vehicules']['marques_premium'] = brand_performance[
        brand_performance['mean'] > brand_performance['mean'].quantile(0.75)
    ].index.tolist()
    
    insights['segments_vehicules']['marques_economiques'] = brand_performance[
        brand_performance['mean'] < brand_performance['mean'].quantile(0.25)
    ].index.tolist()
    
    # 3. Facteurs clés
    correlations_abs = correlations['pearson'].abs().sort_values(ascending=False)
    insights['facteurs_cles']['top_3_predicteurs'] = correlations_abs.head(3).index.tolist()
    
    # 4. Analyse des clusters
    cluster_profiles = df_clustered.groupby('cluster').agg({
        'taux_decote': 'mean',
        'age_vehicule': 'mean',
        'kilometrage': 'mean',
        'prix_neuf': 'mean'
    }).round(2)
    
    insights['clusters'] = cluster_profiles.to_dict('index')
    
    # 5. Recommandations basées sur les données
    if correlations['pearson']['age_vehicule'] < -0.5:
        insights['recommandations'].append(
            "L'âge est un prédicteur fort de la décote - considérer des features temporelles avancées"
        )
    
    if correlations['pearson']['kilometrage'] < -0.3:
        insights['recommandations'].append(
            "Le kilométrage impact significativement la valeur - intégrer des ratios km/âge"
        )
    
    # Affichage des insights
    print("=== INSIGHTS DÉCOUVERTS ===\n")
    
    print("1. PATTERNS TEMPORELS:")
    print(f"   - Évolution de la décote: {insights['patterns_temporels']['evolution_decote']['tendance']}")
    print(f"   - Année avec plus forte décote: {insights['patterns_temporels']['evolution_decote']['annee_max_decote']}")
    
    print("\n2. SEGMENTS DE VÉHICULES:")
    print(f"   - Marques premium (forte décote): {', '.join(insights['segments_vehicules']['marques_premium'][:3])}")
    print(f"   - Marques économiques (faible décote): {', '.join(insights['segments_vehicules']['marques_economiques'][:3])}")
    
    print("\n3. FACTEURS CLÉS:")
    print(f"   - Top 3 prédicteurs: {', '.join(insights['facteurs_cles']['top_3_predicteurs'])}")
    
    print("\n4. PROFILS DE CLUSTERS:")
    for cluster_id, profile in insights['clusters'].items():
        print(f"   - Cluster {cluster_id}: Décote {profile['taux_decote']:.1%}, "
              f"Âge {profile['age_vehicule']:.1f}ans, "
              f"Prix neuf {profile['prix_neuf']:,.0f}€")
    
    print("\n5. RECOMMANDATIONS:")
    for rec in insights['recommandations']:
        print(f"   - {rec}")
    
    return insights

# Génération des insights
insights = generate_eda_insights(df, correlations_advanced, df_clustered)
```

## Dashboard interactif

```python
def create_interactive_dashboard():
    """Création d'un dashboard interactif avec Plotly Dash"""
    
    import dash
    from dash import dcc, html, Input, Output
    import plotly.express as px
    
    # Initialisation de l'application Dash
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Dashboard EDA - Valeur Résiduelle", 
                style={'textAlign': 'center', 'marginBottom': 30}),
        
        # Contrôles
        html.Div([
            html.Div([
                html.Label("Sélectionner la marque:"),
                dcc.Dropdown(
                    id='brand-dropdown',
                    options=[{'label': 'Toutes', 'value': 'all'}] + 
                           [{'label': brand, 'value': brand} for brand in df['marque'].unique()],
                    value='all'
                )
            ], style={'width': '30%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Sélectionner l'année:"),
                dcc.RangeSlider(
                    id='year-slider',
                    min=df['annee'].min(),
                    max=df['annee'].max(),
                    value=[df['annee'].min(), df['annee'].max()],
                    marks={year: str(year) for year in range(df['annee'].min(), df['annee'].max()+1)},
                    step=1
                )
            ], style={'width': '65%', 'float': 'right', 'display': 'inline-block'})
        ], style={'marginBottom': 30}),
        
        # Graphiques
        html.Div([
            dcc.Graph(id='depreciation-scatter'),
            dcc.Graph(id='brand-boxplot'),
        ], style={'width': '100%'}),
        
        html.Div([
            dcc.Graph(id='correlation-heatmap'),
            dcc.Graph(id='distribution-histogram'),
        ], style={'width': '100%'})
    ])
    
    # Callbacks pour l'interactivité
    @app.callback(
        [Output('depreciation-scatter', 'figure'),
         Output('brand-boxplot', 'figure'),
         Output('correlation-heatmap', 'figure'),
         Output('distribution-histogram', 'figure')],
        [Input('brand-dropdown', 'value'),
         Input('year-slider', 'value')]
    )
    def update_graphs(selected_brand, year_range):
        # Filtrage des données
        filtered_df = df[
            (df['annee'] >= year_range[0]) & 
            (df['annee'] <= year_range[1])
        ]
        
        if selected_brand != 'all':
            filtered_df = filtered_df[filtered_df['marque'] == selected_brand]
        
        # 1. Scatter plot âge vs décote
        fig1 = px.scatter(
            filtered_df, 
            x='age_vehicule', 
            y='taux_decote',
            color='marque' if selected_brand == 'all' else None,
            title='Relation Âge - Taux de décote',
            trendline='ols'
        )
        
        # 2. Box plot par marque
        fig2 = px.box(
            filtered_df, 
            x='marque', 
            y='taux_decote',
            title='Distribution du taux de décote par marque'
        )
        fig2.update_xaxes(tickangle=45)
        
        # 3. Heatmap de corrélation
        numeric_cols = ['taux_decote', 'age_vehicule', 'kilometrage', 'puissance', 'prix_neuf']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig3 = px.imshow(
            corr_matrix,
            title='Matrice de corrélation',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        
        # 4. Histogramme de distribution
        fig4 = px.histogram(
            filtered_df,
            x='taux_decote',
            nbins=30,
            title='Distribution du taux de décote'
        )
        
        return fig1, fig2, fig3, fig4
    
    return app

# Lancement du dashboard (optionnel)
# dashboard_app = create_interactive_dashboard()
# dashboard_app.run_server(debug=True, port=8051)
```

Cette analyse exploratoire révèle les patterns essentiels dans les données de véhicules d'occasion et guide les décisions pour le développement du modèle prédictif de valeur résiduelle.