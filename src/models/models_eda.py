"""
Module for Exploratory Data Analysis (EDA) on the preprocessed dataset.

This module provides functions to visualize and analyze the preprocessed data,
helping to gain insights and understand the data distribution and relationships.

"""

# Importing necessary libraries
import os
from tkinter import font
from turtle import left
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import line
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
sns.set_style("dark")


def univariate_analysis(df:pd.DataFrame, list_columns: list[str], dtype: str, target_column: str = None):
    """
    Afficher la distribution des données pour une colonne donnée.

    Args:
        list_columns (list): La liste des colonnes à analyser.
        dtype (str): Le type de données (catégoriel ou numérique).
        target_column (str, optional): Le nom de la colonne cible à analyser. 
                                        Si None, on n'affiche pas la moyenne de la colonne cible.
    """

    if dtype not in ["cat", "num"]:
        raise ValueError("dtype must be either 'cat' for categorical or 'num' for numerical data.")
    if not isinstance(list_columns, list):
        raise ValueError("list_columns must be a list of column names.")

    # Création des subplots
    ncols = 3
    nrows = math.ceil(len(list_columns)/ncols)

    if dtype == "num":
        # Tableau synthétique des statistiques descriptives univariées
        desc_stats = df[list_columns].describe().transpose()
        print(desc_stats)

        # Pour les données numériques, on affiche un histogramme avec une courbe de densité
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
        axes = axes.flatten()
        for i, column in enumerate(list_columns):
            sns.histplot(df[column], kde=True, ax=axes[i])
            axes[i].set_xlabel(column)
            # Add annotation pour la moyenne, médiane, std, min, max
            mean = df[column].mean()
            median = df[column].median()
            std = df[column].std()
            min_val = df[column].min()
            max_val = df[column].max()
            axes[i].annotate(f'Mean: {mean:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}',
                                xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                                bbox=dict(boxstyle="round", fc="w", ec="black", alpha=0.5),
                                ha='left', va='top')

        plt.tight_layout()
        plt.suptitle("Analyse de distribution univariée", fontsize=20)

        plt.subplots_adjust(top=0.95)
        # Supprimer les axes qui dépasse len(list_columns)
        for j in range(len(list_columns), len(axes)):
            fig.delaxes(axes[j])
        plt.show()

    elif dtype == "cat":
        # Tableau synthétique des statistiques descriptives univariées pour les données catégorielles
        desc_stats = df[list_columns].describe(include='object').transpose()
        print(desc_stats)

        # Pour les données catégorielles, on affiche un graphique de fréquence
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
        axes = axes.flatten()
        for i, column in enumerate(list_columns):
            # Vérifier si la colonne est dans le DataFrame et si elle est catégorielle
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            if df[column].dtype not in ['object', 'category']:
                raise ValueError(f"Column '{column}' is not categorical. Please provide a categorical column.")
            
            # Calculer les fréquences pour chaque catégorie et leur moyenne de target_column si elle est fournie
            if target_column is None:
                freq = df[column].value_counts().sort_values(ascending=False)
                # Index = les catégories, valeurs = les fréquences
                freq = pd.DataFrame(freq)
                freq.columns = ['count_freq']
                # Renommer l'index en nom de la colonne
                freq.index.name = column
                # S'il y a plus de 10 catégories, on regroupe les catégories rares
                if len(freq) > 10:
                    # Garder les catégories de fréquence >= 30
                    freq = freq[freq['count_freq'] >= 30]
                    # Sauvegarder le reste des catégories dans "Autres", créer un DataFrame pour "Autres"
                    other_count = df[~df[column].isin(freq.index)].shape[0]
                    if other_count > 0:
                        other_row = pd.DataFrame({freq.index.name: ['Autres'], 'count_freq': [other_count]})
                        other_row.set_index(freq.index.name, inplace=True)
                        print(other_row)
                        freq = pd.concat([freq, other_row], ignore_index=False)
                # Réordonner les catégories par ordre de fréquence
                freq = freq.sort_values("count_freq", ascending=False)
            else:
                if target_column not in df.columns:
                    raise ValueError(f"target_column '{target_column}' not found in DataFrame.")
                if df[target_column].dtype not in ['int64', 'float64']:
                    raise ValueError(f"target_column '{target_column}' is not numeric. Please provide a numeric column.")
                # Calculer la moyenne de target_column pour chaque catégorie
                # et le nombre de fréquences
                freq = df.groupby(column).agg(mean_target=(target_column, 'mean'), 
                                              count_freq=(target_column, 'count'))

                # S'il y a plus de 10 catégories, on regroupe les catégories rares
                if len(freq) > 10:
                    # Garder les catégories de fréquence >= 30
                    freq = freq[freq['count_freq'] >= 30]
                    # Sauvegarder le reste des catégories dans "Autres", créer un DataFrame pour "Autres"
                    other_mean = df[~df[column].isin(freq.index)][target_column].mean()
                    other_count = df[~df[column].isin(freq.index)][target_column].count()
                    if other_count > 0:
                        other_row = pd.DataFrame({column: ['Autres'], 
                                                'mean_target': [other_mean], 
                                                'count_freq': [other_count]
                                                }
                                                )   
                        freq = pd.concat([freq, other_row.set_index(column)])
                # Réordonner les catégories par ordre de fréquence
                freq = freq.sort_values("count_freq", ascending=False)
            print(freq)
            
            # Créer un graphique à barres
            sns.barplot(x=freq.index, y=freq['count_freq'], ax=axes[i])
            # Ajouter un linechart pour la moyenne de target_column si le target_column est fourni
            if target_column is not None:
                ax2 = axes[i].twinx()
                sns.lineplot(x=freq.index, y=freq['mean_target'], 
                                ax=ax2, color='orange', marker='o', 
                                linewidth=1, label=f'Moyenne de {target_column}')
                ax2.set_ylim(bottom = 0, top = 1.1 * freq['mean_target'].max())
            
            # Ajouter des bar labels en haut des barres
            for index, value in enumerate(freq['count_freq']):
                axes[i].text(index, value + 0.02 * max(freq['count_freq']), f"{value / freq['count_freq'].sum():.0%}", ha='center', va='bottom')

            axes[i].set_xlabel(column)
            axes[i].set_ylabel("Fréquence")

            # Rotation des étiquettes de l'axe x si le nom de la catégorie est long
            if (freq.index.dtype == 'object' and any(len(str(cat)) > 8 for cat in freq.index)) or len(freq.index) > 10:
                # Rotation des étiquettes de l'axe x pour une meilleure lisibilité
                axes[i].tick_params(axis='x', rotation=45)

            if len(freq) > 10:
                # Réduire la taille des étiquettes de l'axe x s'il y a plus de 10 catégories
                for label in axes[i].get_xticklabels():
                    label.set_fontsize(7)

        # Ajuster la mise en page
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        plt.tight_layout()
        plt.suptitle("Analyse de distribution univariée", fontsize=20)
        plt.subplots_adjust(top=0.95)
        # Supprimer les axes qui dépasse len(list_columns)
        for j in range(len(list_columns), len(axes)):
            fig.delaxes(axes[j])
        plt.show()
    return None
    
def bivariate_analysis(df:pd.DataFrame, list_columns: list[str], dtype: str, target_column: str):
    """
    Analyse bivariée des données pour visualiser la relation entre les colonnes et la colonne cible.
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        list_columns (list): La liste des colonnes à analyser.
        dtype (str): Le type de données ('cat' pour catégoriel, 'num' pour numérique).
        target_column (str): Le nom de la colonne cible à analyser.
    """
    # Data Validation
    if dtype not in ["cat", "num"]:
        raise ValueError("dtype must be either 'cat' for categorical or 'num' for numerical data.")
    if not isinstance(list_columns, list):
        raise ValueError("list_columns must be a list of column names.")
    if target_column not in df.columns:
        raise ValueError("target_column must be a column name in the DataFrame.")
    if df[target_column].dtype not in ['int64', 'float64']:
        raise ValueError("target_column must be a numeric column for bivariate analysis with numerical data.")

    # Création des subplots
    ncols = 3
    nrows = math.ceil(len(list_columns)/ncols)

    if dtype == "num":
        # Pour les données numériques, on affiche un nuage de points
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
        axes = axes.flatten()
        for i, column in enumerate(list_columns):
            sns.scatterplot(data=df, x=column, y=target_column, ax=axes[i])
            axes[i].set_xlabel(column)
            axes[i].set_ylabel(target_column)
            # Ajouter une ligne de régression
            sns.regplot(data=df, x=column, y=target_column, ax=axes[i], scatter=False, color='red')
            axes[i].annotate(f'Corrélation Spearman: {df[column].corr(df[target_column], method="spearman"):.2f}',
                            xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                            bbox=dict(boxstyle="round", fc="w", ec="black", alpha=0.5),
                            ha='left', va='top')
            axes[i].set_title(f"{column} vs {target_column}")

        plt.tight_layout()
        plt.suptitle(f"Analyse bivariée avec {target_column}", fontsize=20)
        plt.subplots_adjust(top=0.95)
        # Supprimer les axes qui dépasse le nombre de variables
        for j in range(len(list_columns), len(axes)):
            fig.delaxes(axes[j])
        plt.show()

    elif dtype == "cat":
        # Tableau synthétique des statistiques de target_column pour chaque modalité de list_columns
        for column in list_columns:
            desc_stats = df.groupby(column)[target_column].describe()
            # Réordonner les statistiques par la moyenne de la target_column
            desc_stats = desc_stats.sort_values(by='mean', ascending=False)
            print(f"Statistiques de Target value pour {column} :")
            # Afficher les statistiques descriptives
            print(desc_stats)

        # Pour les données catégorielles, on affiche un boxplot
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
        axes = axes.flatten()
        for i, column in enumerate(list_columns):
            sns.boxplot(data=df, x=column, y=target_column, ax=axes[i])
            # Rotation des étiquettes de l'axe x pour une meilleure lisibilité
            axes[i].tick_params(axis='x', rotation=45)
            # Réduire la taille des étiquettes de l'axe x
            for label in axes[i].get_xticklabels():
                label.set_fontsize(7)
        plt.tight_layout()
        plt.suptitle(f"Analyse bivariée avec {target_column}", fontsize=20)
        plt.subplots_adjust(top=0.95)
        # Supprimer les axes qui dépasse len(list_columns)
        for j in range(len(list_columns), len(axes)):
            fig.delaxes(axes[j])
        plt.show()

def multivariate_analysis(df: pd.DataFrame, x: str, y: str, hue: str = None, col: str = None, row: str = None):
    """
    Création de graphiques multivariés pour visualiser les relations entre plusieurs variables.
    """
    sns.lmplot(data=df, x=x, y=y, hue=hue, col=col, row=row, height = 4, fit_reg=False, aspect=1.5)
    plt.title(f"Analyse multivariée de {x} et {y}", fontsize=12)
    plt.show()
    return None

if __name__ == "__main__":
    # Charger les données prétraitées
    # Obtenir le chemin du répertoire actuel et le chemin du répertoire racine du projet
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    # Le chemin vers les données
    data_path = os.path.join(project_root, "data", "processed_data", "preprocessed_data.csv")
    data = pd.read_csv(data_path)


    # Effectuer une analyse univariée
    cat_vars = data.select_dtypes(include=['object']).columns.tolist()
    num_vars = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    #univariate_analysis(data, list_columns=cat_vars, dtype="cat", target_column='ratio_vr')
    #univariate_analysis(data, list_columns=num_vars, dtype="num")
    bivariate_analysis(data, list_columns=num_vars, dtype="num", target_column='ratio_vr')
    #bivariate_analysis(data, list_columns=cat_vars, dtype="cat", target_column='ratio_vr')