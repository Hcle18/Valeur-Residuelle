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


class DataAnalysis:

    def __init__(self, data: pd.DataFrame, target_column: str = None):
        """
        Initialiser la classe DataAnalysis avec le DataFrame de données.
        Args:
            data (pd.DataFrame): Le DataFrame contenant les données prétraitées.
        """
        self.data = data
        self.target_column = target_column
        if target_column is not None and target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data cannot be an empty DataFrame.")
        
    def univariate_analysis(self, list_columns: list, dtype: str):
        """
        Afficher la distribution des données pour une colonne donnée.

        Args:
            list_columns (list): La liste des colonnes à analyser.
            dtype (str): Le type de données (catégoriel ou numérique).
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
            desc_stats = self.data[list_columns].describe().transpose()
            print(desc_stats)

            # Pour les données numériques, on affiche un histogramme avec une courbe de densité
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
            axes = axes.flatten()
            for i, column in enumerate(list_columns):
                sns.histplot(self.data[column], kde=True, ax=axes[i])
                axes[i].set_xlabel(column)
                # Add annotation pour la moyenne, médiane, std, min, max
                mean = self.data[column].mean()
                median = self.data[column].median()
                std = self.data[column].std()
                min_val = self.data[column].min()
                max_val = self.data[column].max()
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
            desc_stats = self.data[list_columns].describe(include='object').transpose()
            print(desc_stats)
    
            # Pour les données catégorielles, on affiche un graphique de fréquence
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
            axes = axes.flatten()
            for i, column in enumerate(list_columns):
                # Calculer les fréquences pour chaque catégorie et leur moyenne de target_column
                freq = self.data.groupby(column).agg(mean_target=(self.target_column, 'mean'), 
                                                     count_freq=(self.target_column, 'count')).sort_values("count_freq", ascending=False)

                # S'il y a plus de 10 catégories, on regroupe les catégories rares
                if len(freq) > 10:
                    # Garder les catégories de fréquence >= 30
                    freq = freq[freq['count_freq'] >= 30]
                    # Sauvegarder le reste des catégories dans "Autres", créer un DataFrame pour "Autres"
                    other_mean = self.data[~self.data[column].isin(freq.index)][self.target_column].mean()
                    other_count = self.data[~self.data[column].isin(freq.index)][self.target_column].count()
                    if other_count > 0:
                        other_row = pd.DataFrame({column: ['Autres'], 'mean_target': [other_mean], 'count_freq': [other_count]})
                        freq = pd.concat([freq, other_row.set_index(column)])

                # Réordonner les catégories par ordre de fréquence
                freq = freq.sort_values("count_freq", ascending=False)
                print(freq)
                
                # Créer un graphique à barres
                sns.barplot(x=freq.index, y=freq['count_freq'], ax=axes[i])
                # Ajouter un linechart pour la moyenne de target_column
                ax2 = axes[i].twinx()
                sns.lineplot(x=freq.index, y=freq['mean_target'], 
                             ax=ax2, color='orange', marker='o', 
                             linewidth=1, label=f'Moyenne de {self.target_column}')
                
                # Ajouter des bar labels en haut des barres
                for index, value in enumerate(freq['count_freq']):
                    axes[i].text(index, value + 0.02 * max(freq['count_freq']), f'{value / freq['count_freq'].sum():.0%}', ha='center', va='bottom')

                axes[i].set_xlabel(column)
                axes[i].set_ylabel("Fréquence")
                ax2.set_ylim(bottom = 0, top = 1.1 * freq['mean_target'].max())

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
    
    def bivariate_analysis(self, list_columns: list, dtype: str):
        if dtype not in ["cat", "num"]:
            raise ValueError("dtype must be either 'cat' for categorical or 'num' for numerical data.")
        if not isinstance(list_columns, list):
            raise ValueError("list_columns must be a list of column names.")

        # Création des subplots
        ncols = 3
        nrows = math.ceil(len(list_columns)/ncols)

        if dtype == "num":
            # Pour les données numériques, on affiche un nuage de points
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
            axes = axes.flatten()
            for i, column in enumerate(list_columns):
                sns.scatterplot(data=self.data, x=column, y=self.target_column, ax=axes[i])
                axes[i].set_xlabel(column)
                axes[i].set_ylabel(self.target_column)
                # Ajouter une ligne de régression
                sns.regplot(data=self.data, x=column, y=self.target_column, ax=axes[i], scatter=False, color='red')
                axes[i].annotate(f'Corrélation: {self.data[column].corr(self.data[self.target_column]):.2f}',
                             xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                             bbox=dict(boxstyle="round", fc="w", ec="black", alpha=0.5),
                             ha='left', va='top')
                axes[i].set_title(f"{column} vs {self.target_column}")

            plt.tight_layout()
            plt.suptitle(f"Analyse bivariée avec {self.target_column}", fontsize=20)
            plt.subplots_adjust(top=0.95)
            # Supprimer les axes qui dépasse le nombre de variables
            for j in range(len(list_columns), len(axes)):
                fig.delaxes(axes[j])
            plt.show()

        elif dtype == "cat":
            # Tableau synthétique des statistiques de target_column pour chaque modalité de list_columns
            for column in list_columns:
                desc_stats = self.data.groupby(column)[self.target_column].describe()
                # Réordonner les statistiques par la moyenne de la target_column
                desc_stats = desc_stats.sort_values(by='mean', ascending=False)
                print(f"Statistiques de Target value pour {column} :")
                # Afficher les statistiques descriptives
                print(desc_stats)

            # Pour les données catégorielles, on affiche un boxplot
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
            axes = axes.flatten()
            for i, column in enumerate(list_columns):
                sns.boxplot(data=self.data, x=column, y=self.target_column, ax=axes[i])
                # Rotation des étiquettes de l'axe x pour une meilleure lisibilité
                axes[i].tick_params(axis='x', rotation=45)
                # Réduire la taille des étiquettes de l'axe x
                for label in axes[i].get_xticklabels():
                    label.set_fontsize(7)
            plt.tight_layout()
            plt.suptitle(f"Analyse bivariée avec {self.target_column}", fontsize=20)
            plt.subplots_adjust(top=0.95)
            # Supprimer les axes qui dépasse len(list_columns)
            for j in range(len(list_columns), len(axes)):
                fig.delaxes(axes[j])
            plt.show()
    def multivariate_analysis(self, list_columns: list):
        """
        
        """

if __name__ == "__main__":
    # Charger les données prétraitées
    # Obtenir le chemin du répertoire actuel et le chemin du répertoire racine du projet
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    # Le chemin vers les données
    data_path = os.path.join(project_root, "data", "processed_data", "preprocessed_data.csv")
    data = pd.read_csv(data_path)

    # Créer une instance de la classe DataAnalysis
    eda = DataAnalysis(data, target_column="ratio_vr")

    eda.data['nb_porte'] = eda.data['nb_porte'].astype(str)

    # Effectuer une analyse univariée
    cat_vars = data.select_dtypes(include=['object']).columns.tolist()
    num_vars = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    eda.univariate_analysis(list_columns=num_vars, dtype="num")
    #eda.univariate_analysis(list_columns=cat_vars, dtype="cat")
    #eda.bivariate_analysis(list_columns=num_vars, dtype="num")
    #eda.bivariate_analysis(list_columns=cat_vars, dtype="cat")