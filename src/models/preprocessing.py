"""
Module pour le prétraitement des données d'annonces d'automobiles scrapées.
Ce module permet de charger, afficher, prétransformer les données d'annonces d'automobiles scrapées.
"""

# Importing necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Définir la classe DataPreprocessor pour le prétraitement des données d'annonces automobiles scrapées
# Cette classe permet de charger, afficher, prétransformer et visualiser les données

class DataPreprocessor:
    # Dictionnaire des noms de marque pour que ce soit homogène entre les différentes sources de données (scraping et prix neuf)
    # Il est important de garder les noms de marque homogènes pour éviter les doublons et faciliter l'analyse
    dict_name_marque = {'Abarth' : 'Abarth', 'Alfa' : 'Alfa Romeo', 'Audi' : 'Audi', 
                    'BMW' : 'BMW', 
                    'Citroen' : 'Citroen', 'Cupra' : 'Cupra', 
                    'DS' : 'DS', 'Dacia' : 'Dacia',
                    'Fiat' : 'Fiat', 'Ford' : 'Ford', 
                    'Honda' : 'Honda', 'Hyundai' : 'Hyundai', 
                    'Infiniti' : 'Infiniti', 
                    'Jaguar' : 'Jaguar', 'Jeep' : 'Jeep',
                    'Kia' : 'Kia', 
                    'Land' : 'Land Rover', 'Lexus' : 'Lexus', 
                    'MG' : "MG", 'MINI' : 'MINI', 'Mazda' : 'Mazda', 'Mercedes-Benz' : 'Mercedes', 'Mitsubishi' : 'Mitsubishi', 
                    'Nissan' : 'Nissan', 
                    'Opel' : 'Opel', 
                    'Peugeot' : 'Peugeot', 
                    'Renault' : 'Renault', 
                    'Seat' : 'Seat', 'Skoda' : 'Skoda', 'Smart' : 'Smart', 'Suzuki' : 'Suzuki', 
                    'Toyota' : 'Toyota', 
                    'Volkswagen' : 'Volkswagen', 'Volvo' : 'Volvo'
                    }
    # Homogénéiser le type de boite de vitesse entre les différentes sources de données
    dict_boite = {'Double embrayage / DCT' : 'Boite de vitesse automatique',
                  'Semi-automatique' : 'Boite de vitesse automatique',
                  'Automatique': 'Boite de vitesse automatique',
                  'Mécanique': 'Boite de vitesse manuelle'
                }
    # Homogénéiser le type de carburant entre les différentes sources de données
    dict_carburant = {'Diesel' : 'Diesel', 'Essence' : 'Essence', 'Electrique' : 'Electrique',
                    'Hybride' : 'Hybride', 'Ethanol' : 'Ethanol',
                    'Ess.' : 'Essence', 'Dies.' : 'Diesel', 'Ess./Elec.': 'Hybride',
                    'Ess./GPL': 'Essence/GPL', 'Ess./Bio.' : 'Ethanol',
                    'Dies./Elec.' : 'Hybride'
                    }
    # Corriger le nom du modèle pour qu'il soit aligné avec celui utilisé par le site
    dict_model_corr = {
                    'Abarth 595' : '500',
                    'Abarth 595C' : '500',
                    'Citroen C4 Grand Picasso' : 'C4 PICASSO',
                    'Citroen C4 Picasso' : 'C4 PICASSO', 
                    'Citroen DS3' : 'DS3',
                    'Citroen DS3 Cabrio' : 'DS3',
                    'Citroen DS4' : 'DS4',
                    'Citroen DS4 Crossback' : 'DS4 CROSSBACK',
                    'Citroen DS5' : 'DS5',
                    'Ford Tourneo' : 'TOURNEO COURIER',
                    "Kia cee'd" : 'CEED',
                    "Kia pro_cee'd": 'PROCEED',
                    'Land Rover Evoque' : 'RANGE ROVER EVOQUE',
                    'Mercedes-Benz Classe GLB' : 'GLB',
                    'Mercedes-Benz Classe GLC' : 'GLC',
                    'Mercedes-Benz Classe GLE' : 'GLE',
                    'Opel Crossland X' : 'CROSSLAND' ,
                    'Suzuki SX4 S-Cross' : 'S-CROSS'
                    }
    # Importing scraped data from CSV file
    def __init__(self, file_path, delimiter = ','):
        """
        Initializes the preprocessing class with the specified file path and delimiter.

        Args:
            file_path (str): The path to the file to be processed.
            delimiter (str, optional): The delimiter used in the file. Defaults to ','.

        Raises:
            FileNotFoundError: If the specified file does not exist.

        Attributes:
            file_path (str): The path to the file to be processed.
            delimiter (str): The delimiter used in the file.
            data (None): Placeholder for the data to be loaded from the file.
        """
        # Vérifier si le fichier existe
        try:
            with open(file_path, 'r') as f:
                pass
        except FileNotFoundError:
            raise FileNotFoundError(f"❌File {file_path} not found. Please check the path.")
        # Initialiser les attributs de la classe
        self.file_path = file_path
        self.delimiter = delimiter
        self.data = None

    def load_data(self):
        """
        Loads data from a CSV file specified by the file_path attribute.

        This method attempts to read the CSV file using pandas with the specified
        delimiter. If the file is successfully loaded, the data is stored in the
        `self.data` attribute and a success message is printed. If an error occurs
        during loading, an error message is printed, and `self.data` is set to None.

        Returns:
            pd.DataFrame or None: The loaded data as a pandas DataFrame if successful,
            otherwise None.
        """
        try:
            self.data = pd.read_csv(self.file_path, delimiter=self.delimiter)
            print(f"Data loaded successfully from {self.file_path}")
        except Exception as e:
            print(f"❌Error loading data: {e}")
            self.data = None
        return self.data
    
    # Afficher les premières lignes du DataFrame
    # def display_data(self, n=5):
    #     """
    #     Displays the first `n` rows of the dataset.

    #     Parameters:
    #         n (int, optional): The number of rows to display. Defaults to 5.

    #     Prints:
    #         The first `n` rows of the dataset if `self.data` is not None.
    #         Otherwise, prints a message indicating that no data is available.
    #     """
    #     if self.data is not None:
    #         print(self.data.head(n))
    #     else:
    #         print("No data to display. Please load the data first.")

    # Afficher les colonnes et leurs types de données
    # def display_data_types(self):
    #     """
    #     Displays the data types of each column in the dataset.

    #     This method prints the data types of all columns in the dataset if the 
    #     `data` attribute is not None. If the `data` attribute is None, it 
    #     informs the user that no data is available and suggests loading the data first.

    #     Returns:
    #         None
    #     """
    #     if self.data is not None:
    #         print(self.data.dtypes)
    #     else:
    #         print("No data to display. Please load the data first.")

    # Afficher la dimension du DataFrame
    # def display_shape(self):
    #     if self.data is not None:
    #         print(f"Data shape: {self.data.shape}")
    #     else:
    #         print("No data to display. Please load the data first.")

    # Afficher les statistiques descriptives
    def display_summary_statistics(self):
        if self.data is not None:
            print(self.data.describe())
        else:
            print("No data to display. Please load the data first.")

    # Afficher les valeurs manquantes par colonne
    def display_missing_values(self):
        if self.data is not None:
            missing_values = self.data.isnull().sum()
            print("Missing values in each column:")
            print(missing_values[missing_values > 0])
        else:
            print("No data to display. Please load the data first.")

    # Visualiser les valeurs manquantes par une heatmap
    def visualize_missing_values(self):
        if self.data is not None:
            plt.figure(figsize=(10, 6))
            sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
            plt.title('Missing Values Heatmap')
            plt.show()
        else:
            print("No data to display. Please load the data first.")
    
    # Pretransforming the data by creating some new features
    def pretransform_data(self):
        """
        Preprocess and transform the dataset by cleaning and converting various columns.
        This method performs the following transformations:
        1. Converts the "prix" column to float by removing unwanted characters (€, spaces, commas) and replacing them appropriately.
        2. Converts the "kilométrage" column to float by removing unwanted characters (km, spaces).
        3. Extracts the year from the "annee_mise_en_circulation" column and converts it to integer.
        4. Converts the "puissance" column to float by extracting numeric values before "CV".
        5. Converts the "nb_porte" and "nb_place" columns to integer.
        6. Extracts the CO2 emission value from the "emission_CO2" column and converts it to float.
        7. Calculates the age of the car in days and years based on the difference between the "scraped_at" date and the "annee_mise_en_circulation" date.
        8. Extracts the car brand from the "modele" column and inserts it as a new column "marque".
        9. Replaces brand names in the "marque" column with standardized names using a dictionary.
        Raises:
            Exception: If any error occurs during the preprocessing steps, it prints the error message.
        Notes:
            - Ensure that the `self.data` DataFrame is loaded before calling this method.
            - The `self.dict_name_marque` dictionary should be defined for brand name standardization.
        """
        if self.data is not None:
            try:
                # Convertir la colonne "prix" en float
                # Supprimer les caractères indésirables et convertir en float
                self.data["prix"] = (self.data["prix"]
                                    .str.replace("€", "")
                                    .str.replace("\u202f", "")
                                    .str.replace(" ", "")
                                    .str.replace(",", ".")
                                    .astype(float))
                
                # Convertir la colonne "kilométrage" en float
                # Supprimer les caractères indésirables et convertir en float
                self.data['kilometrage'] = (self.data['kilometrage']
                                            .str.replace("km", "")
                                            .str.replace("\u202f", "")
                                            .str.replace(" ", "")
                                            .astype(float))
    
                # Extraire l'année de mise en circulation et la convertir en int
                self.data["annee"] = (self.data["annee_mise_en_circulation"]
                                    .str.extract(r'(\d{4})')[0] # expression régulière pour extraire l'année de 4 chiffres, \d = any digit, {4} = 4 digits
                                    .astype("Int64")
                                    .astype(str))
                
                # Convertir la colonne "puissance" en puissance_cv
                # Extraire uniquement les chiffres avant "CV"
                self.data["puissance"] = (self.data["puissance"]
                                        .str.extract(r'(\d+) CV')[0] # expression régulière pour extraire la puissance
                                        .astype('Int64') # convertir en integer
                                        )
                # Convertir la colonne "nb_porte" en int puis en object
                self.data["nb_porte"] = (self.data["nb_porte"].astype('Int64').astype(str))

                # Convertir la colonne "nb_place" en int puis en object
                self.data["nb_place"] = (self.data["nb_place"].astype('Int64').astype(str))

                # Convertir la colonne "nb_ancien_proprietaire" en int
                #self.data["nb_ancien_proprietaire"] = (self.data["nb_ancien_proprietaire"].astype('Int64'))

                # Extraire l'émission CO2 et la convertir en float
                self.data["emission_CO2"] = (self.data["emission_CO2"]
                                            .str.extract(r'(\d+) g/km')[0] # expression régulière pour extraire l'émission CO2
                                            .astype(float) # convertir en float
                                            )
                # Calculer l'âge de la voiture en jours, en année et en mois (différence entre la date de scraping et la date de mise en circulation)
                self.data["age_days"] = (pd.to_datetime(self.data["scraped_at"]) -
                                        pd.to_datetime(self.data["annee_mise_en_circulation"], dayfirst=True))
                self.data["age_days"] = self.data["age_days"].apply(lambda x: x.days if pd.notnull(x) else None)
                self.data["age_years"] = self.data["age_days"] / 365 # en années
                self.data["age_years"] = self.data["age_years"].apply(lambda x: round(x, 1) if pd.notnull(x) else None)
                self.data["age_months"] = self.data["age_days"] / 30 # en mois
                self.data["age_months"] = self.data["age_months"].apply(lambda x: round(x, 1) if pd.notnull(x) else None)

                # Création de la colonne kilométrage par age
                self.data["km_per_year"] = self.data["kilometrage"] / self.data["age_years"]
                self.data["km_per_year"] = self.data["km_per_year"].apply(lambda x: round(x, 1) if pd.notnull(x) else None)
                self.data["km_per_month"] = self.data["kilometrage"] / self.data["age_months"]
                self.data["km_per_month"] = self.data["km_per_month"].apply(lambda x: round(x, 1) if pd.notnull(x) else None)

                # Extraire la marque de la voiture et l'insérer à côté de la colonne "modele"
                # en utilisant la première partie de la chaîne avant l'espace
                self.data.insert(self.data.columns.get_loc("modele"), "marque", self.data["modele"].str.split(" ").str[0])
                
                # Remplacer les noms de marque par les noms homogènes et le mettre en majuscule
                self.data["marque"] = self.data["marque"].replace(self.dict_name_marque)
                self.data["marque"] = self.data["marque"].str.upper()
                
                # Homogénéiser le type de Boite de vitesse
                self.data['transmission'] = self.data['transmission'].replace(self.dict_boite)
                
                # Homgénéiser le type de carburant
                self.data['carburant'] = self.data['carburant'].replace(self.dict_carburant)

                # Modèle alternatif à prendre si ce n'est pas disponible dans le site
                if self.dict_model_corr is not None and len(self.dict_model_corr) > 0:
                    self.data['modele_alt'] = self.data['modele'].replace(self.dict_model_corr).str.upper()
                
                # Reformater les colonnes "finition" et "modele" en majuscules
                self.data['finition'] = self.data['finition'].str.upper()
                self.data['modele'] = self.data['modele'].str.upper()
                
                # Concaténer puissance avec la finition
                self.data['finition_puissance'] = self.data['finition'].astype(str) + " " + self.data['puissance'].astype(str) + " CV"
                
                # Numéroter les annonces
                self.data['id_annonce'] = np.arange(1, len(self.data) + 1)

                # Data quality: Rename "marque" Citroen = DS if "modele" contains DS and "marque" contains Citroen
                self.data.loc[(self.data['marque'].str.lower().str.contains('citroen', na=False)) 
                        & (self.data['modele'].str.lower().str.contains('ds', na=False)), "marque"] = "DS"
                
            except Exception as e:
                print(f"❌ Error during pretransformation: {e}")
        else:
            print("No data to pretransform. Please load the data first.")

    @staticmethod
    def calcule_score_proximite(df, var_1:str, var_2:str, var_score:str, len_var=False):
        if len_var:
            # Calculer le score de proximité entre les deux variables
            df[var_score] = df.apply(
                lambda x: len(set(x[var_1].split()) & set(x[var_2].split())) if pd.notnull(x[var_1]) and pd.notnull(x[var_2]) else 0,
                axis=1
            )
        else:
            # Calculer le score de proximité entre les deux variables
            df[var_score] = df.apply(
                lambda x: 1 if pd.notnull(x[var_1]) and pd.notnull(x[var_2]) and x[var_1] == x[var_2] else 0,
                axis=1
            )
        
    # Importer le csv du prix neuf et le fusionner avec le DataFrame 
    def merge_new_price(self, new_price_path):
        if self.data is not None:
            try:
                # Importer le csv du prix neuf
                new_price_df = pd.read_csv(new_price_path, delimiter=self.delimiter)
                # Reformater les colonnes, ajouter le prefixe "np_" pour les colonnes du dataframe prix neuf
                new_price_df['np_marque'] = new_price_df['option_marque_select'].str.upper()
                new_price_df['np_versions'] = new_price_df['Versions'].str.upper()
                new_price_df['np_model'] = new_price_df['source_model'].str.upper()
                new_price_df['np_version_selected'] = new_price_df['Version_selected'].str.upper()
                new_price_df['np_nb_porte'] = new_price_df['Portes'].astype('Int64')
                new_price_df['np_year'] = new_price_df['option_year_select'].astype('Int64').astype(str)
                new_price_df['np_boite'] = new_price_df['Boite'].replace(self.dict_boite)
                new_price_df['np_energie'] = new_price_df['Energie'].replace(self.dict_carburant)
                new_price_df.rename(columns={
                                             "url" : 'np_url_prix_neuf', 
                                             "CO2 (g/km)" : "np_CO2_emission"}, inplace=True
                                    )
                new_price_df['np_prix_neuf'] = (new_price_df['Prix']
                                                .str.replace("€", "")
                                                .str.replace("\u202f", "")
                                                .str.replace(" ", "")
                                                .str.replace(",", ".")
                                                )
                # Concaténer "Versions" avec "Version_selected" pour avoir la finition finale
                new_price_df['np_version_finale'] = new_price_df['np_version_selected'].astype(str) + " " + new_price_df['np_versions'].astype(str)

                # Supprimer les colonnes inutiles
                new_price_df.drop(columns=['option_marque_select', 'Versions', 'source_model', 'Energie',
                                           'Version_selected', 'Portes', 'option_year_select', 'Boite', 'Prix'], inplace=True)
                
                # 1. Left join sur la colonne "marque", "modele" et "annee" => plusieurs finitions possibles à filtrer après
                self.data = self.data.merge(new_price_df, how='left', left_on=['marque', 'modele', 'annee'], right_on=['np_marque', 'np_model', 'np_year'])
                      
                # 2. Sélectionner les versions les plus proches du cible
                # Création d'un système de notation pour évaluer la proximité entre les versions
                # finition_puissance & np_version_finale, transmission & np_boite, nb_porte vs np_nb_porte

                # Calculer le score de proximité entre la finition_puissance & np_version_finale
                self.calcule_score_proximite(self.data, var_1='finition_puissance', var_2='np_version_finale', var_score='note_version_commune', len_var=True)

                # Calculer le score de proximité entre la transmission & np_boite
                self.calcule_score_proximite(self.data, var_1='transmission', var_2='np_boite', var_score='note_transmission_commune')

                # Calculer le score de proximité entre carburant & np_energie
                self.calcule_score_proximite(self.data, var_1='carburant', var_2='np_energie', var_score='note_carburant_commun')

                # Calculer le score de proximité entre le nombre de portes & np_nb_porte
                self.calcule_score_proximite(self.data, var_1='nb_porte', var_2='np_nb_porte', var_score='note_nb_porte_commun')

                # Calculer le score total de proximité
                self.data['note_totale_commune'] = self.data['note_version_commune'] + \
                      self.data['note_transmission_commune'] + self.data['note_carburant_commun'] + self.data['note_nb_porte_commun'] 

                # Garder toutes les lignes ayant les scores les plus élevés pour chaque annonce
                # Calculer la colonne de note maximale pour chaque annonce
                self.data['max_note'] = self.data.groupby(['id_annonce'])['note_totale_commune'].transform('max')
                self.data = self.data[self.data['note_totale_commune'] == self.data['max_note']]
                #self.data.sort_values(by=['id_annonce'], ascending=[True], inplace=True)
                self.data.reset_index(drop=True, inplace=True)

                # Nb matchs par annonce
                self.data['nb_match_par_annonce'] = self.data.groupby('id_annonce')['id_annonce'].transform('count')
                self.data.sort_values(by=['id_annonce'], ascending=[True], inplace=True)
                self.data.reset_index(drop=True, inplace=True)

                # df_found = self.data[self.data['np_prix_neuf'].notnull()]

                # # 3. Left join en utilisant le modèle alternatif si le prix neuf n'est pas trouvable avec le 1er left join
                # df_not_found = self.data[self.data['np_prix_neuf'].isnull()].drop(columns= ["np_version_finale", "np_boite", "np_energie", "np_nb_porte"])
                # df_not_found = df_not_found.merge(new_price_df, how='left', left_on=['marque', 'modele_alt', 'annee'], right_on = ['np_marque', 'np_model', 'np_year'])
                
                # # Calculer le score de proximité entre la finition_puissance & np_version_finale
                # self.calcule_score_proximite(df_not_found, var_1='finition_puissance', var_2='np_version_finale', var_score='note_version_commune_alt', len_var=True)

                # # Calculer le score de proximité entre la transmission & np_boite
                # self.calcule_score_proximite(df_not_found, var_1='transmission', var_2='np_boite', var_score='note_transmission_commune_alt')

                # # Calculer le score de proximité entre carburant & np_energie
                # self.calcule_score_proximite(df_not_found, var_1='carburant', var_2='np_energie', var_score='note_carburant_commun_alt')

                # # Calculer le score de proximité entre le nombre de portes & np_nb_porte
                # self.calcule_score_proximite(df_not_found, var_1='nb_porte', var_2='np_nb_porte', var_score='note_nb_porte_commun_alt')

                # # Calculer le score total de proximité
                # df_not_found['note_totale_commune_alt'] = df_not_found['note_version_commune_alt'] + \
                #       df_not_found['note_transmission_commune_alt'] + df_not_found['note_carburant_commun_alt'] + df_not_found['note_nb_porte_commun_alt']
                
                # df_not_found['max_note_alt'] = df_not_found.groupby(['id_annonce'])['note_totale_commune_alt'].transform('max')
                # df_not_found = df_not_found[df_not_found['note_totale_commune_alt'] == df_not_found['max_note_alt']]
                # #self.data.sort_values(by=['id_annonce'], ascending=[True], inplace=True)
                # df_not_found.reset_index(drop=True, inplace=True)

                # # Nb matchs par annonce
                # df_not_found['nb_match_par_annonce_alt'] = df_not_found.groupby('id_annonce')['id_annonce'].transform('count')
                # df_not_found.sort_values(by=['id_annonce'], ascending=[True], inplace=True)
                # df_not_found.reset_index(drop=True, inplace=True)

                # # 4. Combiner
                # self.data = pd.concat([df_found, df_not_found], ignore_index = True)

            except Exception as e:
                print(f"❌Error merging new price data: {e}")
            return new_price_df
    
    # Une annonce peut avoir plusieurs versions qui matchent
    # Déterminer le prix neuf final par annonce par la moyenne des prix neufs ou la médiane
    # et le stocker dans la colonne np_prix_neuf_moy, np_prix_neuf_median

    def fix_new_price (self):
        if self.data is not None:
            # Calculer prix neuf moyen et médian par annonce
            prix_stats = self.data.groupby('id_annonce').agg({
                'np_prix_neuf': ['mean', 'median', 'std']
                }).reset_index()
            prix_stats.columns = ['id_annonce', 'mean', 'median', 'std']

            # Statistiques descriptives des prix neufs par annonce
            #print("Statistiques descriptives des prix neufs par annonce:")
            #print(self.data.groupby('id_annonce')['np_prix_neuf'].describe())

            # Supprimer les outliers par annonce (en se basant sur l'IQR)
            final_prices = []
            for id_ad in prix_stats['id_annonce']:
                prix = self.data[self.data['id_annonce'] == id_ad]['np_prix_neuf']
                # Calculer l'IQR
                if len(prix) > 1:
                    q1 = prix.quantile(0.25)
                    q3 = prix.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    # Filtrer les prix en dehors de l'IQR
                    prix_valide = prix[(prix >= lower_bound) & (prix <= upper_bound)]
                    prix_valide_moyen = prix_valide.mean() if not prix_valide.empty else np.nan
                else:
                    # Si une annonce n'a qu'un seul prix, on le garde
                    prix_valide_moyen = prix.iloc[0] if not prix.empty else np.nan
                final_prices.append({'id_annonce': id_ad, 'prix_neuf_moyen_iqr': prix_valide_moyen})

            # Convertir la liste en DataFrame
            final_prices_df = pd.DataFrame(final_prices)

            # Calculer la moyenne des prix neuf par annonce avant la suppression des outliers
            self.data['np_prix_neuf_moy'] = self.data.groupby('id_annonce')['np_prix_neuf'].transform('mean')
            # Calculer la médiane des prix neuf par annonce avant la suppression des outliers
            self.data['np_prix_neuf_median'] = self.data.groupby('id_annonce')['np_prix_neuf'].transform('median')
            # Fusionner les prix finaux avec le DataFrame principal
            self.data = self.data.merge(final_prices_df, on='id_annonce', how='left')
            # Calculer le ratio de Valeur résiduelle (VR) entre le prix d'occasion et le prix neuf moyen
            self.data['ratio_vr'] = self.data['prix'] / self.data['np_prix_neuf_moy']
            self.data['ratio_vr_iqr'] = self.data['prix'] / self.data['prix_neuf_moyen_iqr']

            # Visualiser prix neuf moyen par annonce avant et après la suppression des outliers
            # Supprimer les doublons
            prix_neuf_moyen = self.data[['id_annonce', 'np_prix_neuf_moy', 'prix_neuf_moyen_iqr', 'prix', 'ratio_vr', 'ratio_vr_iqr']].drop_duplicates()

            # Vérifier que prix est bien inférieur à np_prix_neuf_moy et prix_neuf_moyen_iqr
            prix_neuf_moyen_ko = prix_neuf_moyen[prix_neuf_moyen['prix'] > prix_neuf_moyen['np_prix_neuf_moy']]
            prix_neuf_moyen_iqr_ko = prix_neuf_moyen[prix_neuf_moyen['prix'] > prix_neuf_moyen['prix_neuf_moyen_iqr']]
            print(f"Nombre d'annonces: {prix_neuf_moyen.shape[0]}")
            print(f"Nombre d'annonces avec prix neuf moyen avant suppression des outliers < prix d'occasion: {prix_neuf_moyen_ko.shape[0]}")
            print(f"Nombre d'annonces avec prix neuf moyen après suppression des outliers < prix d'occasion: {prix_neuf_moyen_iqr_ko.shape[0]}")

            # Scatter plot & Hist plot des prix neufs moyens avant et après la suppression des outliers
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Scatter plot", "Histogramme"))
            # Scatter plot des prix neufs moyens avant et après la suppression des outliers
            fig.add_trace(go.Scatter(
                x=prix_neuf_moyen['np_prix_neuf_moy'],
                y=prix_neuf_moyen['prix_neuf_moyen_iqr'],
                mode='markers',
                name='Prix Neuf Moyen Avant et Après Suppression des Outliers',
                marker=dict(color='red')
            ), row=1, col=1)

            # Histogramme des prix neufs moyens avant et après la suppression des outliers
   
            fig.add_trace(go.Histogram(
                x=prix_neuf_moyen['np_prix_neuf_moy'],
                name='Prix Neuf Moyen Avant Suppression des Outliers',
                opacity=0.8,
                marker_color='blue'
            ), row=1, col=2)
            fig.add_trace(go.Histogram(
                x=prix_neuf_moyen['prix_neuf_moyen_iqr'],
                name='Prix Neuf Moyen Après Suppression des Outliers',
                opacity=0.8,
                marker_color='green'
            ), row=1, col=2)

            # fig.add_trace(go.Histogram(
            #     x=prix_neuf_moyen['prix'],
            #     name="Prix d'occasion",
            #     opacity=0.8,
            #     marker_color="grey"
            # ), row=1, col=2)  

            fig.update_layout(
                title_text="Comparaison des Prix Neuf Moyens par Annonce Avant et Après Suppression des Outliers",
                xaxis_title="Prix Neuf Moyen Avant Suppression des Outliers",
                yaxis_title="Prix Neuf Moyen Après Suppression des Outliers",
                barmode='overlay',
                height=600,
                width=1200
            )
            fig.show()
        return self.data

 # Tester la classe DataPreprocessor
if __name__ == "__main__":
    # Obtenir le chemin du répertoire actuel et le chemin du répertoire racine du projet
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    # Le chemin vers les données d'annonce
    data_path = os.path.join(project_root, "data", "raw_data", "autohero.csv")
    
    # Créer une instance de la classe DataPreprocessor
    preprocessor = DataPreprocessor(file_path=data_path)
    
    # Charger les données
    preprocessor.load_data()
    
    # Afficher la dimension du DataFrame
    print(f"La dimension du DataFrame est: {preprocessor.data.shape}") 
        
    # Prétransformer les données
    preprocessor.pretransform_data()

    print(f"Après la pré-transformation, la nouvelle dimension est: {preprocessor.data.shape}") 
    
    # Rename "marque" Citroen = DS if "modele" contains DS and "marque" contains Citroen
    # preprocessor.data.loc[(preprocessor.data['marque'].str.lower().str.contains('citroen', na=False)) 
    #                       & (preprocessor.data['modele'].str.lower().str.contains('ds', na=False)), "marque"] = "DS"

    # Récupérer le prix neuf
    new_price_path = os.path.join(project_root, "data", "raw_data", "prix_neuf_voitures_vf.csv")
    preprocessor.merge_new_price(new_price_path)
    print(f"Après la récupération du prix neuf, la nouvelle dimension est: {preprocessor.data.shape}") 
    print(f"Liste des types de données après la pré-transformation:\n {preprocessor.data.dtypes}")
    df = preprocessor.data.copy()
    print(f"Id annonces: {df['id_annonce'].describe()}")

    # Nombre des annonces où le prix neuf est renseigné
    print(f"Nombre total d'annonces: {preprocessor.data['id_annonce'].drop_duplicates().shape[0]}")
    print(f"Nombre d'annonces où le prix neuf est renseigné: {preprocessor.data[preprocessor.data['np_prix_neuf'].notnull()][['id_annonce']].drop_duplicates().shape[0]}")
    print(f"Nombre d'annonces où le prix neuf n'est pas trouvé: {preprocessor.data[preprocessor.data['np_prix_neuf'].isnull()][['id_annonce']].drop_duplicates().shape[0]}")