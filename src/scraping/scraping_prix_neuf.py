import re
import time
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from datetime import datetime
from pandas import DataFrame
import constants as c
import os
import glob


class TextPreprocessor:
    """
    Classe pour le nettoyage et la standardisation du texte:
    supprimer les accents, normaliser les espaces
    """
    def __init__(self):
        pass
    
    @staticmethod
    def remove_accents(text):
        accents = {
            'a': r'[àáâãäå]',
            'e': r'[èéêë]',
            'i': r'[ìíîï]',
            'o': r'[òóôõö]',
            'u': r'[ùúûü]',
            'c': r'[ç]',
            'n': r'[ñ]',
            'A': r'[ÀÁÂÃÄÅ]',
            'E': r'[ÈÉÊË]',
            'I': r'[ÌÍÎÏ]',
            'O': r'[ÒÓÔÕÖ]',
            'U': r'[ÙÚÛÜ]',
            'C': r'[Ç]',
            'N': r'[Ñ]',
        }
        for remplacement, pattern in accents.items():
            text = re.sub(pattern, remplacement, text)
        return text
    
    @staticmethod
    def clean_model_name(model):
        model = TextPreprocessor.remove_accents(model)
        return re.sub(r'[^\w\s\-!]', '', model).strip().lower().replace(' ', '-')

class DataLoader:
    """ Classe pour charger et prétraiter les données d'annonce"""
    def __init__(self, file_path):
        self.file_path = file_path
        # Vérifier si le fichier existe
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                pass
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File {self.file_path} not found. Please check the path.") from exc  
        self.file_path = file_path
        self.data = None
        self.dict_model_corrections = {}
        self.dict_marque_corrections = {}
    
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.pretransform_data()
    
    def pretransform_data(self):
        if self.data is not None:
            # Extraction année en 4 chiffres
            self.data["annee"] = (self.data["annee_mise_en_circulation"]
                                  .str.extract(r'(\d{4})')[0]
                                  .astype('Int64')
                                  )
            # Extraction du nom de la marque
            self.data.insert(self.data.columns.get_loc("modele"), 
                             "marque", self.data["modele"].str.split(" ").str[0])
            
            # Corriger le nom de marque si besoin
            if self.dict_marque_corrections is not None:
                self.data['marque'] = self.data['marque'].replace(self.dict_marque_corrections)
            self.data.loc[( self.data['marque'].str.lower().str.contains('citroen', na=False)) 
                        & ( self.data['modele'].str.lower().str.contains('ds', na=False)), "marque"] = "DS"

            # Corriger le nom de modèle si besoin
            if self.dict_model_corrections is not None:
                self.data['modele'] = self.data["modele"].replace(self.dict_model_corrections)

            # Création d'une colonne modèle & année
            self.data['modele_annee'] = self.data['modele']+ ' ' + self.data['annee'].astype(str)

            # Nombre d'occurence par marque, modèle et année
            self.data = self.data[['marque', 'modele', 'annee', 'modele_annee']].value_counts().reset_index(name="nb_occurences")
            self.data.reset_index(drop=True, inplace=True)
            return self.data
        
class WebScraperPrixNeufSetup:
    """ Classe pour créer l'environnement de web-scraping """
    def __init__(self, wait_time=10, headless = True):
        self.base_url = c.BASE_URL_PRIX_NEUF
        self.wait_time = wait_time
        self.headless = headless
        self.text_processor = TextPreprocessor()
        
        # selctor pour le scraping
        self.id_marque = 'brands'
        self.id_modele = 'models'
        self.id_annee = 'year'
        self.id_version = 'modelscomm'
        self.css_popup = '.didomi-continue-without-agreeing'
        self.css_tableau_prix = 'table.listingTab'

    def collect_prix_neuf(self, marque, modele, annee):
        try:
            self.driver = self.init_driver()
            self.driver.get(self.base_url)
            time.sleep(2)

            # Accepter le popup de consentement
            self.accept_popup(self.css_popup)
            time.sleep(2)

            df_all_versions = pd.DataFrame()

            # Sélectionner la marque
            marque_dropdown = self.driver.find_element(By.ID, self.id_marque)
            option_marque, match_type_marque = self.select_option_contain(marque_dropdown, marque)
            print(f"Marque sélectionnée: {option_marque} ({match_type_marque})")
            time.sleep(2)

            # Sélectionner le modèle
            modele_dropdown = self.driver.find_element(By.ID, self.id_modele)
            option_modele, match_type_modele = self.select_option_contain(modele_dropdown, modele)
            print(f"Modèle sélectionné: {option_modele} ({match_type_modele})")
            time.sleep(2)

            # Sélectionner l'année
            annee_dropdown = self.driver.find_element(By.ID, self.id_annee)
            option_annee, match_type_annee = self.select_option_contain(annee_dropdown, str(annee))
            print(f"Année sélectionnée: {option_annee} ({match_type_annee})")
            time.sleep(2)
            
            # Sélectionner la version
            modelscomm_dropdown = self.driver.find_element(By.ID, self.id_version)
            select_modelscomm  = Select(modelscomm_dropdown)
            versions = [option.text for option in select_modelscomm.options
                        if option.get_attribute("value").strip() != ""]
            print(f"Versions disponibles: {versions}")

            if versions:
                for version in versions:
                    data = []
                    option_marque_cleaned = self.text_processor.clean_model_name(option_marque)
                    option_version_cleaned = self.text_processor.clean_model_name(version)
                    url_version = f"{self.base_url}/modele--{option_marque_cleaned}-{option_version_cleaned}/{option_annee}"
                    print(f"Extraire la fiche technique de: {version} - année {option_annee}")
                    print(f"Search url est : {url_version}")

                    try:
                        self.driver.get(url_version)
                        time.sleep(2)

                        # Récupérer le tableau de prix
                        table = WebDriverWait(self.driver, self.wait_time).until(EC.presence_of_element_located((By.CSS_SELECTOR, self.css_tableau_prix)))

                        # Extraire toutes les lignes du tableau, header inclus
                        rows = table.find_elements(By.TAG_NAME, "tr")

                        # Extraire le header (1ère ligne)
                        headers = rows[0].find_elements(By.TAG_NAME, 'th')
                        headers_texts = [header.text for header in headers]

                        # Extraire les données dans les lignes restantes
                        # et les ajouter à la liste data
                        for row in rows[1:]:
                            cells = row.find_elements(By.TAG_NAME, 'td')
                            row_data = [cell.text for cell in cells]
                            if row_data:
                                data.append(row_data)

                        # Sauvegarder le tableau (list data) dans un DataFrame
                        df = pd.DataFrame(data, columns=headers_texts)

                        # Ajouter les colonnes pour la marque, le modèle et l'année source
                        df['url'] = self.driver.current_url
                        df['option_marque_select'] = option_marque
                        df['option_modele_select'] = option_modele
                        df['option_year_select'] = option_annee
                        df['match_type_marque'] = match_type_marque
                        df['match_type_modele'] = match_type_modele
                        df['match_type_annee'] = match_type_annee

                        # Concaténer le DataFrame avec df_all_versions
                        df_all_versions = pd.concat([df_all_versions, df], ignore_index=True)

                    except Exception:
                        print(f"Erreur lors de l'extraction des données pour la version {version} - année {option_annee}")
                        continue
                # Fermer le driver
                self.driver.quit()
                return df_all_versions
        except Exception as e:
            print(f"Erreur lors de la collecte des prix: {e}")
            self.driver.quit()
            return None 
        
    def init_driver(self):    
        service = Service(EdgeChromiumDriverManager().install())
        options = webdriver.EdgeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")  # Cache Selenium
        if self.headless:
            options.add_argument('--headless')  # Exécute le navigateur en arrière-plan
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")  # Facultatif : exécute en arrière-plan
        driver = webdriver.Edge(service=service, options=options)
        driver.maximize_window()  # Maximiser la fenêtre du navigateur
        return driver
    
    def accept_popup(self, css_selector):
        try:
            accept_button = WebDriverWait(self.driver, self.wait_time).until(EC.element_to_be_clickable((By.CSS_SELECTOR, css_selector)))
            accept_button.click()
            time.sleep(2)  # Attendre que le popup disparaisse
        except Exception:
            print("Popup de consentement non trouvé ou déjà fermé.")
        return None
    
    def select_option_contain(self, select_element, partial_text):
        select = Select(select_element)
        partial_text = self.text_processor.remove_accents(partial_text)

        # Vérifier les correspondances exactes entre les options du site web et partial_text
        for option in select.options:
            if (option.get_attribute("value").strip() != "" and partial_text.lower() == option.text.lower()):
                select.select_by_visible_text(option.text)
                print(f"Found exact match: {option.text}")
                option_select = "Very Exact match"
                return option.text, option_select                     

        # Vérifier les correspondances après nettoyage
        for option in select.options:
            if (option.get_attribute("value").strip() != "" and 
                re.sub(r"[\s\-]", "", partial_text.strip().lower()) == re.sub(r"[\s\-]", "", option.text.strip().lower())
                ): # remove any space and "-"
                select.select_by_visible_text(option.text)
                print(f"Found exact match: {option.text}")
                option_select = "Exact match"
                return option.text, option_select
                
        # Rechercher la meilleure correspondance partielle    
        best_match = None
        max_common_chars = 0
    
        for option in select.options:
            if option.get_attribute("value").strip() != "":
                option_text = re.sub(r"[\s\-]", "", option.text.strip().lower()) # remove any space
                partial_text_lower = re.sub(r"[\s\-]", "", partial_text.strip().lower()) # remove any space
                
                # Count common characters
                common_chars = sum(1 for c in option_text if c in partial_text_lower)
                
                # Check if this is a partial match and has more common characters
                if ((partial_text_lower in option_text) or (option_text in partial_text_lower)) and common_chars > max_common_chars:
                    max_common_chars = common_chars
                    best_match = option
        # Retain the best partial match = the option that has more common characters with the partial text
        if best_match is not None:
            select.select_by_visible_text(best_match.text)
            print(f"Found best partial match: {best_match.text}")
            option_select = "Partial match"
            return best_match.text, option_select
    
class WebScraperPrixNeufRunner:
    def __init__(self, annonce_data_path,  output_dir):
        self.data_loader = DataLoader(annonce_data_path)
        self.scraper = WebScraperPrixNeufSetup()
        self.output_dir = output_dir

    def run_collection(self, batch_pattern, out_pattern, nb_batch=10):
        # Charger les données d'annonce
        df = self.data_loader.load_data()

        # Diviser le DataFrame en lots
        #self.split_into_batches(df, nb_batch, batch_pattern)

        # Traiter chaque lot
        print(f"Number of models & years to be collected: {len(pd.unique(df["modele_annee"]))} \n")
        files = glob.glob(os.path.join(self.output_dir, f'{batch_pattern}_*.csv'))
        collects = []
        uncollects = []
        df_collected = pd.DataFrame()
        print(f"---Web Scraping by batch---\n")
        for i, path_split in enumerate(files):
            collect, uncollect, df_batch = self.process_batch(path_split, i+1, out_pattern)
            if collect:
                collects.extend(collect)
                df_collected = pd.concat([df_collected, df_batch], ignore_index = True)
            if uncollect:
                uncollects.extend(uncollect) 

        # Liste des collectés
        print(f"\n 📊 Scraping completed:")
        print(f"✅ Total models & years collected: {len(collects)}")
        print(collects)
        print(f"⚠️ Total models & years remaining for collection: {len(uncollects)}")
        print(uncollects)
        df_ko = df[df['modele_annee'].isin(uncollects)]
        print(df_ko.groupby('modele_annee')['nb_occurences'].sum().reset_index().sort_values('nb_occurences', ascending = False))

        # Liste des non collectés
        return collects, uncollects, df_collected

    def split_into_batches(self, df, nb_batch, batch_pattern):
        """ Diviser le DataFrame en lots de taille batch_size et les enregistrer dans des fichiers CSV
            Pour lancer le scraping en plusieurs batchs et enregistrer au fur et à mesure les résultats
            Eviter de relancer si le script plante
            df : DataFrame
            batch_size : int
            return : None
          """
        
        if nb_batch <= 0:
            raise ValueError("Le nombre de batches doit être supérieur à 0.")
    
        n = len(df)
        batch_sizes = [(n + i) // nb_batch for i in range(nb_batch)]  # tailles équilibrées
        indices = [0] + list(pd.Series(batch_sizes).cumsum())

        list_batches = [df.iloc[indices[i]:indices[i+1]] for i in range(nb_batch)]
    
        somme_originale = len(df)
        somme_lignes = sum(len(split) for split in list_batches)

        print(f"Somme des lignes de la base brute = {somme_originale}")
        print(f"Somme des lignes de tous les splits dataframe = {somme_lignes}")
        print(f'# Data split en: {len(list_batches)} batches')

        # Save each split dataframe to CSV
        for i, split_df in enumerate(list_batches):
            #csv_path = f'{csv_root}/split_car_models_{i+1}.csv'
            csv_path = os.path.join(self.output_dir, f'{batch_pattern}_{i+1}.csv')
            split_df.to_csv(csv_path, index=False)
            print(f'Split {i+1} saved to {csv_path}')
        return None
    
    def process_batch(self, batch_path, batch_num, pattern_out):
        output_file = f"{self.output_dir}/{pattern_out}_{batch_num}.csv"

        # Charger le lot
        batch_data = pd.read_csv(batch_path)
        total_models = len(batch_data)
        batch_data['modele_annee'] = batch_data['modele'] + ' ' + batch_data['annee'].astype(str)
        unique_modele_annee = sorted(pd.unique(batch_data["modele_annee"]))
        unique_modele_annee_collected =[]
        
        if not os.path.exists(output_file):
            price_data_batch = pd.DataFrame()
            #nb_found = 0
            
            for idx, row in batch_data.iterrows():
                print(f"Traitement du modèle {idx+1}/{total_models}: {row['modele']} ({row['annee']})")
                price_data = self.scraper.collect_prix_neuf(row['marque'], row['modele'], row['annee'])

                if price_data is not None:
                    # Ajouter des colonnes pour identifier la source des données
                    price_data.insert(0, 'source_model', row['modele'])
                    price_data.insert(1, 'source_year', row['annee'])

                    # Concaténer les données de prix avec le DataFrame principal
                    price_data_batch = pd.concat([price_data_batch, price_data], ignore_index=True)

                    print(f"✅Successfully collected for batch {batch_num}, {row['modele']} ({row['annee']})")
                else:
                    print(f"❌Error processing batch {batch_num}, {row['modele']} ({row['annee']})")

            # Exporter les données dans un fichier CSV
            price_data_batch.to_csv(output_file, index=False, encoding='utf-8-sig')
        else:
            price_data_batch = pd.read_csv(output_file)
            print(f"File {output_file} already exists, skipping...")

        print(f"\n 🆗 Scraping completed for batch {batch_num}. Total entries collected: {len(price_data_batch)}")
        print(f"Total models & years to be collected for batch {batch_num}: {len(unique_modele_annee)}")

        # Print le nombre de couple modèles & années trouvés, sans doublons
        if not price_data_batch.empty:
            
            #nb_found = len(price_data_batch[['source_model', 'source_year']].drop_duplicates())
            price_data_batch['modele_annee'] = price_data_batch['source_model'] + ' ' + price_data_batch['source_year'].astype(str)
            # Retreive only elements present in unique_modele_annee
            unique_modele_annee_collected = sorted(pd.unique(price_data_batch["modele_annee"]))

            print(f"✅ Total unique models & years found in batch {batch_num}: {len(unique_modele_annee_collected)}")
    
        # Print le nombre de couple modèles & années non trouvés
        remaining_to_collect = set(unique_modele_annee) - set(unique_modele_annee_collected)
        print(f"⚠️ Total models & years not found in batch {batch_num}: {len(remaining_to_collect)}")

        return unique_modele_annee_collected, remaining_to_collect, price_data_batch
    
class WebScraperPrixNeufRunnerFix:
    def __init__(self, data_in, output_dir, dict_model_correction = {}):
        self.data_in = data_in
        self.scraper = WebScraperPrixNeufSetup() # Augmenter le wait time à 20 sec
        self.dict_model_correction = dict_model_correction
        self.output_dir = output_dir 
        self.data_in['modele_annee'] = self.data_in['modele'] + ' ' + self.data_in['annee'].astype(str)
        self.unique_modele_annee = sorted(pd.unique(self.data_in["modele_annee"]))

    def run_collection_fix(self, pattern_out):
        output_file = f"{self.output_dir}/{pattern_out}.csv"
        if not os.path.exists(output_file):
            # Charger le lot
            total_models = len(self.data_in)
            price_data_batch = pd.DataFrame()
            # Create columns to keep original "marque" & "modele" name
            self.data_in["marque_init"] = self.data_in["marque"]
            self.data_in['modele_init'] = self.data_in['modele']
            # Corriger le nom du modèle si besoin
            if self.dict_model_correction:
                self.data_in['modele'] = self.data_in['modele'].replace(self.dict_model_correction)

            for idx, row in self.data_in.iterrows():
                print(f"Traitement du modèle {idx+1}/{total_models}: {row['modele']} ({row['annee']})")
                price_data = self.scraper.collect_prix_neuf(row['marque'], row['modele'], row['annee'])

                if price_data is not None:
                    # Ajouter des colonnes pour identifier la source des données
                    price_data.insert(0, 'source_model', row['modele_init'])
                    price_data.insert(1, 'source_year', row['annee'])
                    price_data.insert(2, 'model_alternative', row['modele'])

                    # Concaténer les données de prix avec le DataFrame principal
                    price_data_batch = pd.concat([price_data_batch, price_data], ignore_index=True)

                    print(f"✅Successfully collected for {row['modele']} ({row['annee']})")
                else:
                    print(f"❌Error processing {row['modele']} ({row['annee']})")
            # Exporter les données dans un fichier CSV
            price_data_batch.to_csv(output_file, index=False, encoding='utf-8-sig')      
        else:
            price_data_batch = pd.read_csv(output_file)
            print(f"File {output_file} already exists, skipping...")

        print(f"\n 🆗 Scraping completed. Total entries collected: {len(price_data_batch)}")
        # Print le nombre de couple modèles & années trouvés, sans doublons
        if not price_data_batch.empty:
            price_data_batch['modele_annee'] = price_data_batch['source_model'] + ' ' + price_data_batch['source_year'].astype(str)
            unique_modele_annee_collected = sorted(pd.unique(price_data_batch["modele_annee"]))
            print(f"✅ Total unique models & years found: {len(unique_modele_annee_collected)}")
            print(unique_modele_annee_collected)
    
        # Print le nombre de couple modèles & années non trouvés
        remaining_to_collect = set(self.unique_modele_annee) - set(unique_modele_annee_collected)
        print(f"⚠️ Total models & years not found: {len(remaining_to_collect)}")
        print(remaining_to_collect)
        df_ko = self.data_in[self.data_in['modele_annee'].isin(remaining_to_collect)]
        print(df_ko.groupby('modele_annee')['nb_occurences'].sum().reset_index().sort_values('nb_occurences', ascending = False))
        return remaining_to_collect, price_data_batch

class FinalDataCollected:
    def __init__(self, list_df:list[DataFrame]):
        self.list_df = list_df
    def combined_df(self, output_csv):
        # Concatenate the dataframes
        df_combined = pd.concat([df for df in self.list_df], ignore_index=True)
        # Remove duplicates
        df_combined_nodup = df_combined.drop_duplicates(subset=['source_model', 'source_year', 'option_marque_select', 
                                                                'option_modele_select', 'option_year_select', 'Versions'])
        # Remove missing price
        df_combined_nodup = df_combined_nodup[df_combined_nodup["Prix"].notna()]

        # Extract the version from the URL
        df_combined_nodup['Version_selected'] = df_combined_nodup['url'].str.extract(r'--(.*?)/')
        df_combined_nodup['Version_selected'] = df_combined_nodup['Version_selected'].str.replace('-', ' ', regex=False)

        df_combined_nodup['option_marque_select'] = df_combined_nodup['option_marque_select'].str.upper()
        df_combined_nodup['Versions'] = df_combined_nodup['Versions'].str.upper()
        df_combined_nodup.drop(columns=['source_year', 'option_modele_select',
                                        'match_type_marque', 'match_type_modele', 'match_type_year'], inplace=True)

        # Exporter les données dans un fichier CSV pour l'utilisation ultérieure
        df_combined_nodup.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Data exported successfully to {output_csv}")

        # Print nombre modèle & année collectés
        print(f"Total models & years collected: {len(pd.unique(df_combined_nodup["modele_annee"]))}")
        return df_combined_nodup
    
if __name__ == "__main__":

    #######################################
    #           Chemins                   #
    #######################################
    # Obtenir le chemin du répertoire actuel et le chemin du répertoire racine du projet
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # Le chemin vers les données d'annonce
    annonce_data_path = os.path.join(project_root, "data", "raw_data", "autohero.csv")

    output_prix = os.path.join(project_root, "data", "scraping_prix_neuf")
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_prix):
        os.makedirs(output_prix)

    #######################################
    #           1. First attempt          #
    #######################################
    collecteur = WebScraperPrixNeufRunner(
        annonce_data_path = annonce_data_path,
        output_dir = output_prix
    )

    # Corriger les noms de marques
    collecteur.data_loader.dict_marque_corrections= c.DICT_MARQUE
    collected_1, uncollected_1, df_prix_neuf_1 = collecteur.run_collection(
        batch_pattern = "split_car_models",
        out_pattern = "prix_neuf_voitures_pack",
        nb_batch = 16
    )

    ##############################################
    #               2. Second attempt            #
    ##############################################
    # Dataframe des modèles restant à collecter
    df = collecteur.data_loader.load_data()
    df['modele_annee'] = df['modele'] + ' ' + df['annee'].astype(str)
    df2 = df[df['modele_annee'].isin(uncollected_1)]
    df2 = df2.reset_index(drop=True)
    
    collecteur_2 = WebScraperPrixNeufRunnerFix(
        data_in = df2,
        output_dir = output_prix
    )
    # Augmenter wait time à 20 secondes
    collecteur_2.scraper.wait_time = 20
    # Lancer le deuxième essai
    uncollected_2, df_prix_neuf_2 = collecteur_2.run_collection_fix(
        pattern_out = "prix_neuf_voitures_essaie_2",
    )

    ##############################################
    #             3. Third attempt               #
    ##############################################
    # Dataframe des modèles restant à collecter
    df3 = df2[df2['modele_annee'].isin(uncollected_2)].reset_index(drop=True)
    # Liste des modèles (sans année)
    list_modele_df3 = pd.unique(df3['modele'])
    print(f"Liste des modèles à collecter (sans année): {list_modele_df3}")
    # modifier le nom du modèle pour que ce soit cohérent avec celui utilisé par le site caradisiac 
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

    collecteur_3 = WebScraperPrixNeufRunnerFix(
        data_in = df3,
        output_dir = output_prix,
        dict_model_correction= dict_model_corr
    )
    # Augmenter wait time à 20 secondes
    collecteur_3.scraper.wait_time = 20
    # Lancer le 3è essai
    uncollected_3, df_prix_neuf_3 = collecteur_3.run_collection_fix(
        pattern_out = "prix_neuf_voitures_essaie_3",
    )

    ##############################################
    #             4. Fourth attempt              #
    ##############################################
    # Dataframe des modèles restant à collecter
    df4 =  df2[df2['modele_annee'].isin(uncollected_3)].reset_index(drop=True)
    collecteur_4 = WebScraperPrixNeufRunnerFix(
        data_in = df4,
        output_dir = output_prix,
        dict_model_correction= dict_model_corr
    )
    # Augmenter wait time à 20 secondes
    collecteur_4.scraper.wait_time = 20
    # Lancer le 4è essai
    uncollected_4, df_prix_neuf_4 = collecteur_4.run_collection_fix(
        pattern_out = "prix_neuf_voitures_essaie_4",
    )

    ##############################################
    #              Combiner les résultats        #
    ##############################################
    output_data_final = os.path.join(project_root, "data", "scraping_prix_neuf", "prix_neuf_voitures_vf.csv")
    collecteur_all = FinalDataCollected(list_df=[df_prix_neuf_1, df_prix_neuf_2, df_prix_neuf_3, df_prix_neuf_4])
    df_combined = collecteur_all.combined_df(output_data_final)