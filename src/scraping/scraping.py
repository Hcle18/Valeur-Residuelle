from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import pandas as pd
import time

# Fonction pour initialiser le driver Selenium
def init_driver():
    # Initialize the Edge driver
    service = Service(EdgeChromiumDriverManager().install())
    options = webdriver.EdgeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")  # Cache Selenium
    options.add_argument('--headless')  # Exécute le navigateur en arrière-plan
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")  # Facultatif : exécute en arrière-plan

    driver = webdriver.Edge(service=service, options=options)
    driver.maximize_window()  # Maximiser la fenêtre du navigateur
    
    return driver

# Fonction pour accepter le popup de consentement
def accept_popup(driver):
    try:
        # Attendre que le popup de consentement apparaisse et l'accepter'
        #consent_button = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "root___3ffa6")))
        accept_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-qa-selector='cookie-consent-accept-all']")))
        accept_button.click()
        time.sleep(2)  # Attendre que le popup disparaisse
    except:
        print("Popup de consentement non trouvé ou déjà fermé.")

def accept_popup_general(driver, css_selector):
    try:
        # Attendre que le popup de consentement apparaisse et l'accepter'
        #consent_button = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "root___3ffa6")))
        accept_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.CSS_SELECTOR, css_selector)))
        accept_button.click()
        time.sleep(2)  # Attendre que le popup disparaisse
    except:
        print("Popup de consentement non trouvé ou déjà fermé.")

# Fonction pour extraire les liens de la page courante
def get_link_annonces_scroll(driver):
    # Calculer la hauteur de la page avant le scrolling
    last_height = driver.execute_script("return document.body.scrollHeight")
    #compteur = 0
    # Déclarer une liste pour stocker tous les liens
    all_liens = []

    # Scrolling down pour charger plus d'annonces
    while True:
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5) # Attendre le chargement des annonces 
        
        # Récupérer les annonces sur la page après le scrolling
        #annonces = driver.find_elements(By.CSS_SELECTOR, "a.link___2Maxt")  
        annonces =  driver.find_elements(By.CSS_SELECTOR, 'a[data-qa-selector="ad-card-link"]')
        liens_ = [annonce.get_attribute("href") for annonce in annonces if annonce.get_attribute("href")]

        # Ajouter les liens à la liste all_liens
        all_liens.extend(liens_)

        # Calculer la nouvelle hauteur et comparer avec l'ancienne: si elles sont égales, on a atteint le bas de la page
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        # Si la hauteur n'a pas changé, on sort de la boucle
        if new_height == last_height:
            break
        last_height = new_height

        #print(f"Le nombre d'annonces chargées dans le compteur numéro {compteur} sont: {len(liens_)}")
        #print(f"Le nombre total d'annonces chargées est: {len(all_liens)}")
        #compteur += 1

    # Extraire les liens des annonces et enlever les doublons
    liens = list(set(all_liens))
    print(f"Le nombre total d'annonces chargées est: {len(liens)}")
    return liens

# Fonction pour extraire les informations d'une annonce
def get_info_annonce(driver, dict_data):
    time.sleep(5)  # Attendre que la page se charge
    data  = {"scraped_at": datetime.today().strftime('%Y-%m-%d')}

    # Extraire les informations de l'annonce
    for label, selector in dict_data.items():
        try:
            element = driver.find_element(By.CSS_SELECTOR, selector)
            data[label] = element.text.strip()
        except Exception as e:
            print(f"Erreur lors de l'extraction de {label}: {e}")
            data[label] = ''
    return data

# Fonction principale de scraping autohero
def scraping_autohero(base_url, year_min, km_max, csv_path, dict_data):
    ''' 
    Scraping des annonces sur le site autohero.com
    Arguments:
        base_url : str : URL de la page à scraper
        year_min : int : Année minimum du véhicule
        km_max : int : Kilométrage maximum du véhicule
        csv_path : str : Chemin du fichier CSV de sortie
    '''
    # Initialiser le driver Edge
    driver = init_driver()

    # Ouvrir la page avec Selenium
    url = base_url + "/search/" + "?yearMin=" + str(year_min) + "&mileageMax=" + str(km_max) + ""
    driver.get(url)
    time.sleep(5)  # Attendre que la page se charge

    # Déclarer une variable pour sauvegarder les données scrapées
    all_data = []

    # Accepter le popup de consentement
    accept_popup(driver)

    # Extraire les liens d'annonces
    liens = get_link_annonces_scroll(driver)

    # Boucle pour parcourir toutes les annonces
    for i, annonce in enumerate(liens):
        try:
            print(f"Traitement de l'annonce {i + 1}/{len(liens)} : {annonce}")

            # Ouvrir l'annonce
            driver.get(annonce)

            # Récupérer les informations de l'annonce
            data = get_info_annonce(driver, dict_data)
            data["url_annonce"] = annonce  # Ajouter l'URL de l'annonce
            
            # Sauvegarder les données dans la liste all_data
            all_data.append(data)
            #driver.close()
        except Exception as e:
            # En cas d'erreur, afficher un message et continuer
            print(f"Erreur lors du traitement de l'annonce {i + 1}/{len(liens)} : {annonce}")
            print(f"Erreur: {e}")
            continue

    # Fermer le driver
    driver.quit()

    # Enregistrer les données dans un dataframe pandas
    df = pd.DataFrame(all_data)

    # Exporter les données dans un fichier CSV
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"{len(df)} annonces enregistrées dans le fichier csv")
    print("Fin du scraping !")

    return df