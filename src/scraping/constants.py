# Paramètres pour la fonction de scraping des annonces AutoHero
BASE_URL = "https://www.autohero.com/fr"
YEAR_MIN = 2017
KM_MAX = 100000
CSV_PATH = "../data/raw_data/autohero_scraping.csv"

DICT_DATA = {
    "modele" : ".desktopTitleContainer___2In8q span",
    "finition" : ".desktopLayout___3j5kK span.subtitleText___2wcYx",
    "prix" : "p.vehiclePrice___1uUmJ",
    "annee_mise_en_circulation" : "[data-qa-selector= 'motor-info-title-builtYear']",
    "kilometrage" : "[data-qa-selector= 'motor-info-title-mileage']",
    "carburant" : "[data-qa-selector= 'motor-info-title-undefined']",
    "transmission" : "[data-qa-selector= 'motor-info-title-gearType']",
    "puissance" : "[data-qa-selector= 'motor-info-title-power']",
    "nb_ancien_proprietaire" : "[data-qa-selector= 'motor-info-title-carPreownerCount']",
    "classe_vehicule" : "[data-qa-selector= 'feature-section-item-bodyType-body']",
    "nb_porte" : "[data-qa-selector= 'feature-section-item-doorCount-body']",
    "nb_place" : "[data-qa-selector= 'feature-section-item-seatCount-body']",
    "couleur" : "[data-qa-selector= 'feature-section-item-color-body']",
    "sellerie" : "[data-qa-selector= 'feature-section-item-upholstery-body']",
    "classe_emission" : "[data-qa-selector= 'feature-section-item-emissionStandard-body']",
    "emission_CO2" : "[data-qa-selector= 'feature-section-item-co2-body']",
    "crit_air" : "[data-qa-selector= 'feature-section-item-emissionSticker-body']",
    "usage_commerciale_anterieure" : "[data-qa-selector= 'feature-section-item-wasInCommercialUse-body']",
}

# Paramètres pour la fonction de scraping des prix neufs
BASE_URL_PRIX_NEUF = "https://www.caradisiac.com/fiches-techniques"

DICT_MARQUE ={'Abarth' : 'Abarth', 'Alfa' : 'Alfa Romeo', 'Audi' : 'Audi', 
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