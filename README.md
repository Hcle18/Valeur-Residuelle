# Residual value modelling + Webapp pour l'implémentation  

Description: 
Le projet se déroulera en deux phases :
- Développement d’un modèle de machine learning permettant d’estimer le prix de vente des véhicules d’occasion et leur courbe de décote  
- Développement d’une application web en Dash/Streamlit intégrant le modèle retenu en phase 1 dans le calculateur de VR.  

# PoC - Phase 1: Données et modèles  
## 1. Scraping des données
-- Annonces de voitures d'occasion sur le site: autohero.com    
    - Voitures de l'année >= 2017  
    - Kilométrages <= 100 000 Km  
    Ces conditions ont été prises pour que
-- Fiches techniques auto pour récupérer le prix neuf: source https://www.caradisiac.com/fiches-techniques/  
On a besoin du prix neuf pour calculer le taux de décote (VR = taux de décote x prix neuf).  
Taux de décote = VR/Prix neuf

## 2. Data preprocessing


