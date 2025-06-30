from .server import db
from .models import CarData
import os
import pandas as pd

def load_data(server, csv_path, table_name='car_data'):

    """    Load data from a CSV file into the database if the database is empty.

    Args:
        server (Flask): The Flask server instance.
        csv_path (str): The path to the CSV file containing the data.
        table_name (str): The name of the database table to load the data into.
    """
    #db.init_app(server)
            
    with server.app_context():
        try:
            db.create_all()
            print('Database tables created successfully.')

            # load data from the CSV file into the database if the database is empty
            print(f"Number of records: {db.session.query(CarData).count()}")
            if db.session.query(CarData).count() == 0:
                # Check if the CSV file exists
                if not os.path.exists(csv_path):
                    print(f"CSV file {csv_path} does not exist.")
                    return
                
                # Read the csv file
                df = pd.read_csv(csv_path)
                print(df.head())

                # Load the csv file into the database
                for _, row in df.iterrows():
                    db.session.add(CarData(
                         marque=row['marque'],
                         modele=row['modele'],
                         kilometrage=row['kilometrage'],
                         carburant=row['carburant'],
                         transmission=row['transmission'],
                         puissance=row['puissance'],
                         nb_ancien_proprietaire=row['nb_ancien_proprietaire'],
                         classe_vehicule=row['classe_vehicule'],
                         couleur=row['couleur'],
                         sellerie=row['sellerie'],
                         emission_CO2=row['emission_CO2'],
                         crit_air=row['crit_air'],
                         usage_commerciale_anterieure=row['usage_commerciale_anterieure'],
                         annee=row['annee'],
                         prix_neuf=row['prix_neuf'],
                         mise_en_circulation=row['Mise_en_circulation'],
                         fin_du_contrat=row['Fin_du_contrat']
                ))
                db.session.commit() 
                print(f"Data loaded from {csv_path} into database table {table_name}.")

            else:
                print(f"Database table {table_name} is not empty.")
        except Exception as e:
            print(f"An error occurred while loading data: {e}")