#################################################
# Entry Point for the application               #
# This Dash application aims to:

# - Enable users to add new simulations through a form and predict the residual value of cars based on various features (e.g., make, model, year, mileage)
# - Enable users to adapt the residual value by applying manual adjustments
# - Calculate the rent for leasing contracts based on predicted residual values
# - Provide an interactive dashboard for visualizing car data performance and trends
# - Allow users to compare different car models and their predicted residual values

#################################################

# Import useful packages & modules
import os
from dash import dcc, html, Dash, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from flask import Flask

############## APP INITIALIZATION ####################

CSV_PATH = "data/outil_data/sample_app_car_data.csv"
DB_URI = "car_data.db"

# Create a Flask app and configure it for SQLAlchemy
flask_app = Flask(__name__)
flask_app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///" + DB_URI
db = SQLAlchemy(flask_app)

# Pass the Flask app to Dash
app = Dash(__name__, server=flask_app, external_stylesheets=[dbc.themes.MORPH, dbc.icons.FONT_AWESOME])

class CarData(db.Model):
    __tablename__ = 'car_data'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    marque = db.Column(db.String(length=50), nullable=False)
    modele = db.Column(db.String(length=50), nullable=False)
    kilometrage = db.Column(db.Integer(), nullable =False)
    carburant = db.Column(db.String(length=20))
    transmission = db.Column(db.String)
    puissance = db.Column(db.Integer())
    nb_ancien_proprietaire = db.Column(db.String)
    classe_vehicule = db.Column(db.String)
    couleur = db.Column(db.String)
    sellerie = db.Column(db.String)
    emission_CO2 = db.Column(db.Integer())
    crit_air = db.Column(db.String)
    usage_commerciale_anterieure = db.Column(db.String)
    annee = db.Column(db.Integer())
    prix_neuf = db.Column(db.Float)
    mise_en_circulation = db.Column(db.String)

# Create the database and tables if they do not exist
with flask_app.app_context():
    db.create_all()

# Charge the database with data from the csv file if the db is empty
def load_data(csv_path, db_path):
    """
    Load data from a CSV file into an SQLite database.
    
    Parameters:
    csv_path (str): Path to the CSV file containing car data.
    db_path (str): Path to the SQLite database file.
    """
    # If the database file does not exist, create it
    if db.session.query(CarData).count() == 0:
        df = pd.read_csv(csv_path)
        print(df.head())
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
            ))
        db.session.commit()
        print(f"Data loaded from {csv_path} into database {db_path}.")
    else:
        print(f"Database {db_path} already contains data or CSV file does not exist.")

with flask_app.app_context():
    load_data(CSV_PATH, DB_URI)

################# APP LAYOUT ####################

app.layout = html.Div(html.H1('Test app'))





# if __name__ == "__main__":
#     app.run(debug=True)


