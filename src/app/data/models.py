# Define the database model for db
from .create_server import db

class CarData(db.Model):
    __tablename__ = 'car_data'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    marque = db.Column(db.String(length=50), nullable=False)
    modele = db.Column(db.String(length=50), nullable=False)
    kilometrage = db.Column(db.Float(), nullable =False)
    carburant = db.Column(db.String(length=20))
    transmission = db.Column(db.String)
    puissance = db.Column(db.Float())
    nb_ancien_proprietaire = db.Column(db.String)
    classe_vehicule = db.Column(db.String)
    couleur = db.Column(db.String)
    sellerie = db.Column(db.String)
    emission_CO2 = db.Column(db.Float())
    crit_air = db.Column(db.String)
    usage_commerciale_anterieure = db.Column(db.String)
    annee = db.Column(db.Integer())
    prix_neuf = db.Column(db.Float())
    mise_en_circulation = db.Column(db.String)
    fin_du_contrat = db.Column(db.String)
    valeur_residuelle = db.Column(db.Float)