############### Model Application Class ###############
# This script is designed to apply a pre-trained model to new data.
# It includes loading the model, applying a transform pipeline, and making predictions. 

# Import necessary libraries
import os
import pandas as pd
import joblib
from dataclasses import dataclass
from src.industrialisation import constants as c

# Class for data: list of input features needed for the model
@dataclass
class CarVrData:
    marque: str
    modele: str
    kilometrage: float
    carburant: str
    transmission: str
    puissance: float
    nb_ancien_proprietaire: str
    classe_vehicule: str
    couleur: str
    sellerie: str
    emission_CO2: float
    crit_air: str
    usage_commerciale_anterieure: str
    annee: int
    prix_neuf: float
    mise_en_circulation: str
    fin_du_contrat: str

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the data class to a pandas DataFrame.
        Returns:
            pd.DataFrame: DataFrame containing the data.
        """
        return pd.DataFrame([self.__dict__])

# Class for model application
class VrModelApplication:
    """
    Handles the application of the residual value prediction model.
    Loads necessary models and transformers, processes input data,
    and generates predictions.
    """

    def __init__(self, marque: str, modele: str, kilometrage: float, carburant: str,
                 transmission: str, puissance: float, nb_ancien_proprietaire: str,
                 classe_vehicule: str, couleur: str, sellerie: str, emission_CO2: float,
                 crit_air: str, usage_commerciale_anterieure: str, annee: int,
                 prix_neuf: float, mise_en_circulation: str, fin_du_contrat: str):
        
        # Initialize the data with the provided parameters
        self.data = CarVrData(
            marque=marque,
            modele=modele,
            kilometrage=kilometrage,
            carburant=carburant,
            transmission=transmission,
            puissance=puissance,
            nb_ancien_proprietaire=nb_ancien_proprietaire,
            classe_vehicule=classe_vehicule,
            couleur=couleur,
            sellerie=sellerie,
            emission_CO2=emission_CO2,
            crit_air=crit_air,
            usage_commerciale_anterieure=usage_commerciale_anterieure,
            annee=annee,
            prix_neuf=prix_neuf,
            mise_en_circulation=mise_en_circulation,
            fin_du_contrat=fin_du_contrat
        )

        # Ensure the paths to the joblib files are correct
        if not all([c.TRANSFORM_JOBLIB, 
                    c.EMBEDDING_MODEL_JOBLIB, 
                    c.EMBEDDING_MARQUE_JOBLIB, 
                    c.PRETRAINED_MODEL_JOBLIB]):
            raise ValueError("One or more model paths are not defined")
        
        # Load the pre-trained model and other necessary components 
        self.model = joblib.load(c.PRETRAINED_MODEL_JOBLIB)
        self.embedding_model = joblib.load(c.EMBEDDING_MODEL_JOBLIB)
        self.embedding_marque = joblib.load(c.EMBEDDING_MARQUE_JOBLIB)
        self.transformer = joblib.load(c.TRANSFORM_JOBLIB)

    def predict(self):
        # Calculate the features and prepare the DataFrame
        self.data.df = self.data.to_dataframe()
        self.data.df['age_months'] = (pd.to_datetime(self.data.df['fin_du_contrat']) - pd.to_datetime(self.data.df['mise_en_circulation'])).dt.days // 30
        self.data.df['age_months'] = self.data.df['age_months'].apply(lambda x: round(x, 1) if pd.notnull(x) else None)
        self.data.df["km_per_month"] = self.data.df["kilometrage"] / self.data.df["age_months"]
        self.data.df["marque"] = self.data.df['marque'].str.upper()
        self.data.df["modele"] = self.data.df['modele'].str.upper()

        # Apply the embedding vectors for 'marque' and 'modele'
        self.data.df = pd.merge(self.data.df, self.embedding_marque, on="marque", how="left")
        self.data.df = pd.merge(self.data.df, self.embedding_model, on="modele", how="left")

        print(self.data.df.head())  # Debugging: print the transformed DataFrame

        # Apply the transformation pipeline
        self.data.df = self.transformer.transform(self.data.df)

        predictions = self.model.predict(self.data.df)
        print(f"Predictions VR ratio: {predictions}")
        return predictions
    
if __name__ == "__main__":
    # Example usage
    vr_app = VrModelApplication(
        marque="FORD",
        modele="FORD FIESTA",
        kilometrage=40000,
        carburant="Essence",
        transmission="Manuelle",
        puissance=130,
        nb_ancien_proprietaire="1",
        classe_vehicule="Berline",
        couleur="Rouge",
        sellerie="Tissu",
        emission_CO2=120,
        crit_air="1",
        usage_commerciale_anterieure="Non",
        annee=2020,
        prix_neuf=20000,
        mise_en_circulation="01/01/2020",
        fin_du_contrat="01/01/2025"
    )
    
    predictions = vr_app.predict()