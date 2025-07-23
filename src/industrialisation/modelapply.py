############### Model Application Class ###############
# This script is designed to apply a pre-trained model to new data.
# It includes loading the model, applying a transform pipeline, and making predictions. 

# Import necessary libraries
import os
import pandas as pd
import joblib
from dataclasses import dataclass
import plotly.graph_objects as go
from typing import Union, List

# Local imports
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

    def __init__(self, car_data: Union[CarVrData, List[CarVrData], pd.DataFrame]):

        # Initialize the data with the provided parameters
        if isinstance(car_data, pd.DataFrame):
            # Direct DataFrame input
            self.data_df = car_data
            # Convert DataFrame rows to CarVrData objects for comptability
            self.data_list = []
            for _, row in car_data.iterrows():
                car_obj = CarVrData(**row.to_dict())
                self.data_list.append(car_obj)

        elif isinstance(car_data, list):
            self.data_list = car_data
            # Create a combined DataFrame from all cars
            df_list = [car.to_dataframe() for car in car_data]
            self.data_df = pd.concat(df_list, ignore_index=True)
        else:
            self.data_list = [car_data]
            # Create a single DataFrame if there is a single data entry
            self.data_df = car_data.to_dataframe()

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

        # Prepare the data
        self.data_df= self._prepared_data(self.data_df, self.embedding_marque, self.embedding_model)

    @staticmethod
    def _prepared_data(df: pd.DataFrame, embedding_marque: pd.DataFrame, embedding_model: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the DataFrame and Calculate the features
        """
        # Parse the date columns with the correct format
        #df['mise_en_circulation'] = pd.to_datetime(df['mise_en_circulation'], format="%d/%m/%Y", errors='coerce')
        #df['fin_du_contrat'] = pd.to_datetime(df['fin_du_contrat'], format="%d/%m/%Y", errors='coerce')

        df['age_months'] = (pd.to_datetime(df['fin_du_contrat'], format="%d/%m/%Y", errors='coerce') - pd.to_datetime(df['mise_en_circulation'], format="%d/%m/%Y", errors='coerce')).dt.days // 30
        df['age_months'] = df['age_months'].apply(lambda x: round(x, 1) if pd.notnull(x) else None)
        df["km_per_month"] = df["kilometrage"] / df["age_months"]
        df["marque"] = df['marque'].str.upper()
        df["modele"] = df['modele'].str.upper()

        # Apply the embedding vectors for 'marque' and 'modele'
        df = pd.merge(df, embedding_marque, on="marque", how="left")
        df = pd.merge(df, embedding_model, on="modele", how="left")

        return df

    def predict(self) -> Union[float, List[float]]:
        """
        Make predictions for all cars in the dataset
        Returns:
            Union[float, List[float]]: Single prediction or list of predictions
        """

        # Apply the transformation pipeline
        #print(self.data_df.head())
        df_transform = self.transformer.transform(self.data_df)
        predictions = self.model.predict(df_transform)
        
        # Return single value if only one car, otherwise returs list
        if len(predictions) == 1:
            return predictions[0]
        return predictions.tolist()
    
    def predict_curve(self, car_index: int=0):
        """
        Create a prediction curve by age_months, for a specific car in the list
        """

        # For each line of the DataFrame, create new rows for each month up to age_months
        # This will create a DataFrame with multiple rows for each car, one for each month
        # Keep all existing columns, but change the 'age_months' column to reflect the month of the contract
        # Also change the 'kilometrage' column to reflect the expected kilometrage for that month

        if car_index >= len(self.data_list):
            raise ValueError(f"Car index {car_index} out of range")
        
        # Get the specific car data
        car_data = self.data_list[car_index]
        car_df = car_data.to_dataframe()

        # Prepare the dataframe
        car_df = self._prepared_data(car_df, self.embedding_marque, self.embedding_model)

        age_months = car_df['age_months'].iloc[0]
        for month in range(1, int(age_months)):
            new_row = car_df.iloc[0].copy()
            new_row['age_months'] = month
            new_row['kilometrage'] = new_row['km_per_month'] * month
            car_df = pd.concat([car_df, pd.DataFrame([new_row])], ignore_index=True)

        # Sort the DataFrame by 'age_months' to ensure the curve is in the correct order
        car_df = car_df.sort_values(by='age_months')

        #print(car_df.head())  # Debugging: print the transformed DataFrame

        # Apply the transformation pipeline and predict
        df_transform_curve = self.transformer.transform(car_df)
        predictions_curve = self.model.predict(df_transform_curve)

        # Add predictions to the DataFrame
        car_df['prediction_vr_ratio'] = predictions_curve
        car_df['prediction_vr'] = round(car_df['prediction_vr_ratio'] * car_df['prix_neuf'], 0)

        # Add a line for age_months = 0 with the initial price
        initial_row = car_df.iloc[0].copy()
        initial_row['age_months'] = 0
        initial_row['kilometrage'] = 0
        initial_row['prediction_vr_ratio'] = 1.0
        initial_row['prediction_vr'] = initial_row['prix_neuf']
        
        car_df = pd.concat([pd.DataFrame([initial_row]), car_df], ignore_index=True)

        # Create a figure for the prediction vr curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=car_df["age_months"],
            y=car_df["prediction_vr"],
            mode='lines+markers',
            name='Predicted VR',
            line=dict(color='blue', width=2),
            marker=dict(size=5, color='blue', symbol='circle')
        ))
 
        print(f"Predictions VR ratio: {predictions_curve}")
        print(f"Predictions VR: {car_df['prediction_vr']}")

        return fig
    
    def predict_all_curves(self):
        """
        Create prediction curves for all cars
        Returns:
            go.Figure: Plotly figure with curves for all cars
        """
        fig = go.Figure()
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, car_data in enumerate(self.data_list):
            # Prepare car data using static method
            car_df = self._prepared_data(car_data.to_dataframe(), self.embedding_marque, self.embedding_model)

            # Create curve data
            age_months = car_df['age_months'].iloc[0]
            curve_rows = []
            
            for month in range(1, int(age_months)):
                new_row = car_df.iloc[0].copy()
                new_row['age_months'] = month
                new_row['kilometrage'] = new_row['km_per_month'] * month
                curve_rows.append(new_row)

            if curve_rows:
                curve_df = pd.DataFrame(curve_rows)
                car_df = pd.concat([car_df, curve_df], ignore_index=True)

            car_df = car_df.sort_values(by='age_months')

            # Predict
            df_transform_curve = self.transformer.transform(car_df)
            predictions_curve = self.model.predict(df_transform_curve)
            car_df['prediction_vr_ratio'] = predictions_curve
            car_df['prediction_vr'] = round(car_df['prediction_vr_ratio'] * car_df['prix_neuf'], 0)

            # Add initial point
            initial_row = car_df.iloc[0].copy()
            initial_row['age_months'] = 0
            initial_row['kilometrage'] = 0
            initial_row['prediction_vr_ratio'] = 1.0
            initial_row['prediction_vr'] = initial_row['prix_neuf']
            car_df = pd.concat([pd.DataFrame([initial_row]), car_df], ignore_index=True)

            # Add trace to figure
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=car_df["age_months"],
                y=car_df["prediction_vr_ratio"],
                mode='lines+markers',
                name=f'{car_data.marque} {car_data.modele}',
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color, symbol='circle')
            ))

        fig.update_layout(
            title="Courbes de Valeur Résiduelle pour Toutes les Voitures",
            xaxis_title="Âge (mois)",
            yaxis_title="Valeur Résiduelle (%)",
            hovermode='x unified'
        )

        return fig
    
if __name__ == "__main__":

    # Example usage with single car
    car_data_single = CarVrData(
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
    
    # Example usage with multiple cars
    car_data_list = [
        CarVrData(
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
        ),
        CarVrData(
            marque="RENAULT",
            modele="RENAULT CLIO",
            kilometrage=35000,
            carburant="Essence",
            transmission="Automatique",
            puissance=110,
            nb_ancien_proprietaire="1",
            classe_vehicule="Citadine",
            couleur="Blanc",
            sellerie="Tissu",
            emission_CO2=115,
            crit_air="1",
            usage_commerciale_anterieure="Non",
            annee=2021,
            prix_neuf=18000,
            mise_en_circulation="01/06/2021",
            fin_du_contrat="01/06/2025"
        )
    ]
    
    # Test with single car
    # print("=== Test avec une seule voiture ===")
    # vr_app_single = VrModelApplication(car_data=car_data_single)
    # prediction_single = vr_app_single.predict()
    # print(f"Prédiction pour une voiture: {prediction_single}")
    
    # # Test with multiple cars
    # print("\n=== Test avec plusieurs voitures ===")
    vr_app_multiple = VrModelApplication(car_data=car_data_list)
    # predictions_multiple = vr_app_multiple.predict()
    # print(f"Prédictions pour plusieurs voitures: {predictions_multiple}")
    
    # # Test curves
    curve_single = vr_app_multiple.predict_curve(car_index=0)
    #all_curves = vr_app_multiple.predict_all_curves()