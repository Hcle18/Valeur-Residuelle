#################################################
# Entry Point for the application               #
# This Dash application aims to:

# - Enable users to add new simulations through a form and predict the residual value of cars based on various features (e.g., make, model, year, mileage)
# - Enable users to adapt the residual value by applying manual adjustments
# - Calculate the rent for leasing contracts based on predicted residual values
# - Provide an interactive dashboard for visualizing car data performance and trends
# - Allow users to compare different car models and their predicted residual values

#################################################

# Import useful libraries
import os
from dash import dcc, html, Dash, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
#from flask_login import LoginManager


# Local imports
from src.app.data import db, create_server, CarData, load_data
from src.app.components import navbar
#from src.app.components.login import User, login_location

############## APP INITIALIZATION ####################

CSV_PATH = "data/outil_data/sample_app_car_data.csv"
DB_URI = "car_data.db"

# Create the server for the Dash app (that is from Flask)
server = create_server(DB_URI)
db.init_app(server)

# Pass the Flask app to Dash
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])

# Load the database with data from the csv file if the db is empty
load_data(server, CSV_PATH, 'car_data')

################# APP LAYOUT ###########################

# Login manager to login / logout users
# login_manager = LoginManager()
# login_manager.init_app(server)
# login_manager.login_view = "/login"
# @login_manager.user_loader
# def load_user(username):
#     return User(username)

def serve_layout():
    '''
    Function to serve the layout of the Dash app.
    This function is called when the app is run.
    '''
    return html.Div(
        [
            #login_location,
            navbar

        ]
    )
app.layout = serve_layout



################# APP RUNNING ###########################
if __name__ == '__main__':
    app.run(debug=True)