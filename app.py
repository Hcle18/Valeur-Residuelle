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
import dash
from dash import dcc, html, Dash, callback, Input, Output, dash_table, clientside_callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
#from flask_login import LoginManager

# Local imports
from src.app.data import db, create_server, CarData, load_data
from src.app.components import navbar, sidebar
from src.app.config import CSV_PATH, DB_URI
#from src.app.components.login import User, login_location

############## APP INITIALIZATION ####################

# CSV_PATH = "data/outil_data/sample_app_car_data.csv"
# DB_URI = "car_data.db"

# Create the server for the Dash app (that is from Flask)
server = create_server(DB_URI)
db.init_app(server)

# Load the database with data from the csv file if the db is empty
load_data(server, CSV_PATH, 'car_data')

# Pass the Flask app to Dash
app = Dash(__name__, server=server, 
           external_stylesheets=[dbc.icons.FONT_AWESOME], 
           use_pages=True, pages_folder="src/app/pages",
           assets_folder="src/app/assets",
           title="Leasing - Valeur Résiduelle",
           suppress_callback_exceptions=True)



################# APP LAYOUT ###########################

# Login manager to login / logout users
# login_manager = LoginManager()
# login_manager.init_app(server)
# login_manager.login_view = "/login"
# @login_manager.user_loader
# def load_user(username):
#     return User(username)

content = html.Div(
    dbc.Spinner(
        dash.page_container,
        delay_show = 0,
        delay_hide=100,
        color="primary",
        spinner_class_name = "fixed-top",
        spinner_style={"margin-top": "100px"},
    ),
    className="content",
)

def serve_layout():
    '''
    Function to serve the layout of the Dash app.
    '''
    return html.Div(
        [
            #login_location,
            sidebar.location,
            sidebar.sidebar,
            navbar,
            content
        ]
    )
app.layout = serve_layout


# clientside_callback(
#     """
#     (switchOn) => {
#        document.documentElement.setAttribute("data-bs-theme", switchOn ? "light" : "dark");
#        return window.dash_clientside.no_update
#     }
#     """,
#     Output("switch", "id"),
#     Input("switch", "value"),
# )

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Car Residual Value Dashboard</title>
        <meta name="description" content="Predict and analyze car residual values">
        {%css%}
        <link rel="icon" href="src/app/assets/NexiaLog_RVB_Original.png">
    </head>
    <body>
        {%app_entry%}
        {%config%}
        {%scripts%}
        {%renderer%}
    </body>
</html>

<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:ital,wght@0,300..900;1,300..900&display=swap');
body {
    font-family: 'Figtree', sans-serif;
}
</style>
"""

################# APP RUNNING ###########################
if __name__ == '__main__':
    app.run(debug=True)