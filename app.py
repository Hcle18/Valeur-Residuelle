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
from dash import dcc, html, Dash, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

app = Dash(__name__, external_stylesheets=[dbc.themes.MORPH, dbc.icons.FONT_AWESOME])


app.layout = html.Div(html.H1('Test app'))




############## DATASET LOADING ####################



if __name__ == "__main__":
    app.run(debug=True)


