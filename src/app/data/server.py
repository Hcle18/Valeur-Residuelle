# Define the Flask application and its server to be used by Dash
# Initialize the db object to store the database connection sqlite

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import pandas as pd

db = SQLAlchemy()

def create_server(db_uri):
    server = Flask(__name__)
    server.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_uri
    return server