'''
This file is for creating the navigation bar at the top of the Dash application.

'''

# Import necessary libraries
from dash import html, callback, Input, Output, State, dcc
import dash_bootstrap_components as dbc
from src.app.utils.images import logo_encoded, icon_encoded
#from src.app.components.login import login_info

NAVBAR = {
    "Accueil" : {"icon": "fa-solid fa-car fa-sm", "relative_path" : "/"},
    "Simulation" : {"icon": "fa-solid fa-calculator", "relative_path" : "/forecast"},
    "Reforecast" : {"icon": "fa-solid fa-sync", "relative_path": "/reforecast"},
    "About" : {"icon": "fa-solid fa-circle-info", "relative_path": "/about"},
}


def generate_nav_links(navbar_dict):
    """
    To generate the navbar items
    """
    nav_items = []
    for key, value in navbar_dict.items():
        nav_items.append(
            dbc.NavLink(
                [
                    html.I(className=value['icon'], style={'margin-right':"0.5rem"}),
                    key,
                ],
                href = value["relative_path"],
                className="nav-link",
            )
        )
    return nav_items


# Components for the navigation bar
navbar = dbc.Navbar(
    dbc.Container(
        [
            dcc.Link(
                [
                    html.Img(src=icon_encoded, width=30, height=30, className="d-inline-block align-top mr-2"), # Logo image
                    "Leasing - Valeur Résiduelle"
                ],
                href="/",  # Link to Home page
                className="navbar-brand"
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),  # Button to toggle the navbar for smaller screens
            dbc.Collapse(
                generate_nav_links(NAVBAR),
                id="navbar-collapse",
                is_open=False,  # Navbar is initially collapsed
                navbar=True 
            )
        ]

        ),
        color="light",
        dark=False,
        className="d-block d-md-none"
    )

# Callback to toggle the navbar collapse for smaller screens on button click 
@callback(
    Output("navbar-collapse", "is_open"),
    Input("navbar-toggler", "n_clicks"),
    State("navbar-collapse", "is_open")
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open