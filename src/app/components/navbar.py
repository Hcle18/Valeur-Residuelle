'''
This file is for creating the navigation bar at the top of the Dash application.

'''

# Import necessary libraries
from dash import html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from src.app.utils.images import logo_encoded
#from src.app.components.login import login_info

# Components for the navigation bar
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=logo_encoded, height="90px")), # Logo image
                        #dbc.Col(dbc.NavbarBrand("Nexialog Consulting", className="ms-2"))
                    ],
                    align="center",  # Aligns items in the center
                    className="g-0 me-5",  # Removes default gutter spacing
                ),
                href="https://nexialog.com",  # Link to Home page
                style={'textDecoration': 'none'} # Removes underline from the link
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),  # Button to toggle the navbar for smaller screens
            dbc.Collapse(
                dbc.Nav(
                    [
                        dbc.NavItem(
                            dbc.NavLink(
                                'Home',
                                href="/"
                            )
                        ),
                        dbc.NavItem(
                            dbc.NavLink(
                                'About',
                                href="/about"
                            )
                        ),
                        dbc.NavItem(
                            dbc.NavLink(
                                'Contact',
                                href="/contact"
                            )
                        ),
                        #html.Div(login_info)
                    ]
                ),
                id="navbar-collapse",
                is_open=False,  # Navbar is initially collapsed
                navbar=True 
            )
        ]

    ),
    color="light",
    dark=False
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