from dash import html, callback, Input, Output, dcc
import dash_bootstrap_components as dbc

# Local
from src.app.utils.images import icon_encoded, logo_encoded
from src.app.components.navbar import NAVBAR

# color_mode_switch = html.Div([
#     html.I(className="fas fa-moon text-primary me-2"),
#     dbc.Switch(
#         id="switch",
#         value=True,
#         className="d-inline-block custom-switch",
#         persistence=True,
#         #style={"transform": "scale(0.8)"}
#     ),
#     html.I(className="fas fa-sun text-primary ms-2"),
# ], className="d-flex align-items-center mt-1 ms-4")

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [
        html.Img(src=logo_encoded, width=220, height="80px"),
        html.Hr(),
        html.Div(
            [
                html.P("Leasing Voiture", className="lead mb-0 text-center"),
                html.P("Valeur Résiduelle", className="lead mb-0 text-center"),
            ]
        ),
        html.Hr(),
        dbc.Nav([], vertical=True, pills=True, id="sidebar-nav"),
        #color_mode_switch,
    ],
    style=SIDEBAR_STYLE,
    className="d-none d-md-block",
    id="sidebar",
)

location = dcc.Location(id="url")


# Update side bar based on url
@callback(
    Output("sidebar-nav", "children"),
    Input("url", "pathname")
)
def update_sidebar(url:str) -> list:
    return [
        html.Div(
            [
                dcc.Link(
                    html.Div(
                        [
                            html.I(className=page_value["icon"], style={'margin-right':"0.5rem"}),
                            page_key,
                        ], 
                        className="d-flex align-items-center",
                    ),
                    href = page_value['relative_path'],
                    className = "nav-link active mb-1"
                    if page_value['relative_path'] == url
                    else "nav-link mb-1",
                )
                for page_key, page_value in NAVBAR.items()
            ]
        )
    ]

