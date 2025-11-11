from dash import html, callback, Input, Output, dcc, State
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
        # Toggle button for sidebar
        html.Div([
            dbc.Button(
                html.I(className="fas fa-bars"),
                id="sidebar-toggle",
                color="light",
                size="sm",
                className="position-absolute",
                style={"top": "10px", "right": "10px", "z-index": "1051"}
            )
        ]),
        html.Div(html.Img(src=logo_encoded, width=220, height="80px", className="img-fluid"), className="sidebar-header"),
        #html.Hr(),
        #html.Div(html.H2("Leasing Voiture", className="lead mb-0 text-center"), className="sidebar-header"),
        html.Div(html.H2("PoC Valeur Résiduelle", className="lead mb-0 text-left"), className="sidebar-header"),
        html.Hr(),
        dbc.Nav([], vertical=True, pills=True, id="sidebar-nav"),
        
        # Custom content that appears only when expanded
        html.Div([
            html.Hr(className="my-3"),
            html.Div([
                html.I(className="fas fa-info-circle me-2 text-primary"),
                html.Small("Version 1.0.0", className="text-muted")
            ], className="d-flex align-items-center mb-2"),
            # html.Div([
            #     html.I(className="fas fa-clock me-2 text-success"),
            #     html.Small("Dernière mise à jour: aujourd'hui", className="text-muted")
            # ], className="d-flex align-items-center mb-2"),
            # html.Div([
            #     dbc.Button([
            #         html.I(className="fas fa-cog me-1"),
            #         "Paramètres"
            #     ], color="outline-secondary", size="sm", className="w-100 mb-2"),
            #     dbc.Button([
            #         html.I(className="fas fa-question-circle me-1"),
            #         "Aide"
            #     ], color="outline-info", size="sm", className="w-100")
            # ])
        ], className="sidebar-extra-content", id="sidebar-extra"),
        
        #color_mode_switch,
    ],
    #style=SIDEBAR_STYLE,
    className="d-none d-md-block sidebar",
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

                dbc.NavLink(
                    [
                        html.I(className=page_value["icon"], 
                               style={'margin-right':"0.5rem"}
                               ),
                        html.Span(page_key),
                        ], 
                        className="d-flex align-items-center",
                    href = page_value['relative_path'],
                    active="exact",
                    # className = "nav-link active mb-1"
                    # if page_value['relative_path'] == url
                    # else "nav-link mb-1",
                )
                for page_key, page_value in NAVBAR.items()
            ]

# Toggle sidebar with button click
@callback(
    [Output("sidebar", "className"),
     Output("sidebar-toggle", "children"),
     Output("main-content", "className")],
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar", "className")],
    prevent_initial_call=True
)
def toggle_sidebar_state(n_clicks, current_class):
    """Toggle sidebar expanded/collapsed state"""
    if n_clicks:
        if "sidebar-expanded" in current_class:
            # Collapse sidebar
            new_class = current_class.replace("sidebar-expanded", "").strip()
            toggle_icon = html.I(className="fas fa-bars")
            content_class = "content"
        else:
            # Expand sidebar
            new_class = f"{current_class} sidebar-expanded"
            toggle_icon = html.I(className="fas fa-times")
            content_class = "content content-shifted"
        
        return new_class, toggle_icon, content_class
    
    return current_class, html.I(className="fas fa-bars"), "content"

