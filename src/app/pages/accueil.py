import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

# Local import
from src.app.data import db, CarData
from src.app.utils.title import TITLE

PAGE_TITLE = "Accueil"

# Register this page
dash.register_page(__name__, name=PAGE_TITLE,
                   title=f"{PAGE_TITLE} | {TITLE}", path="/", order=0)


# Add overview dashboard
# NB Car in current portfolio
# Total Residual Value at date

def get_car_count():
    """Get the number of cars in the database"""
    try:
        return db.session.query(CarData).count()
    except Exception as e:
        print(f"Error getting car count: {e}")
        return 0


def layout():
    return [
        dbc.Container([
            html.H3("Accueil", className="mb-3"),
            html.P(
                """
                Ce tableau de bord récapitule les informations clés à date concernant le portefeuille de leasing
                """
            ), # à remplacer date par la dernière date valeur dans la db
            
            # Highlight numbers
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    #html.I(className="fas fa-car fa-2x text-primary mb-2"),
                                    html.H2(
                                        html.Span([
                                            html.I(className="fas fa-car text-primary mr-3"),
                                            "-"
                                            ]),
                                        id="car-count-display",
                                        className="text-primary mb-0"
                                        ),
                                    html.P("Véhicules en portefeuille", className="text-muted mb-0"),
                                    html.Small("Nombre total de véhicules", className="text-muted")
                                ])
                            ])
                        ], className="border-0 shadow-sm", 
                        )
                    ], md=4, sm=12),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.H2(
                                        html.Span([
                                            html.I(className="fas fa-piggy-bank text-success mr-3"),
                                            "€ 2.5M"
                                            ]),
                                        id="card-vr-init",
                                        className="text-success mb-0"
                                        ),
                                    html.P("Valeur Résiduelle Totale", className="text-muted mb-0"),
                                    html.Small("Estimation initiale", className="text-muted")
                                ])
                            ])
                        ], className="border-0 shadow-sm")
                    ], md=4, sm=12
                    ),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.H2(
                                        html.Span([
                                            html.I(className="fas fa-chart-area text-info mr-3"),
                                            "€ 2.5M"
                                            ]),
                                        id="card-vr-reforecast",
                                        className="text-info mb-0"
                                        ),

                                    html.P("Valeur Résiduelle Totale", className="text-muted mb-0"),
                                    html.Small("Estimation reforecast", className="text-muted")
                                ])
                            ])
                        ], className="border-0 shadow-sm")
                    ], md=4, sm=12
                    ),    
                ],
                className="mb-4"
            ), 
            
            # Action Cards Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-calculator fa-3x text-primary mb-3"),
                            html.H4("Nouvelle Simulation", className="card-title"),
                            html.P("Calculez la valeur résiduelle pour un nouveau contrat de leasing"),
                            dbc.Button([
                                html.I(className="fas fa-arrow-right me-2"),
                                "Commencer"
                            ], color="primary", href="/forecast", external_link=True)
                        ], className="text-center")
                    ], className="h-100 border-0 shadow-sm")
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-sync fa-3x text-secondary mb-3"),
                            html.H4("Reforecast Portfolio", className="card-title"),
                            html.P("Mettez à jour les valeurs résiduelles du portefeuille existant"),
                            dbc.Button([
                                html.I(className="fas fa-arrow-right me-2"),
                                "Voir Portfolio"
                            ], color="secondary", href="/reforecast", external_link=True)
                        ], className="text-center")
                    ], className="h-100 border-0 shadow-sm")
                ], width=6),
            ], className="mb-4"
            ),

            # Refresh button for live updates
            dbc.Row([
                dbc.Col([
                    dbc.Button([
                        html.I(className="fas fa-refresh me-2"),
                        "Actualiser les données"
                    ], id="refresh-btn", color="outline-primary", size="sm")
                ], width=12, className="text-end")
            ])
        ], fluid=True)
    ]

# Callback to update car count
@callback(
    Output("car-count-display", "children"),
    [Input("refresh-btn", "n_clicks")],
    prevent_initial_call=False
)
def update_car_count(n_clicks):
    count = get_car_count()
    return html.Span([
        html.I(className="fas fa-car mr-3"),
        f"{count:,}"
    ])