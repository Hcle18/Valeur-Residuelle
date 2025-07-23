import dash
from dash import html, dcc, callback, Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import uuid
import joblib
import pandas as pd
import json
from datetime import datetime, date
import plotly.graph_objects as go

# Local import
from src.app.data import db, CarData
from src.app.utils.title import TITLE
from src.industrialisation.modelapply import VrModelApplication
from src.industrialisation import constants as c
from src.app.components.DropdownManager import dropdown_manager
from src.industrialisation.modelapply import CarVrData

PAGE_TITLE = "Simulation"

# Register this page
dash.register_page(__name__, name=PAGE_TITLE,
                   title=f"{PAGE_TITLE} | {TITLE}", path="/forecast", order=1)

# Create dropdown options using data/outil_data/sample_app_car_data.csv


# def load_dropdown_options():
#     """Load marque and modele options from joblib embedding files"""
#     try:
#         # Load embedding files (joblib)
#         embedding_marque = joblib.load(c.EMBEDDING_MARQUE_JOBLIB)
#         embedding_modele = joblib.load(c.EMBEDDING_MODEL_JOBLIB)

#         # Extract unique values for marque
#         if isinstance(embedding_marque, pd.DataFrame):
#             marque_list = sorted(embedding_marque["marque"].unique())
#         elif isinstance(embedding_marque, dict):
#             marque_list = sorted(embedding_marque.keys())
#         else:
#             marque_list = sorted(list(embedding_marque.index) if hasattr(embedding_marque, 'index') else []) 

#         # Extract unique values for modele
#         if isinstance(embedding_modele, pd.DataFrame):
#             modele_list = sorted(embedding_modele["modele"].unique())
#         elif isinstance(embedding_modele, dict):
#             modele_list = sorted(embedding_modele.keys())
#         else:
#             modele_list = sorted(list(embedding_modele.index) if hasattr(embedding_modele, 'index') else []) 

#         marque_options = [{"label": marque, "value": marque} for marque in marque_list]
#         modele_options = [{"label": modele, "value": modele} for modele in modele_list]
        
#         return marque_options, modele_options
    
#     except Exception as e:
#         print(f"Error loading embedding files: {e}")
#         # Fallback options
#         marque_options = [
#             {"label": "Toyota", "value": "Toyota"},
#             {"label": "Renault", "value": "Renault"},
#             {"label": "Peugeot", "value": "Peugeot"},
#             {"label": "Volkswagen", "value": "Volkswagen"}
#         ]
#         modele_options = [
#             {"label": "Corolla", "value": "Corolla"},
#             {"label": "Clio", "value": "Clio"},
#             {"label": "208", "value": "208"},
#             {"label": "Golf", "value": "Golf"}
#         ]
#         return marque_options, modele_options

# MARQUE_OPTIONS, MODELE_OPTIONS = load_dropdown_options()

def create_simulation_form():
    """Create a form for a single simulation"""
    simulation_id = str(uuid.uuid4())

    return html.Div([
        # Header avec bouton de collapse
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    dbc.Button([
                        html.I(id={"type": "collapse_icon", "index": simulation_id}, 
                               className="fas fa-chevron-down mr-2"),
                        html.H5("Nouvelle Simulation", className="mb-0 d-inline")
                    ], 
                    id={"type": "collapse_toggle", "index": simulation_id},
                    color="link", 
                    className="p-0 text-decoration-none text-start w-100"),
                    
                    dbc.Button("✕", 
                              color="light", 
                              size="sm", 
                              id={"type": "remove_sim", "index": simulation_id},
                              className="ms-auto")
                ], className="d-flex justify-content-between align-items-center w-100")
            ], className="py-2"),
        ], className="mb-2"),
        
        # Contenu collapsible
        dbc.Collapse([
            dbc.Card([
                dbc.CardBody([
                    # Vehicle Information Section
                    html.H6("Information Véhicule", className="text-primary mb-3"),
                    dbc.Row([
                        # Marque
                        dropdown_manager.create_dropdown(
                            "marque",
                            component_id={"type": "marque", "index": simulation_id},
                            md_width=4, sm_width=12
                        ),
                        # Modèle - initialement désactivé, réactivé en fonction de marque choisie
                        dbc.Col([
                            dbc.Label("Modèle"),
                            dcc.Dropdown(
                                id={"type": "modele", "index": simulation_id}, 
                                placeholder="Sélectionnez un modèle",
                                searchable=True,
                                clearable=True,
                                disabled=True)
                        ], md=4, sm=12),
                        # Année avec plage automatique
                        dropdown_manager.create_numeric_input(
                            "annee",
                            component_id={"type": "annee", "index": simulation_id},
                            md_width=4, sm_width=12
                        )
                    ], className="mb-3"),

                    dbc.Row([
                        # Emission CO2
                        dbc.Col([
                            dbc.Label("Emission CO2"),
                            dbc.Input(id={"type": "emission_CO2", "index": simulation_id}, type="number", placeholder="120")
                        ], md=4, sm=12),
                        # Puissance (CV)
                        dbc.Col([
                            dbc.Label("Puissance (CV)"),
                            dbc.Input(id={"type": "puissance", "index": simulation_id}, type="number", placeholder="130")
                        ], md=4, sm=12),
                        # Type de transmission
                        dropdown_manager.create_dropdown(
                            "transmission", 
                            component_id={"type": "transmission", "index": simulation_id}, 
                            md_width=4, sm_width=12
                        ),
                    ], className="mb-3"),

                    dbc.Row([
                        # Carburant
                        dropdown_manager.create_dropdown(
                            "carburant",
                            component_id={"type": "carburant", "index": simulation_id},
                            md_width=3, sm_width=12
                        ),
                        # Classe véhicule
                        dropdown_manager.create_dropdown(
                            "classe_vehicule",
                            component_id={"type": "classe_vehicule", "index": simulation_id},
                            md_width=3, sm_width=12
                        ),       
                        # Couleur
                        dropdown_manager.create_dropdown(
                            "couleur",
                            component_id={"type": "couleur", "index": simulation_id},
                            md_width=3, sm_width=12
                        ),
                        # Sellerie
                        dropdown_manager.create_dropdown(
                            "sellerie",
                            component_id={"type": "sellerie", "index": simulation_id},
                            md_width=3, sm_width=12
                        ), 
                    ], className="mb-3"),

                    dbc.Row([
                        # Crit'Air
                        dropdown_manager.create_dropdown(
                            "crit_air",
                            component_id={"type": "crit_air", "index": simulation_id},
                            md_width=4, sm_width=12
                        ),
                        # Usage commercial antérieur
                        dropdown_manager.create_dropdown(
                            "usage_commerciale_anterieure",
                            component_id={"type": "usage_commerciale_anterieure", "index": simulation_id},
                            md_width=4, sm_width=12
                        ),
                        # Date de mise en circulation
                        dbc.Col([
                            dbc.Label("Mise en circulation"),
                            dcc.DatePickerSingle(
                                id={"type": "mise_en_circulation", "index": simulation_id},
                                date=date.today(),
                                display_format='DD/MM/YYYY',
                                placeholder='Sélectionnez une date',
                                first_day_of_week=1,
                                month_format='MMMM YYYY',
                                show_outside_days=True,
                                stay_open_on_select=False,
                                calendar_orientation='horizontal',
                                clearable=True,
                                style={'width': '100%'},
                                min_date_allowed=date(date.today().year - 5, 1, 1),
                                max_date_allowed=date.today()
                            )
                        ], md=4, sm=12)
                    ], className="mb-4"),

                    # Section Contrat de Leasing
                    html.H6("Informations Contrat", className="text-success mb-3"),
                    dbc.Row([
                        # Prix à neuf
                        dbc.Col([
                            dbc.Label("Prix neuf (€)"),
                            dbc.Input(
                                id={"type": "prix_neuf", "index": simulation_id},
                                type="number",
                                placeholder="35000",
                                value=35000
                            )
                        ], md=4, sm=12),

                        # Apport initial slider
                        dbc.Col([
                            dbc.Label("Apport initial (€)"),
                            html.Div([
                                dcc.Slider(
                                    id={"type": "downpayment_slider", "index": simulation_id},
                                    min=0,
                                    max=20000,
                                    step=500,
                                    value=5000,
                                    marks={
                                        0: {'label': '0€', 'style': {'color': '#666'}},
                                        5000: {'label': '5k€', 'style': {'color': '#666'}},
                                        10000: {'label': '10k€', 'style': {'color': '#666'}},
                                        15000: {'label': '15k€', 'style': {'color': '#666'}},
                                        20000: {'label': '20k€', 'style': {'color': '#666'}}
                                    },
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                html.Div(
                                    id={"type": "downpayment_display", "index": simulation_id},
                                    children="5 000 €",
                                    className="text-center mt-2 fw-bold text-success"
                                )
                            ])
                        ], md=4, sm=12),

                        # Kilométrage théorique
                        dbc.Col([
                            dbc.Label("Kilométrage théorique"),
                            html.Div([
                                dcc.Slider(
                                    id={"type": "kilometrage_slider", "index": simulation_id},
                                    min=5000,
                                    max=50000,
                                    step=1000,
                                    value=30000,
                                    marks={
                                        5000: {'label': '5k', 'style': {'color': '#666'}},
                                        15000: {'label': '15k', 'style': {'color': '#666'}},
                                        25000: {'label': '25k', 'style': {'color': '#666'}},
                                        35000: {'label': '35k', 'style': {'color': '#666'}},
                                        50000: {'label': '50k', 'style': {'color': '#666'}}
                                    },
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                html.Div(
                                    id={"type": "kilometrage_display", "index": simulation_id},
                                    children="30 000 km",
                                    className="text-center mt-2 fw-bold text-primary"
                                )
                            ])
                        ], md=4, sm=12),
                    ], className="mb-3"),

                    # Date du contrat
                    dbc.Row([
                        # Taux d'intérêt slider
                        dbc.Col([
                            dbc.Label("Taux d'intérêt (%)"),
                            html.Div([
                                dcc.Slider(
                                    id={"type": "interest_rate_slider", "index": simulation_id},
                                    min=0.5,
                                    max=8.0,
                                    step=0.1,
                                    value=3.2,
                                    marks={
                                        0.5: {'label': '0.5%', 'style': {'color': '#666'}},
                                        2.0: {'label': '2%', 'style': {'color': '#666'}},
                                        4.0: {'label': '4%', 'style': {'color': '#666'}},
                                        6.0: {'label': '6%', 'style': {'color': '#666'}},
                                        8.0: {'label': '8%', 'style': {'color': '#666'}}
                                    },
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                html.Div(
                                    id={"type": "interest_rate_display", "index": simulation_id},
                                    children="3.2%",
                                    className="text-center mt-2 fw-bold text-info"
                                )
                            ])
                        ], md=6, sm=12),

                        dbc.Col([
                            dbc.Label("Fin du contrat"),
                            dcc.DatePickerSingle(
                                id={"type": "fin_contrat", "index": simulation_id},
                                date=date.today(),
                                display_format='DD/MM/YYYY',
                                placeholder='Sélectionnez une date',
                                first_day_of_week=1,
                                month_format='MMMM YYYY',
                                show_outside_days=True,
                                stay_open_on_select=False,
                                calendar_orientation='horizontal',
                                clearable=True,
                                style={'width': '100%'},
                                min_date_allowed=date.today()
                            )
                        ], md=6, sm=12),
                    ], className="mb-4"),

                    # Calculate Button
                    dbc.Row([
                        dbc.Col([
                            dbc.Button([
                                html.I(className="fas fa-calculator mr-2"),
                                "Commencer le calcul"
                            ], id={"type": "calculate", "index": simulation_id},
                                color="primary", size="lg", className="w-100")
                        ], width=12)
                    ], className="mb-3"),

                    # Results Section
                    html.Div(id={"type": "results", "index": simulation_id})
                ])
            ], className="border-0")  # Retire la bordure de la carte interne
        ], 
        id={"type": "collapse_content", "index": simulation_id},
        is_open=True  # Ouvert par défaut
        )
    ], className="mb-4", id={"type": "simulation-card", "index": simulation_id})

def layout():
    return dbc.Container([
        # Header
        html.Div([
            html.H2([
                html.I(className="fas fa-calculator mr-2"),
                "Simulation Valeur Résiduelle"
            ], className="text-primary"),
            html.P("Créez et comparez des simulations", className="text-muted mb-1"),
        ], className="mb-4"),

        # Button
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button([
                        html.I(className="fas fa-plus mr-2"),
                        "Ajouter une simulation"
                    ], id="add-simulation", color="success"),
                    dbc.Button([
                        html.I(className="fas fa-refresh mr-2"),
                        "Actualiser les données"
                    ], id="refresh-data", color="outline-secondary")
                ])
            ], className="mb-4")
        ]),

        # Simulations container
        html.Div(id="simulations-container", children=[]),

        # Summary section
        html.Div(id="summary-section", className="mt-4")
    ], 
    fluid=True)


###########################
###### Add callback #######
###########################

@callback(
    Output("simulations-container", "children"),
    [Input("add-simulation", "n_clicks"),
     Input({"type": "remove_sim", "index": ALL}, "n_clicks")],
    [State("simulations-container", "children")],
    prevent_initial_call=True
)
def manage_simulations(add_clicks, remove_clicks, current_simulations):
    """Manage adding and removing simulation forms"""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return current_simulations
    
    # Parse trigger
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Add new simulation
    if "add-simulation" in trigger_id:
        if add_clicks:
            new_simulation = create_simulation_form()
            current_simulations.append(new_simulation)
            return current_simulations
    
    # Remove simulation
    if "remove_sim" in trigger_id:
        # Parse the simulation ID from the trigger
        trigger_dict = json.loads(trigger_id)
        simulation_to_remove = trigger_dict["index"]
        
        # Filter out the simulation to remove
        filtered_simulations = []
        for sim in current_simulations:
            sim_id = sim["props"]["id"]["index"]
            if sim_id != simulation_to_remove:
                filtered_simulations.append(sim)
        
        return filtered_simulations
    
    return current_simulations


@callback(
    [Output({"type": "collapse_content", "index": MATCH}, "is_open"),
     Output({"type": "collapse_icon", "index": MATCH}, "className")],
    [Input({"type": "collapse_toggle", "index": MATCH}, "n_clicks")],
    [State({"type": "collapse_content", "index": MATCH}, "is_open")],
    prevent_initial_call=True
)
def toggle_collapse(n_clicks, is_open):
    """Toggle collapse state of simulation cards"""
    if n_clicks:
        new_state = not is_open
        icon_class = "fas fa-chevron-up mr-2" if new_state else "fas fa-chevron-down mr-2"
        return new_state, icon_class
    return is_open, "fas fa-chevron-down mr-2"


@callback(
    Output({"type": "downpayment_display", "index": MATCH}, "children"),
    [Input({"type": "downpayment_slider", "index": MATCH}, "value")]
)
def update_downpayment_display(value):
    """Update downpayment display based on slider value"""
    if value is not None:
        return f"{value:,} €".replace(",", " ")
    return "0 €"


@callback(
    Output({"type": "kilometrage_display", "index": MATCH}, "children"),
    [Input({"type": "kilometrage_slider", "index": MATCH}, "value")]
)
def update_kilometrage_display(value):
    """Update kilometrage display based on slider value"""
    if value is not None:
        return f"{value:,} km".replace(",", " ")
    return "0 km"


@callback(
    Output({"type": "interest_rate_display", "index": MATCH}, "children"),
    [Input({"type": "interest_rate_slider", "index": MATCH}, "value")]
)
def update_interest_rate_display(value):
    """Update interest rate display based on slider value"""
    if value is not None:
        return f"{value}%"
    return "0%"


@callback(
    [Output({"type": "modele", "index": MATCH}, "options"),
     Output({"type": "modele", "index": MATCH}, "disabled"),
     Output({"type": "modele", "index": MATCH}, "value")],
    [Input({"type": "marque", "index": MATCH}, "value")]
)
def update_modele_options(selected_marque):
    """Update modele dropdown options based on selected marque"""
    if selected_marque:
        try:
            modele_options = dropdown_manager.get_modeles_by_marque(selected_marque)
            return modele_options, False, None
        except Exception as e:
            print(f"Error updating modele options: {e}")
            return [], True, None
    return [], True, None


@callback(
    Output("refresh-data", "children"),
    [Input("refresh-data", "n_clicks")],
    prevent_initial_call=True
)
def refresh_dropdown_data(n_clicks):
    """Refresh dropdown data from database"""
    if n_clicks:
        try:
            dropdown_manager.refresh_all_options()
            return [
                html.I(className="fas fa-check mr-2"),
                "Données actualisées"
            ]
        except Exception as e:
            print(f"Error refreshing data: {e}")
            return [
                html.I(className="fas fa-exclamation-triangle mr-2"),
                "Erreur lors de l'actualisation"
            ]
    return [
        html.I(className="fas fa-refresh mr-2"),
        "Actualiser les données"
    ]


@callback(
    Output({"type": "results", "index": MATCH}, "children"),
    [Input({"type": "calculate", "index": MATCH}, "n_clicks")],
    [State({"type": "marque", "index": MATCH}, "value"),
     State({"type": "modele", "index": MATCH}, "value"),
     State({"type": "annee", "index": MATCH}, "value"),
     State({"type": "emission_CO2", "index": MATCH}, "value"),
     State({"type": "puissance", "index": MATCH}, "value"),
     State({"type": "transmission", "index": MATCH}, "value"),
     State({"type": "carburant", "index": MATCH}, "value"),
     State({"type": "classe_vehicule", "index": MATCH}, "value"),
     State({"type": "couleur", "index": MATCH}, "value"),
     State({"type": "sellerie", "index": MATCH}, "value"),
     State({"type": "crit_air", "index": MATCH}, "value"),
     State({"type": "usage_commerciale_anterieure", "index": MATCH}, "value"),
     State({"type": "mise_en_circulation", "index": MATCH}, "date"),
     State({"type": "prix_neuf", "index": MATCH}, "value"),
     State({"type": "downpayment_slider", "index": MATCH}, "value"),
     State({"type": "kilometrage_slider", "index": MATCH}, "value"),
     State({"type": "interest_rate_slider", "index": MATCH}, "value"),
     State({"type": "fin_contrat", "index": MATCH}, "date")],
    prevent_initial_call=True
)
def calculate_residual_value(n_clicks, marque, modele, annee, emission_co2, puissance, 
                           transmission, carburant, classe_vehicule, couleur, sellerie,
                           crit_air, usage_commerciale, mise_en_circulation, prix_neuf,
                           downpayment, kilometrage, interest_rate, fin_contrat):
    """Calculate residual value based on input parameters"""
    
    if not n_clicks:
        return html.Div()
    
    # Validate required fields
    required_fields = {
        "Marque": marque,
        "Modèle": modele,
        "Année": annee,
        "Prix neuf": prix_neuf,
        "Fin du contrat": fin_contrat
    }
    
    missing_fields = [field for field, value in required_fields.items() if not value]
    
    if missing_fields:
        return dbc.Alert([
            html.H5("Champs requis manquants", className="alert-heading"),
            html.P(f"Veuillez remplir: {', '.join(missing_fields)}")
        ], color="warning", className="mt-3")
    
    try:
        # Create CarVrData instance
 
        car_data = CarVrData(
            marque=marque,
            modele=modele,
            kilometrage=kilometrage,
            carburant=carburant,
            transmission=transmission,
            puissance=puissance,
            nb_ancien_proprietaire="1",  # Default value
            classe_vehicule=classe_vehicule,
            couleur=couleur,
            sellerie=sellerie,
            emission_CO2=emission_co2,
            crit_air=crit_air,
            usage_commerciale_anterieure=usage_commerciale,
            annee=annee,
            prix_neuf=prix_neuf,
            mise_en_circulation=mise_en_circulation,
            fin_du_contrat=fin_contrat
        )
        
        # Create model application instance and get predictions
        model_app = VrModelApplication(car_data)
        
        # Get single prediction
        prediction = model_app.predict()
        
        # Calculate residual value
        residual_value = prix_neuf * prediction[0] if prediction else 0
        residual_percentage = prediction[0] * 100 if prediction else 0
        
        # Get prediction curve for chart
        try:
            model_app_curve = VrModelApplication(car_data)
            curve_predictions = model_app_curve.predict_curve()
            
            # Create plotly chart

            
            # Calculate months from mise_en_circulation to fin_contrat
            from datetime import datetime
            if isinstance(mise_en_circulation, str):
                start_date = datetime.fromisoformat(mise_en_circulation)
            else:
                start_date = datetime.combine(mise_en_circulation, datetime.min.time())
                
            if isinstance(fin_contrat, str):
                end_date = datetime.fromisoformat(fin_contrat)
            else:
                end_date = datetime.combine(fin_contrat, datetime.min.time())
            
            total_months = ((end_date.year - start_date.year) * 12 + 
                          end_date.month - start_date.month)
            
            months = list(range(1, min(len(curve_predictions) + 1, total_months + 1)))
            values = [prix_neuf * pred for pred in curve_predictions[:len(months)]]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=months,
                y=values,
                mode='lines+markers',
                name='Valeur résiduelle',
                line=dict(color='#007bff', width=3),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Évolution de la valeur résiduelle",
                xaxis_title="Mois",
                yaxis_title="Valeur (€)",
                template="plotly_white",
                height=400
            )
            
            chart_component = dcc.Graph(figure=fig)
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            chart_component = html.Div("Graphique non disponible", className="text-muted")
        
        # Calculate financial metrics
        monthly_depreciation = (prix_neuf - residual_value) / total_months if total_months > 0 else 0
        total_cost = prix_neuf - downpayment + (monthly_depreciation * total_months)
        
        return html.Div([
            dbc.Alert([
                html.H4([
                    html.I(className="fas fa-check-circle mr-2"),
                    "Calcul terminé"
                ], className="alert-heading text-success"),
                
                dbc.Row([
                    dbc.Col([
                        html.H5("Valeur résiduelle prédite"),
                        html.H3(f"{residual_value:,.0f} €".replace(",", " "), 
                               className="text-primary"),
                        html.P(f"({residual_percentage:.1f}% du prix neuf)", 
                              className="text-muted")
                    ], md=4),
                    
                    dbc.Col([
                        html.H5("Dépréciation totale"),
                        html.H3(f"{prix_neuf - residual_value:,.0f} €".replace(",", " "), 
                               className="text-warning"),
                        html.P(f"({100 - residual_percentage:.1f}% du prix neuf)", 
                              className="text-muted")
                    ], md=4),
                    
                    dbc.Col([
                        html.H5("Dépréciation mensuelle"),
                        html.H3(f"{monthly_depreciation:,.0f} €".replace(",", " "), 
                               className="text-info"),
                        html.P(f"Sur {total_months} mois", className="text-muted")
                    ], md=4)
                ], className="mb-3"),
                
                # Chart
                chart_component,
                
                # Summary table
                html.H5("Résumé financier", className="mt-4 mb-3"),
                dbc.Table([
                    html.Tbody([
                        html.Tr([
                            html.Td("Prix d'achat neuf", className="fw-bold"),
                            html.Td(f"{prix_neuf:,.0f} €".replace(",", " "))
                        ]),
                        html.Tr([
                            html.Td("Apport initial", className="fw-bold"),
                            html.Td(f"{downpayment:,.0f} €".replace(",", " "))
                        ]),
                        html.Tr([
                            html.Td("Valeur résiduelle", className="fw-bold"),
                            html.Td(f"{residual_value:,.0f} €".replace(",", " "))
                        ]),
                        html.Tr([
                            html.Td("Kilométrage prévu", className="fw-bold"),
                            html.Td(f"{kilometrage:,.0f} km".replace(",", " "))
                        ]),
                        html.Tr([
                            html.Td("Durée du contrat", className="fw-bold"),
                            html.Td(f"{total_months} mois")
                        ]),
                        html.Tr([
                            html.Td("Taux d'intérêt", className="fw-bold"),
                            html.Td(f"{interest_rate}%")
                        ])
                    ])
                ], striped=True, hover=True)
                
            ], color="success", className="mt-3")
        ])
        
    except Exception as e:
        return dbc.Alert([
            html.H5("Erreur de calcul", className="alert-heading"),
            html.P(f"Une erreur est survenue lors du calcul: {str(e)}")
        ], color="danger", className="mt-3")


@callback(
    Output("summary-section", "children"),
    [Input("simulations-container", "children")],
    prevent_initial_call=True
)
def update_summary(simulations):
    """Update summary section when simulations change"""
    if not simulations or len(simulations) <= 1:
        return html.Div()
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-chart-bar mr-2"),
                "Comparaison des simulations"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.P(f"Vous avez {len(simulations)} simulation(s) en cours.", 
                  className="text-muted"),
            dbc.Button([
                html.I(className="fas fa-download mr-2"),
                "Exporter les résultats"
            ], color="outline-primary", size="sm")
        ])
    ], className="mt-4")
