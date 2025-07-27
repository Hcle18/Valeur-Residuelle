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

def create_simulation_form_with_id(simulation_id):
    """Create a form for a single simulation"""
    #simulation_id = str(uuid.uuid4())

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
                        # Émission CO2
                        dbc.Col([
                            dbc.Label("Émission CO2"),
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
                                date=date(date.today().year + 3, date.today().month, date.today().day),
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

def restore_simulation_form(sim_data):
    """Restore a simulation form from stored data"""
    simulation_id = sim_data["id"]
    values = sim_data.get("values", {})

    # Créer le formulaire avec les valeurs restaurées
    form = create_simulation_form_with_id (simulation_id)

    return form


def layout():
    return dbc.Container([

        # Stockage des données de simulation
        dcc.Store(id="simulations-store", storage_type="session"),
        dcc.Store(id="simulation-counter", data=0, storage_type="session"),

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

# Callback pour charger les simulations au chargement de la page
@callback(
        Output("simulations-container", "children", allow_duplicate=False),
        Input("simulations-store", "data"),
        prevent_initial_call = False
)
def load_simulations_from_store(stored_data):
    """Charge les simulations depuis le stockage"""
    if not stored_data:
        return []
    
    simulations = []
    for sim_data in stored_data:
        simulation_form = restore_simulation_form(sim_data)
        simulations.append(simulation_form)

    return simulations


# Creating or closing a form for VR simulation
@callback(
    [Output("simulations-container", "children", allow_duplicate=True),
     Output("simulations-store", "data", allow_duplicate=True),
     Output("simulation-counter", "data", allow_duplicate=True)
     ],
    [Input("add-simulation", "n_clicks"),
     Input({"type": "remove_sim", "index": ALL}, "n_clicks")],
    [State("simulations-container", "children"),
     State("simulations-store", "data"),
     State("simulation-counter", "data")
     ],
    prevent_initial_call=True
)
def manage_simulations(add_clicks, remove_clicks, current_simulations, stored_data, counter):
    """Manage adding and removing simulation forms"""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return current_simulations, stored_data or [], counter or 0
    
    # Initialiser stored_data si None
    if stored_data is None:
        stored_data = []

    # Parse trigger
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Add new simulation
    if "add-simulation" in trigger_id:
        if add_clicks:
            new_counter = (counter or 0) + 1
            simulation_id = f"sim_{new_counter}"

            new_simulation = create_simulation_form_with_id(simulation_id)
            current_simulations.append(new_simulation)

            # Ajouter aux données stockées dans dcc.Store
            stored_data.append(
                {
                    "id": simulation_id,
                    "values": {},
                    "results": None
                }
            )
            return current_simulations, stored_data, new_counter
    
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

        # Filtrer les données stockées
        stored_data = [s for s in stored_data if s["id"] != simulation_to_remove]

        return filtered_simulations, stored_data, counter
    
    return current_simulations, stored_data, counter


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
            modele_options = dropdown_manager.get_filtered_modeles(selected_marque)
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


# CALLBACK PRINCIPAL - Calcul avec sauvegarde intégrée

@callback(
    Output({"type": "results", "index": MATCH}, "children"),

    [Input({"type": "calculate", "index": MATCH}, "n_clicks")],
    [
        State({"type": "marque", "index": MATCH}, "value"),
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
        State({"type": "fin_contrat", "index": MATCH}, "date"),
        State("simulations-store", "data"),
        State({"type": "calculate", "index": MATCH}, "id")
     ],
    prevent_initial_call=True
)
def calculate_residual_value(n_clicks, marque, modele, annee, emission_co2, puissance, 
                           transmission, carburant, classe_vehicule, couleur, sellerie,
                           crit_air, usage_commerciale, mise_en_circulation, prix_neuf,
                           downpayment, kilometrage, interest_rate, fin_contrat, 
                           stored_data, button_id):
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
        alert_result= dbc.Alert([
            html.H5("Champs requis manquants", className="alert-heading"),
            html.P(f"Veuillez remplir: {', '.join(missing_fields)}")
        ], color="warning", className="mt-3")
        return alert_result
    
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
        residual_value = prix_neuf * prediction if prediction else 0
        residual_percentage = prediction * 100 if prediction else 0
        
        # Calculate months from mise_en_circulation to fin_contrat
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
            
        # Get prediction curve for chart
        try:
            #model_app_curve = VrModelApplication(car_data)
            #curve_predictions = model_app_curve.predict_curve()
            curve_predictions = model_app.predict_curve()

            # Create a figure for the prediction vr curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=curve_predictions["age_months"],
                y=curve_predictions["prediction_vr"],
                mode='lines+markers',
                name='Predicted VR',
                line=dict(color='#007bff', width=3),
                marker=dict(size=6),
                text=curve_predictions["prediction_vr"].apply(lambda x: f"VR: {x:,.0f} €"),
                textposition="top center",
                hoverinfo="text",
                hovertemplate="Valeur résiduelle: %{y:,.0f} €<br>Mois: %{x}<extra></extra>"
            ))

            fig.update_layout(
                title="Évolution de la valeur résiduelle",
                title_x=0.5,
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
        
        # Créer les résultats
        results_html= html.Div([
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
        
        return results_html
        
    except Exception as e:
        error_result = dbc.Alert([
            html.H5("Erreur de calcul", className="alert-heading"),
            html.P(f"Une erreur est survenue lors du calcul: {str(e)}")
        ], color="danger", className="mt-3")
        return error_result
    
# CALLBACK SÉPARÉ pour la sauvegarde dans le store
# @callback(
#     Output("simulations-store", "data", allow_duplicate=True),
#     [Input({"type": "calculate", "index": ALL}, "n_clicks")],
#     [
#         State({"type": "marque", "index": ALL}, "value"),
#         State({"type": "modele", "index": ALL}, "value"),
#         State({"type": "annee", "index": ALL}, "value"),
#         State({"type": "emission_CO2", "index": ALL}, "value"),
#         State({"type": "puissance", "index": ALL}, "value"),
#         State({"type": "transmission", "index": ALL}, "value"),
#         State({"type": "carburant", "index": ALL}, "value"),
#         State({"type": "classe_vehicule", "index": ALL}, "value"),
#         State({"type": "couleur", "index": ALL}, "value"),
#         State({"type": "sellerie", "index": ALL}, "value"),
#         State({"type": "crit_air", "index": ALL}, "value"),
#         State({"type": "usage_commerciale_anterieure", "index": ALL}, "value"),
#         State({"type": "mise_en_circulation", "index": ALL}, "date"),
#         State({"type": "prix_neuf", "index": ALL}, "value"),
#         State({"type": "downpayment_slider", "index": ALL}, "value"),
#         State({"type": "kilometrage_slider", "index": ALL}, "value"),
#         State({"type": "interest_rate_slider", "index": ALL}, "value"),
#         State({"type": "fin_contrat", "index": ALL}, "date"),
#         State("simulations-store", "data")
#      ],
#     prevent_initial_call=True
# )
# def save_simulation_data(n_clicks_list, marque_list, modele_list, annee_list, emission_co2_list, 
#                         puissance_list, transmission_list, carburant_list, classe_vehicule_list, 
#                         couleur_list, sellerie_list, crit_air_list, usage_commerciale_list,
#                         mise_en_circulation_list, prix_neuf_list, downpayment_list, 
#                         kilometrage_list, interest_rate_list, fin_contrat_list, stored_data):
#     """Save simulation data when calculate button is clicked"""
    
#     ctx = dash.callback_context
#     if not ctx.triggered or not any(n_clicks_list):
#         return stored_data or []
    
#     if stored_data is None:
#         stored_data = []
    
#     # Mettre à jour toutes les simulations
#     for i in range(len(marque_list)):
#         simulation_id = f"sim_{i+1}"  # Ajustez selon votre logique d'ID
        
#         # Trouver ou créer l'entrée pour cette simulation
#         sim_found = False
#         for sim_data in stored_data:
#             if sim_data["id"] == simulation_id:
#                 sim_data["values"].update({
#                     "marque": marque_list[i] if i < len(marque_list) else None,
#                     "modele": modele_list[i] if i < len(modele_list) else None,
#                     "annee": annee_list[i] if i < len(annee_list) else None,
#                     "emission_CO2": emission_co2_list[i] if i < len(emission_co2_list) else None,
#                     "puissance": puissance_list[i] if i < len(puissance_list) else None,
#                     "transmission": transmission_list[i] if i < len(transmission_list) else None,
#                     "carburant": carburant_list[i] if i < len(carburant_list) else None,
#                     "classe_vehicule": classe_vehicule_list[i] if i < len(classe_vehicule_list) else None,
#                     "couleur": couleur_list[i] if i < len(couleur_list) else None,
#                     "sellerie": sellerie_list[i] if i < len(sellerie_list) else None,
#                     "crit_air": crit_air_list[i] if i < len(crit_air_list) else None,
#                     "usage_commerciale_anterieure": usage_commerciale_list[i] if i < len(usage_commerciale_list) else None,
#                     "mise_en_circulation": mise_en_circulation_list[i] if i < len(mise_en_circulation_list) else None,
#                     "prix_neuf": prix_neuf_list[i] if i < len(prix_neuf_list) else None,
#                     "downpayment": downpayment_list[i] if i < len(downpayment_list) else None,
#                     "kilometrage": kilometrage_list[i] if i < len(kilometrage_list) else None,
#                     "interest_rate": interest_rate_list[i] if i < len(interest_rate_list) else None,
#                     "fin_contrat": fin_contrat_list[i] if i < len(fin_contrat_list) else None
#                 })
#                 sim_found = True
#                 break
        
#         # Si la simulation n'existe pas dans le store, la créer
#         if not sim_found:
#             stored_data.append({
#                 "id": simulation_id,
#                 "values": {
#                     "marque": marque_list[i] if i < len(marque_list) else None,
#                     "modele": modele_list[i] if i < len(modele_list) else None,
#                     "annee": annee_list[i] if i < len(annee_list) else None,
#                     "emission_CO2": emission_co2_list[i] if i < len(emission_co2_list) else None,
#                     "puissance": puissance_list[i] if i < len(puissance_list) else None,
#                     "transmission": transmission_list[i] if i < len(transmission_list) else None,
#                     "carburant": carburant_list[i] if i < len(carburant_list) else None,
#                     "classe_vehicule": classe_vehicule_list[i] if i < len(classe_vehicule_list) else None,
#                     "couleur": couleur_list[i] if i < len(couleur_list) else None,
#                     "sellerie": sellerie_list[i] if i < len(sellerie_list) else None,
#                     "crit_air": crit_air_list[i] if i < len(crit_air_list) else None,
#                     "usage_commerciale_anterieure": usage_commerciale_list[i] if i < len(usage_commerciale_list) else None,
#                     "mise_en_circulation": mise_en_circulation_list[i] if i < len(mise_en_circulation_list) else None,
#                     "prix_neuf": prix_neuf_list[i] if i < len(prix_neuf_list) else None,
#                     "downpayment": downpayment_list[i] if i < len(downpayment_list) else None,
#                     "kilometrage": kilometrage_list[i] if i < len(kilometrage_list) else None,
#                     "interest_rate": interest_rate_list[i] if i < len(interest_rate_list) else None,
#                     "fin_contrat": fin_contrat_list[i] if i < len(fin_contrat_list) else None
#                 },
#                 "results": None
#             })
    
#     return stored_data

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
