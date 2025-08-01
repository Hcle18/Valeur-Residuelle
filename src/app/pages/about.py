import dash
from dash import html
from src.app.utils.title import TITLE

PAGE_TITLE = "A propos"

dash.register_page(
    __name__,
    name=PAGE_TITLE,
    title=f"{PAGE_TITLE} | {TITLE}",
)

layout = [
    html.H3("A Propos"),
    html.Hr(),
    html.P(
        "Cette page fournit des informations sur l'application, son objectif et son utilisation."
    ),
    html.P(
        [
            "Ce dashboard est créé dans le cadre d'un Proof of Concept sur la Valeur Résiduelle des véhicules.",
            " Il permet de simuler la valeur résiduelle d'un véhicule en fonction de divers paramètres.",
            " Les données utilisées proviennent de sources ouvertes et sont à titre d'exemple.",
        ]
    ),
    html.H5("Attributions"),
    html.Ul(
        [
            html.Li([html.B("Datasets: "), " Liste des jeux de données utilisés pour l'entraînement des modèles."]),
            html.Li([html.B("Etapes:"), " Description des étapes du processus de simulation."])
        ]
    ),
    # Add links to repositories or documentation
    html.H5("Liens Utiles"),
    html.P(
        [
            "Documentation: ",
            html.A("Documentation", href="https://nexialog.com/"),
        ]
    ),

    html.H5("Github"),
    html.P(
        [
            "Le code source de ce projet est disponible sur ",
            html.A("GitHub", href="https://github.com/Hcle18/Valeur-Residuelle",)
        ]
    )
]