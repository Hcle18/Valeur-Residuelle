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
    html.P(
        [
            "Ce dashboard est créé dans le cadre d'un Proof of Concept sur la Valeur Résiduelle"
        ]
    ),
    html.H5("Attributions"),
    html.Ul(
        [
            html.Li([html.B("Datasets: "),]),
            html.Li([html.B("Etapes:")])
        ]
    )


]