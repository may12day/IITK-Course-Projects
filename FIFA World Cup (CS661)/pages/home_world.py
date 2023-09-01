from dash import html
import dash_bootstrap_components as dbc
from components.attach_map import mapping
from components.introcard import WCIntroCard

world_map = html.Div([
    dbc.Row([
            WCIntroCard,
            ]),
    dbc.Row([
            mapping,
            ]),
], style={"padding-top": "40px"})