import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import Dash, dcc, html
import plotly.express as px
import numpy as np
#import altair as alt
#from vega_datasets import data
import pandas as pd
import base64
from components.tours import WCtour, WCcomment2, WCwinner, WCrunner
from components.year import WCparti1
from components.teamcomp6 import WCteamchart6

country_content = html.Div([
        dbc.Row([
        WCcomment2,
        WCtour,
        WCwinner,
        WCrunner,
        ]),
        dbc.Row([
        WCparti1,
        WCteamchart6,
        ]),
], style={"padding-top": "40px"})