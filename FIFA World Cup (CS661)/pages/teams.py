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
from components.teams import WCTeamStats,WCimage,WCwinner,WCmatch
from components.teammin import WCteammin
from components.countrymin import WCcountrymin
from components.pie import WCpiechart
from components.goalsscored import WCgoalsscored
from components.teamvsteam import WCcomment,WCteam1,WCteam1image,WCteam2,WCteam2image,WCcomment1
from components.teamcomp import WCteamchart1
from components.teamcomp2 import WCteamchart2
from components.teamcomp3 import WCteamchart3
from components.teamcomp4 import WCteamchart4
from components.teamcomp5 import WCteamchart5


team_content = html.Div([
        dbc.Row([
        WCcomment1,
        WCTeamStats,
        WCimage,
        WCwinner,
        WCmatch,
        ]),
        dbc.Row([
        WCteammin,
        ]),
        dbc.Row([
        WCcountrymin,
        WCpiechart,
        WCgoalsscored,
        ]),
        dbc.Row([
        WCcomment,
        ]),
        dbc.Row([
        WCteam1,
        WCteam1image,
        WCteam2,
        WCteam2image,
        ]),
        dbc.Row([
        WCteamchart1,
        WCteamchart2,
        WCteamchart5,
        WCteamchart3,
        WCteamchart4,
        ]),
], style={"padding-top": "40px"})