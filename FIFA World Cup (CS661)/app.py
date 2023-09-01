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
#from wordcloud import WordCloud
#import matplotlib.pyplot as plt
#from components.sidebar import sidebar
from pages.home import worldcup_page_content
from pages.home_world import world_map
from pages.teams import team_content
from pages.country import country_content


LOGO = "https://cdn-icons-png.flaticon.com/128/3179/3179858.png"

app = dash.Dash(
    external_stylesheets=[dbc.themes.MORPH, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True,
)

sidebar = html.Div(
    [
        html.Div(
            [
                # width: 3rem ensures the logo is the exact width of the
                # collapsed sidebar (accounting for padding)
                html.Img(src=LOGO, style={"width": "5rem"}),
                html.H3(["FIFA WC",html.Br(), "Analysis"], style={ "align-text": "center"}),
            ],
            className="sidebar-header",
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="fas fa-home me-2"), 
                    html.Span("Home")
                    ],
                    href="/",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-magnifying-glass-chart me-2"),
                        html.Span(" WC Stats"),
                    ],
                    href="/worldcup",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-people-group me-2"),
                        html.Span("Team Analysis"),
                    ],
                    href="/teams",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-calendar-days me-2"),
                        html.Span("WC Tournament"),
                    ],
                    href="/tour",
                    active="exact",
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)

content = html.Div(id="page-content", className="content")
#mapp = html.Iframe(id = 'map', srcDoc= open("./assets/map/test6.html", "r").read(), width= "1400", height= "500")
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


# set the content according to the current pathname
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page_content(pathname):
    if pathname == "/":
        return world_map
    elif pathname == "/worldcup":
        return worldcup_page_content
    elif pathname == "/teams":
        return team_content
    elif pathname == "/tour":
        return country_content
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


if __name__ == "__main__":
    app.run_server(debug=True)