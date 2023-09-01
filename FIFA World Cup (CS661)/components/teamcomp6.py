import plotly.express as px
import pandas as pd
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_loading_spinners as dls
from dash import callback
import plotly.graph_objects as go
import numpy as np

WCteamchart6 = html.Div(className="col-lg-6 col-md-3 col-sm-3 card-chart-container", children=[html.Div(className="card-chart", children=[
        html.Div(className="card-chart",style={'padding-top': '0px'}, children=[
            html.Div(className="card-chart-container", style={'margin-left': '35px'},children=[
                html.Div(className="card-m-3 pb-2",
                        children=[
                            dls.Roller(
                            children = [
                            html.H2(className="mt-4 mb-1 card-title",
                                    id="tour-scorer",
                                    children = 'Top Scorer: ',
                                    style={"font-size": "2vw", "text-align": "center"}),
                            html.H2(className="mt-4 mb-1 card-title",
                                    id="tour-best",
                                    children = 'Best Player: ',
                                    style={"font-size": "2vw", "text-align": "center"}),
                            html.H2(className="mt-4 mb-1 card-title",
                                    id="tour-manager",
                                    children = 'Winning Manager: ',
                                    style={"font-size": "2vw", "text-align": "center"})
                                    ]
                        ),
                        # html.H6(
                        #     className="mb-0 mt-4 card-title", children=["Winner"], style={"font-size": "1.5vw"}
                        # ),
                        ], style={"text-align": "center"}),
                        

            ])

        ])
    ], style={"min-height": "5rem"})]
    )

@callback(
    Output("tour-scorer", "children"),
    Output("tour-best", "children"),
    Output("tour-manager", "children"),
    Input("query-tour-select", "value"),
    #State("goals-df", "data"),
    #State("qualified-teams-df" , "data")
)

def update_tour_select11(value):
    teams_df = pd.read_csv("data/tournaments_p.csv")
    tour_scorer = f"Top Scorer: {teams_df.loc[teams_df.year==int(value) , 'Top_Scorer'].values[0]}"
    tour_best = f"Best Player: {teams_df.loc[teams_df.year==int(value) , 'Best_Player'].values[0]}"
    tour_manager = f"Winning Manager: {teams_df.loc[teams_df.year==int(value) , 'Winning_Coach'].values[0]}"

    return tour_scorer, tour_best, tour_manager