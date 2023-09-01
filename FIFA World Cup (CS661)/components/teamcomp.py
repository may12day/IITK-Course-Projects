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

WCteamchart1 = html.Div(className="card-chart-container col-lg-15 md-1 sm-1",
                    children=[
                        html.Div(
                            className="card-chart",
                            children=[
                                html.H4(id = 'my-head5', children = "V/S",
                                    className=" card-chart-container", style={"font-size": "1.5vw", "text-align" : "center", 'padding-top': '25px'}),
                                        dls.Roller(
                                            id="team-g4",
                                            children=[
                                    ], debounce=0
                                )
                            ]
                        )

                    ],
                    style={"min-height" :"20.25rem"}
                    )

@callback(
    Output("team-g4", "children"),
    Output("my-head5", "children"),
    Input("query-team1", "value"),
    Input("query-team2", "value"),
    #State("goals-df", "data"),
    #State("qualified-teams-df" , "data")
)

def update_chart1(query_team1, query_team2):
    cmg = pd.read_csv('data/country_minute_goals.csv')
    cmg1 = cmg.query(f"team_name=='{query_team1}'")
    cmg2 = cmg.query(f"team_name=='{query_team2}'")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cmg1["minute_regulation"], y=cmg1["goals"], name=query_team1, marker_color='rgb(26, 118, 255)'))
    fig.add_trace(go.Scatter(x=cmg2["minute_regulation"], y=cmg2["goals"], name=query_team2, marker_color='Orange'))
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.7)
    fig.update_xaxes(title="Minute of Regulation")
    fig.update_yaxes(title="Goals Scored")
    

    return dcc.Graph(figure=fig
        .update_layout(paper_bgcolor="rgb(0,0,0,0)",
                    showlegend = True,
                    plot_bgcolor="rgb(0,0,0,0)",
                    legend=dict(bgcolor="#fff"),
                    font_family="Public Sans, Amiri, Qatar2022, Poppins,",
                    #showlegend= True,
                    #legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    #legend_itemsizing ='trace',
                    ),
        config={
        "displayModeBar": False},
        style={"height" : "25.875rem"}

    ) , f'{query_team1} and {query_team2} Goals Minute by Minute '