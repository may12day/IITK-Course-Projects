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

WCteamchart3 = html.Div(className="card-chart-container col-lg-6 md-1 sm-1",
                    children=[
                        html.Div(
                            className="card-chart",
                            children=[
                                html.H4(id = 'my-head8', children = "Match Results",
                                    className=" card-chart-container", style={"font-size": "1.5vw", "text-align" : "center", 'padding-top': '25px'}),
                                        dls.Roller(
                                            id="team-g7",
                                            children=[
                                    ], debounce=0
                                )
                            ]
                        )

                    ],
                    style={"min-height" :"20.25rem"}
                    )

@callback(
    Output("team-g7", "children"),
    Output("my-head8", "children"),
    Input("query-team1", "value"),
    Input("query-team2", "value"),
    #State("goals-df", "data"),
    #State("qualified-teams-df" , "data")
)

def update_chart3(query_team1, query_team2):
    cmb = pd.read_csv('data/country_minute_bookings_a.csv')
    cmb['yellow_card'] = cmb['yellow_card'] + cmb['second_yellow_card']
    cmb['red_card'] = cmb['red_card'] + cmb['second_yellow_card']

    # select team here
    t1 = f'{query_team1}'
    t2 = f'{query_team2}'

    ## YELLOW CARDS
    booking = "yellow_card"
    cmb1 = cmb.query(f"team_name=='{t1}'")
    cmb2 = cmb.query(f"team_name=='{t2}'")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cmb1["minute_regulation"], y=cmb1[booking], name=t1))
    fig.add_trace(go.Scatter(x=cmb2["minute_regulation"], y=cmb2[booking], name=t2, marker_color='Orange'))
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.7)
    fig.update_xaxes(title="Minute of Regulation")
    fig.update_yaxes(title="Yellow / Red Cards")

    ## RED CARDS
    # booking = "red_card"
    # cmb1 = cmb.query(f"team_name=='{t1}'")
    # cmb2 = cmb.query(f"team_name=='{t2}'")
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=cmb1["minute_regulation"], y=cmb1[booking], name=t1))
    # fig.add_trace(go.Scatter(x=cmb2["minute_regulation"], y=cmb2[booking], name=t2, marker_color='Orange'))
    # fig.update_layout(title=f"Minutewise red cards (including second yellow cards) {t1} vs {t2}", barmode='overlay', template='plotly_dark')
    # fig.update_traces(opacity=0.7)
    

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

    ) , f"Minutewise Yellow cards (including second yellow cards) {query_team1} vs {query_team2}"