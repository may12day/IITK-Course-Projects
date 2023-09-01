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

WCteamchart5 = html.Div(className="card-chart-container col-lg-5 md-1 sm-1",
                    children=[
                        html.Div(
                            className="card-chart",
                            children=[
                                html.H4(id = 'my-head10', children = "Match Results",
                                    className=" card-chart-container", style={"font-size": "1.5vw", "text-align" : "center", 'padding-top': '25px'}),
                                        dls.Roller(
                                            id="team-g9",
                                            children=[
                                    ], debounce=0
                                )
                            ]
                        )

                    ],
                    style={"min-height" :"20.25rem"}
                    )

@callback(
    Output("team-g9", "children"),
    Output("my-head10", "children"),
    Input("query-team1", "value"),
    Input("query-team2", "value"),
    #State("goals-df", "data"),
    #State("qualified-teams-df" , "data")
)

def update_chart5(query_team1, query_team2):
    tvt = pd.read_csv('data/teamvsteam_a.csv')

    tvt = tvt.groupby(['team_name', 'opponent_name']).sum()
    tvt = tvt.filter(['team_name', 'opponent_name', 'win', 'lose', 'draw'])
    t1 = f'{query_team1}'
    t2 = f'{query_team2}'
    df = tvt.query(f"team_name=='{t1}'&opponent_name=='{t2}'")
    
    data = dict(labels=[f"{t1} won", f"{t2} won", "Match tied"],
            values=[df.iloc[0][0], df.iloc[0][1], df.iloc[0][2]])
    colors = ['#89EF08', '#e53935',  '#00A4E4']
    fig = px.pie(data, values='values', names='labels',)
    fig.update_traces(marker=dict(colors=colors))
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    

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

    ) , f"Match results of {query_team1} vs {query_team2}"