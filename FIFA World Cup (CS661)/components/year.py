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


WCparti1 = html.Div(className="card-chart-container col-lg-6 md-1 sm-1",
                    children=[
                        html.Div(
                            className="card-chart",
                            children=[
                                html.H4(id = 'my-head6', children = "Continental Representation",
                                    className=" card-chart-container", style={"font-size": "1.5vw", "text-align" : "center", 'padding-top': '25px'}),
                                        dls.Roller(
                                            id="team-g5",
                                            children=[
                                    ], debounce=0
                                )
                            ]
                        )

                    ],
                    style={"min-height" :"20.25rem"}
                    )

@callback(
    Output("team-g5", "children"),
    Output("my-head6", "children"),
    Input("query-tour-select", "value"),
    #State("goals-df", "data"),
    #State("qualified-teams-df" , "data")
)

def update_year1(query_year):
    tour = pd.read_csv('data/tournaments_a.csv')

    df = tour[['tournament_name', 'North America', 'South America', 'Asia', 'Europe', 'Africa', 'Australia']]
    names = ['North America', 'South America', 'Asia', 'Europe', 'Africa', 'Australia']
    st = query_year + ' FIFA World Cup'
    df = df.query(f"tournament_name=='{st}' ")
    fig = px.bar(x=names, y=[df.iloc[0][i] for i in range(1,7)],
                labels={'x': "Continents", 'y': "Number of countries participated"}, text_auto=True)
    #fig.update_layout(title=f"Continental representation at {query_year}")
    fig.update_traces(textfont_size=16, textposition='outside', cliponaxis=False)
    

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

    ) , f'Continental representation at {query_year}'