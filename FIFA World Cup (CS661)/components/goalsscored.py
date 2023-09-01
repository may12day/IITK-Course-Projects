import plotly.express as px
import pandas as pd
from dash import html, dcc
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_loading_spinners as dls
from dash import callback
import plotly.graph_objects as go

WCgoalsscored = html.Div(className="card-chart-container col-lg-15 md-1 sm-1",
                    children=[
                        html.Div(
                            className="card-chart",
                            children=[
                                html.H4(id = 'my-head4', children = "Goals scored by Country in WCs",
                                    className=" card-chart-container", style={"font-size": "1.5vw", "text-align" : "center", 'padding-top': '25px'}),
                                        dls.Roller(
                                            id="team-g3",
                                            children=[
                                    ], debounce=0
                                )
                            ]
                        )

                    ],
                    style={"min-height" :"20.25rem"}
                    )

@callback(
    Output("team-g3", "children"),
    Output("my-head4", "children"),
    [Input("query-team-select", "value")]
    #State("goals-df", "data"),
    #State("qualified-teams-df" , "data")
)

def update_figures(query_team):
    coun = pd.read_csv('data/country_cup_goals.csv')

    fig = px.line(coun.query(f"team_name=='{query_team}'"),
                    x='tournament_name', y='goals_scored',
                markers=True)
    #fig.update_layout(template='plotly_dark', title=f"Goals scored by {query_team} at each world cup")
    fig.update_xaxes(title='Tournament')
    fig.update_yaxes(title='Goals scored')
        
    

    return dcc.Graph(figure=fig
        .update_layout(paper_bgcolor="rgb(0,0,0,0)",
                    showlegend = False,
                    plot_bgcolor="rgb(0,0,0,0)",
                    legend=dict(
                    bgcolor="#fff"),
                    font_family="Public Sans, Amiri, Qatar2022, Poppins,",
                    
                    ),
        config={
        "displayModeBar": False},
        style={"height" : "25.875rem"}

    ) , f'Goals scored by {query_team} in WCs'