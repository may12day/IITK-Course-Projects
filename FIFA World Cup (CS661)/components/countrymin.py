import plotly.express as px
import pandas as pd
from dash import html, dcc
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_loading_spinners as dls
from dash import callback
import plotly.graph_objects as go

WCcountrymin = html.Div(className="card-chart-container col-lg-8 md-6 sm-12",
                    children=[
                        html.Div(
                            className="card-chart",
                            children=[
                                html.H4(id = 'my-head2', children = "Match Results",
                                    className=" card-chart-container", style={"font-size": "1.5vw", "text-align" : "center", 'padding-top': '25px'}),
                                        dls.Roller(
                                            id="team-g1",
                                            children=[
                                    ], debounce=0
                                )
                            ]
                        )

                    ],
                    style={"min-height" :"26.25rem"}
                    )


@callback(
    Output("team-g1", "children"),
    Output("my-head2", "children"),
    [Input("query-team-select", "value")]
    #State("goals-df", "data"),
    #State("qualified-teams-df" , "data")
)

def update_figures(query_team):
    cmg = pd.read_csv('data/country_minute_goals.csv')
    fig = px.histogram(cmg.query(f"team_name=='{query_team}'"), x="minute_regulation", y="goals",
                            nbins=120)
    #fig.update_layout(title=f"Goals scored by {query_team} per minute")
    fig.update_xaxes(title='Minute of regulation')
    fig.update_yaxes(title='Goals')
    
    

    return dcc.Graph(figure=fig
        .update_layout(paper_bgcolor="rgb(0,0,0,0)",
                    plot_bgcolor="rgb(0,0,0,0)",
                    legend=dict(
                    bgcolor="#fff"),
                    font_family="Public Sans, Amiri, Qatar2022, Poppins,",
                    
                    ),
        config={
        "displayModeBar": False},
        style={"height" : "25.875rem"}

    ), f"Goals scored by {query_team} per minute"