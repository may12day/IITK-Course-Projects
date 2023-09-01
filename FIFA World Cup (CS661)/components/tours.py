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

df_tour = pd.read_csv("data/tournaments.csv")


WCcomment2 = html.Div(className="card-chart-container col-lg-15 ",
                    children=[
                        html.Div(
                            className="card-chart",
                            style={"font-size": "1.5vw", "text-align" : "center",},
                            children=[
                                html.H4("Tournament Analysis",
                                    className=" card-chart-container", style={"font-size": "2.3vw", "text-align" : "center", 'padding-top': '15px'}),
                            ]
                        )

                    ],
                    style={"min-height" :"0.25rem"}, 
                    )



# WCwinner = html.Div(html.Div(className="card-chart", children=[
#     html.Div(className="card-chart-container", children=[
#         html.Div(className="d-flex justify-content-between", children=[
#             html.Div(className="card-chart-container w-100",
#                     children=[
#                         dls.Roller(
#                             html.H2(className="mt-4 card-title",
#                                     id="tour-winner",
#                                     children = '',
#                                     style={"font-size": "2.5vw"})
#                         ),
#                         html.H6(
#                             className="mb-4 mt-4 card-title", children=["Winner"], style={"font-size": "1.5vw"}
#                         ),
#                     ], style={"text-align": "center"}),

#             html.Div(className="card-icon d-flex align-items-center", children=[
#                 html.Img(className="img-fluid bx-lg",
#                         src="assets/images/stadium.png", style={"width": "9rem"})
#             ]
#             )
#         ])

#     ])
# ], style={"min-height": "11rem"}),
#     className="col-md-6 col-lg-3 mb-md-0 mb-4 card-chart-container"
# )

WCwinner = html.Div(className="col-lg-4 col-md-3 col-sm-3 card-chart-container", children=[html.Div(className="card-chart", children=[
        html.Div(className="card-chart",style={'padding-top': '0px'}, children=[
            html.Div(className="card-chart-container", style={'margin-left': '35px'},children=[
                html.Div(className="card-m-3 pb-2",
                        children=[
                            dls.Roller(
                            children = [
                            html.H2(className="mt-4 mb-1 card-title",
                                    id="tour-winner",
                                    children = '',
                                    style={"font-size": "2.4vw", "text-align": "center"})
                                    ]
                        ),
                        html.Div(className="card-icon d-flex align-items-center w-20 justify-content-center p-1", children=[
                            dls.Roller(
                                html.Img(className="img-fluid",
                                    id="tour-flag",
                                    src = '',
                                    style={
                                        "width": "5em", "box-shadow": "0 2px 6px 0 rgb(67 89 113 / 20%)"}
                                    ),
                        debounce=0
                    )
                ]
                ),
                        html.H6(
                            className="mb-0 mt-4 card-title", children=["Winner"], style={"font-size": "1.5vw"}
                        ),
                        ], style={"text-align": "center"}),
                        

            ])

        ])
    ], style={"min-height": "5rem"})]
    )

WCrunner = html.Div(className="col-lg-4 col-md-3 col-sm-3 card-chart-container", children=[html.Div(className="card-chart", children=[
        html.Div(className="card-chart",style={'padding-top': '0px'}, children=[
            html.Div(className="card-chart-container", style={'margin-left': '35px'},children=[
                html.Div(className="card-m-3 pb-2",
                        children=[
                            dls.Roller(
                            children = [
                            html.H2(className="mt-4 mb-1 card-title",
                                    id="tour-runner",
                                    children = '',
                                    style={"font-size": "2.4vw", "text-align": "center"})
                                    ]
                        ),
                        html.Div(className="card-icon d-flex align-items-center w-20 justify-content-center p-1", children=[
                            dls.Roller(
                                html.Img(className="img-fluid",
                                    id="tour-flag-runner",
                                    src = '',
                                    style={
                                        "width": "5em", "box-shadow": "0 2px 6px 0 rgb(67 89 113 / 20%)"}
                                    ),
                        debounce=0
                    )
                ]
                ),
                        html.H6(
                            className="mb-0 mt-4 card-title", children=["Runner-up"], style={"font-size": "1.5vw"}
                        ),
                        ], style={"text-align": "center"}),
                        

            ])

        ])
    ], style={"min-height": "5rem"})]
    )

WCtour = html.Div(className="col-lg-4 col-md-3 col-sm-3 card-chart-container", children=[html.Div(className="card-chart", children=[
        html.Div(className="card-chart",style={'padding-top': '25px'}, children=[
            html.Div(className="card-chart-container", style={'margin-left': '85px'},children=[
                html.Div(className="card-m-2 me-4 pb-2  ",
                        children=[
                            dbc.Select(
                                id="query-tour-select",
                                value="1930",
                                options=[
                                    {"label": l, "value": l} for l in df_tour.year.values
                                ],
                                style={"width": "14rem"}
                            ),
                            html.P(className="card-text mb-1 fs-sm mt-4 ml-4 pl-4",
                                    style={"font-size": "15px"},
                                    id="tour-host",
                                    children=[f"Host Country: "]),
                            html.P(className="card-text mb-1 fs-sm",
                                    style={"font-size": "15px"},
                                    id="tour-start",
                                    children=[f"Start Date:"]),
                            html.P(className="card-text mb-1 fs-sm",
                                    style={"font-size": "15px"},
                                    id="tour-end",
                                    children=[f"End Date: "]),
                        ]),

            ])

        ])
    ], style={"min-height": "5rem"})]
    )





@callback(
    Output("tour-host", "children"),
    Output("tour-start", "children"),
    Output("tour-end", "children"),
    Output("tour-winner", "children"),
    Output("tour-flag", "src"),
    Output("tour-runner", "children"),
    Output("tour-flag-runner", "src"),
    [Input("query-tour-select", "value")]
    # State("teams-df", "data"),
)

def update_tour_select(value):
    teams_df = pd.read_csv("data/tournaments.csv")
    teams_df1 = pd.read_csv("data/tournaments_1.csv")
    tour_host = f"Host Country: {teams_df.loc[teams_df.year==int(value) , 'host_country'].values[0]}"
    tour_start = f"Start Date: {teams_df.loc[teams_df.year==int(value) , 'start_date'].values[0]}"
    tour_end = f"End Date: {teams_df.loc[teams_df.year==int(value) , 'end_date'].values[0]}"
    tour_winner = f"{teams_df.loc[teams_df.year==int(value) , 'winner'].values[0]}"
    tour_flag = f"assets/flags/{tour_winner}.svg"
    
    tour_runner = f"{teams_df1.loc[teams_df1.year==int(value) , 'runner_up'].values[0]}"
    tour_flag_runner = f"assets/flags/{tour_runner}.svg"
    return tour_host, tour_start, tour_end, tour_winner, tour_flag, tour_runner, tour_flag_runner

