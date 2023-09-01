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

# WCcomment = html.Div([
#     html.H2([html.Span("Team V/S Team Analysis ")])], style={"margin-top": "3rem", "text-align" : "center"})

WCcomment = html.Div(className="card-chart-container col-lg-15 ",
                    children=[
                        html.Div(
                            className="card-chart",
                            style={"font-size": "1.5vw", "text-align" : "center",},
                            children=[
                                html.H4("Team V/s Team Comparison",
                                    className=" card-chart-container", style={"font-size": "2.3vw", "text-align" : "center", 'padding-top': '15px'}),
                            ]
                        )

                    ],
                    style={"min-height" :"0.25rem"}, 
                    )

WCcomment1 = html.Div(className="card-chart-container col-lg-15 ",
                    children=[
                        html.Div(
                            className="card-chart",
                            style={"font-size": "1.5vw", "text-align" : "center",},
                            children=[
                                html.H4("Team Analysis",
                                    className=" card-chart-container", style={"font-size": "2.3vw", "text-align" : "center", 'padding-top': '15px'}),
                            ]
                        )

                    ],
                    style={"min-height" :"0.25rem"}, 
                    )


df_teams = pd.read_csv("data/teams.csv")

WCteam1 = html.Div(className="col-lg-3 col-md-3 col-sm-3 card-chart-container", children=[html.Div(className="card-chart", children=[
        html.Div(className="card-chart",style={'padding-top': '25px'}, children=[
            html.Div(className="card-chart-container", style={'margin-left': '52px'},children=[
                html.Div(className="card-m-3 me-5 pb-3",
                        children=[
                            dbc.Select(
                                id="query-team1",
                                value="Brazil",
                                options=[
                                    {"label": l, "value": l} for l in df_teams.team_name.values
                                ],
                                style={"width": "12rem"}
                            ),
                            html.P(className="card-text mb-1 fs-sm mt-4 ml-4 pl-4",
                                    id="team-code1",
                                    children=[f"Team Code: "]),
                            html.P(className="card-text mb-1 fs-sm",
                                    id="team-region1",
                                    children=[f"Region:"]),
                            html.P(className="card-text mb-1 fs-sm",
                                    id="team-confederation1",
                                    children=[f"Conf: "]),
                            html.A(id="query-team-wiki1",
                                    href = '',
                                    target="_blank",
                                    style={"font-size": "0.7rem", "color" : "blue"})
                        ]),

            ])

        ])
    ], style={"min-height": "5rem"})]
    )

WCteam1image = html.Div(className="col-lg-3 col-md-3 col-sm-3 card-chart-container", children=[html.Div(className="card-chart", children=[
        html.Div(className="card-chart",style={'padding-top': '25px'}, children=[
            html.Div(className="card-chart-container", style={'margin-left': '62px'},children=[
                html.Div(className="card-m-3 me-5 pb-3",
                        children=[
                            dls.Roller(
                        html.Img(className="img-fluid bx-lg",
                                id="team-flag1",
                                src = '',
                                style={
                                    "width": "50em", 'margin-top':'30px', 'margin-bottom':'20px', 'margin-left':'5px',
                                    "box-shadow": "2px 2px 6px 0 rgb(67 89 113 / 20%)"}
                                ),
                        debounce=0
                    )
                        ]),

            ])

        ])
    ], style={"min-height": "5rem"})]
    )

@callback(
    Output("team-code1", "children"),
    Output("team-region1", "children"),
    Output("team-confederation1", "children"),
    Output("query-team-wiki1", "href"),
    Output("query-team-wiki1", "children"),
    Output("team-flag1", "src"),
    [Input("query-team1", "value")]
    # State("teams-df", "data"),
)

def update_team1(value):
    teams_df = pd.read_csv("data/teams.csv")
    team_code = f"Team Code: {teams_df.loc[teams_df.team_name==value , 'team_code'].values[0]}"
    team_region = f"Region: {teams_df.loc[teams_df.team_name==value , 'region_name'].values[0]}"
    team_confederation = f"Confedration: {teams_df.loc[teams_df.team_name==value , 'confederation_code'].values[0]}"
    wiki_link = teams_df.loc[teams_df.team_name ==value, 'team_wikipedia_link'].values[0]
    team_flag = f"assets/flags/{value}.svg"
    
    
    
    return team_code, team_region, team_confederation, wiki_link, f"Read More About {value}", team_flag

df_rival = pd.read_csv("data/matches.csv")

WCteam2 = html.Div(className="col-lg-3 col-md-3 col-sm-3 card-chart-container", children=[html.Div(className="card-chart", children=[
        html.Div(className="card-chart",style={'padding-top': '25px'}, children=[
            html.Div(className="card-chart-container", style={'margin-left': '52px'},children=[
                html.Div(className="card-m-3 me-5 pb-3",
                        children=[
                            dbc.Select(
                                id="query-team2",
                                value="",
                                options=[
                                ],
                                style={"width": "12rem"}
                            ),
                            html.P(className="card-text mb-1 fs-sm mt-4 ml-4 pl-4",
                                    id="team-code2",
                                    children=[f"Team Code: "]),
                            html.P(className="card-text mb-1 fs-sm",
                                    id="team-region2",
                                    children=[f"Region:"]),
                            html.P(className="card-text mb-1 fs-sm",
                                    id="team-confederation2",
                                    children=[f"Conf: "]),
                            html.A(id="query-team-wiki2",
                                    href = '',
                                    target="_blank",
                                    style={"font-size": "0.7rem", "color" : "blue"})
                        ]),

            ])

        ])
    ], style={"min-height": "5rem"})]
    )

WCteam2image = html.Div(className="col-lg-3 col-md-3 col-sm-3 card-chart-container", children=[html.Div(className="card-chart", children=[
        html.Div(className="card-chart",style={'padding-top': '25px'}, children=[
            html.Div(className="card-chart-container", style={'margin-left': '62px'},children=[
                html.Div(className="card-m-3 me-5 pb-3",
                        children=[
                            dls.Roller(
                        html.Img(className="img-fluid bx-lg",
                                id="team-flag2",
                                src = '',
                                style={
                                    "width": "50em", 'margin-top':'30px', 'margin-bottom':'20px', 'margin-left':'5px',
                                    "box-shadow": "2px 2px 6px 0 rgb(67 89 113 / 20%)"}
                                ),
                        debounce=0
                    )
                        ]),

            ])

        ])
    ], style={"min-height": "5rem"})]
    )

@callback(
    Output("query-team2", "options"),
    Output("query-team2", "value"),
    [Input("query-team1", "value")]
    # State("teams-df", "data"),
)

def update_rival_select(query_team):
    matches_df = pd.read_csv("data/matches.csv")
    away_teams = matches_df.loc[(
        matches_df.away_team_name == query_team)].home_team_name.unique()
    home_teams = matches_df.loc[(
        matches_df.home_team_name == query_team)].away_team_name.unique()
    rival_teams = set(away_teams)
    rival_teams = list(rival_teams.union(home_teams))
    rival_teams.sort()

    options = [{"label": l, "value": l} for l in rival_teams]
    return options, rival_teams[np.random.randint(0, len(rival_teams))] 


@callback(
    Output("team-code2", "children"),
    Output("team-region2", "children"),
    Output("team-confederation2", "children"),
    Output("query-team-wiki2", "href"),
    Output("query-team-wiki2", "children"),
    Output("team-flag2", "src"),
    [Input("query-team2", "value")]
    # State("teams-df", "data"),
)

def update_team2(value):
    teams_df = pd.read_csv("data/teams.csv")
    team_code = f"Team Code: {teams_df.loc[teams_df.team_name==value , 'team_code'].values[0]}"
    team_region = f"Region: {teams_df.loc[teams_df.team_name==value , 'region_name'].values[0]}"
    team_confederation = f"Confedration: {teams_df.loc[teams_df.team_name==value , 'confederation_code'].values[0]}"
    wiki_link = teams_df.loc[teams_df.team_name ==value, 'team_wikipedia_link'].values[0]
    team_flag = f"assets/flags/{value}.svg"
    
    
    
    return team_code, team_region, team_confederation, wiki_link, f"Read More About {value}", team_flag

