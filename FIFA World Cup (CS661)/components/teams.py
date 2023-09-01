from dash.dependencies import Input, Output, State
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html
import dash_loading_spinners as dls
from dash import callback
import plotly.express as px

df_teams = pd.read_csv("data/teams.csv")
df_teams1 = pd.read_csv("data/team_stats.csv")
df_teams2 = pd.read_csv("data/tournaments.csv")



WCmatch = html.Div(html.Div(className="card-chart", children=[
    html.Div(className="card-chart-container", children=[
        html.Div(className="d-flex justify-content-between", children=[
            html.Div(className="card-chart-container w-100",
                    children=[
                        dls.Roller(
                            html.H2(className="mt-4 card-title",
                                    id="matches-count-text",
                                    children = '',
                                    style={"font-size": "5vw"})
                        ),
                        html.H6(
                            className="mb-4 mt-4 card-title", children=["Matches"], style={"font-size": "1.5vw"}
                        ),
                    ], style={"text-align": "center"}),

            html.Div(className="card-icon d-flex align-items-center", children=[
                html.Img(className="img-fluid bx-lg",
                        src="assets/images/stadium.png", style={"width": "9rem"})
            ]
            )
        ])

    ])
], style={"min-height": "11rem"}),
    className="col-md-6 col-lg-3 mb-md-0 mb-4 card-chart-container"
)


WCwinner = html.Div(html.Div(className="card-chart", children=[
    html.Div(className="card-chart-container", children=[
        html.Div(className="d-flex justify-content-between", children=[
            html.Div(className="card-chart-container w-100",
                    children=[
                        dls.Roller(
                            html.H2(className="mb-4 mt-4 card-title",
                                    id="winning-times-text",
                                    children = '',
                                    style={"font-size": "5vw"})
                        ),
                        html.H6(
                            className="mb-1 card-title", children=["Times Winner"], style={"font-size": "1.1vw",'margin-left':'12px'}
                        ),
                        html.Small(
                            className="card-text", id="winning-years-text",
                            children = '',
                            style={"font-size": "0.6rem", 'margin-left':'5px', 'color':'blue'}

                        )
                    ], style={"text-align": "center"}),

            html.Div(className="card-icon d-flex align-items-center", children=[
                html.Img(className="img-fluid bx-lg",
                        src="assets/images/emblem.png", style={"width": "9rem"})
            ]
            )
        ])

    ])
], style={"min-height": "11rem"}),
    className="col-md-6 col-lg-3 mb-md-0 mb-4 card-chart-container"
)


WCimage = html.Div(className="col-lg-3 col-md-3 col-sm-3 card-chart-container", children=[html.Div(className="card-chart", children=[
        html.Div(className="card-chart",style={'padding-top': '25px'}, children=[
            html.Div(className="card-chart-container", style={'margin-left': '62px'},children=[
                html.Div(className="card-m-3 me-5 pb-3",
                        children=[
                            dls.Roller(
                        html.Img(className="img-fluid bx-lg",
                                id="team-flag-main",
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


WCTeamStats = html.Div(className="col-lg-3 col-md-3 col-sm-3 card-chart-container", children=[html.Div(className="card-chart", children=[
        html.Div(className="card-chart",style={'padding-top': '25px'}, children=[
            html.Div(className="card-chart-container", style={'margin-left': '52px'},children=[
                html.Div(className="card-m-3 me-5 pb-3",
                        children=[
                            dbc.Select(
                                id="query-team-select",
                                value="Brazil",
                                options=[
                                    {"label": l, "value": l} for l in df_teams.team_name.values
                                ],
                                style={"width": "12rem"}
                            ),
                            html.P(className="card-text mb-1 fs-sm mt-4 ml-4 pl-4",
                                    id="team-code-text",
                                    children=[f"Team Code: "]),
                            html.P(className="card-text mb-1 fs-sm",
                                    id="team-region-text",
                                    children=[f"Region:"]),
                            html.P(className="card-text mb-1 fs-sm",
                                    id="team-confederation-text",
                                    children=[f"Conf: "]),
                            html.A(id="query-team-wiki-link",
                                    href = '',
                                    target="_blank",
                                    style={"font-size": "0.7rem", "color" : "blue"})
                        ]),

            ])

        ])
    ], style={"min-height": "5rem"})]
    )



@callback(
    Output("team-code-text", "children"),
    Output("team-region-text", "children"),
    Output("team-confederation-text", "children"),
    Output("query-team-wiki-link", "href"),
    Output("query-team-wiki-link", "children"),
    Output("team-flag-main", "src"),
    Output("winning-times-text", "children"),
    Output("winning-years-text", "children"),
    Output("matches-count-text", "children"),
    [Input("query-team-select", "value")]
    # State("teams-df", "data"),
)

def update_team_select(value):
    teams_df = pd.read_csv("data/teams.csv")
    team_code = f"Team Code: {teams_df.loc[teams_df.team_name==value , 'team_code'].values[0]}"
    team_region = f"Region: {teams_df.loc[teams_df.team_name==value , 'region_name'].values[0]}"
    team_confederation = f"Confedration: {teams_df.loc[teams_df.team_name==value , 'confederation_code'].values[0]}"
    wiki_link = teams_df.loc[teams_df.team_name ==value, 'team_wikipedia_link'].values[0]
    team_flag = f"assets/flags/{value}.svg"
    winning_times = df_teams1.loc[df_teams1.team_name ==value]["winning_times"].values[0]
    winning_years = "- ".join(df_teams2.loc[df_teams2.winner ==value, "year"].values.astype("str"))
    matches_count = df_teams1.loc[df_teams1.team_name ==value]["count_matches"].values[0]
    
    
    
    return team_code, team_region, team_confederation, wiki_link, f"Read More About {value}", team_flag, winning_times, winning_years, matches_count
