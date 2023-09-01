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

WCteamchart2 = html.Div(className="card-chart-container col-lg-7 md-1 sm-1",
                    children=[
                        html.Div(
                            className="card-chart",
                            children=[
                                html.H4(id = 'my-head7', children = "Match Results",
                                    className=" card-chart-container", style={"font-size": "1.5vw", "text-align" : "center", 'padding-top': '25px'}),
                                        dls.Roller(
                                            id="team-g6",
                                            children=[
                                    ], debounce=0
                                )
                            ]
                        )

                    ],
                    style={"min-height" :"20.25rem"}
                    )

@callback(
    Output("team-g6", "children"),
    Output("my-head7", "children"),
    Input("query-team1", "value"),
    Input("query-team2", "value"),
    #State("goals-df", "data"),
    #State("qualified-teams-df" , "data")
)

def update_chart2(query_team1, query_team2):
    tms = pd.read_csv('data/team_minute_stats_a.csv')
    tvt = pd.read_csv('data/teamvsteam_a.csv')
    tvt = pd.merge(tvt[['match_id', 'match_date', 'team_name', 'opponent_name', 'goals_for', 'goals_against', 'result']],
                tms[['match_id', 'given_name', 'family_name', 'minute_regulation', 'Event']])

    events = tvt
    events['Event'] = events['Event'].astype(str)

    for i in range(events.shape[0]):
        if events['Event'].values[i]=='2':
            events['Event'].values[i] = 'yellow card'
        elif events['Event'].values[i]=='1':
            events['Event'].values[i] = 'second yellow'
        elif events['Event'].values[i]=='3':
            events['Event'].values[i] = 'red card'
        elif events['Event'].values[i]=='4':
            events['Event'].values[i] = 'goal'
        elif events['Event'].values[i]=='5':
            events['Event'].values[i] = 'own goal'
        else:
            events['Event'].values[i] = 'penalty'

    events['offset'] = np.ones(events.shape[0], dtype=int)

    # select these teams from dropdown
    t1 = f'{query_team1}'
    t2 = f'{query_team2}'

    # get all matches id
    tvtmatches = events.query(f"team_name=='{t1}'&opponent_name=='{t2}'")

    # match options for dropdown menu
    options = tvtmatches['match_date'].unique()

    option = options[-1] # select this from dropdown

    colors = {'yellow card': 'yellow', 'second yellow': 'red', 'goal': 'green', 'penalty': 'purple', 'own goal': 'aliceblue', 'red card': 'red'}
    fig = px.scatter(tvtmatches.query(f"match_date=='{option}'"), x="minute_regulation", y="offset",
                    color="Event", color_discrete_map=colors, hover_data=['family_name', 'given_name'])
    fig.update_traces(marker=dict(size=25,
                                line=dict(width=2,
                                color='DarkSlateGrey')),
                    selector=dict(mode='markers'),
                    opacity=0.7)
    fig.update_layout(
                    yaxis=dict(
                        tickmode='linear',
                        showticklabels=False
                    ))
    fig.update_yaxes(title="Events")
    fig.update_xaxes(title="Minute of regulation")
    

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

    ) , f"{query_team1} V/S {query_team2} Minute by Minute"