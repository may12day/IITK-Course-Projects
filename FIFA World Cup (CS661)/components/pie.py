import plotly.express as px
import pandas as pd
from dash import html, dcc
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_loading_spinners as dls
from dash import callback
import plotly.graph_objects as go

WCpiechart = html.Div(className="card-chart-container col-lg-4 md-1 sm-1",
                    children=[
                        html.Div(
                            className="card-chart",
                            children=[
                                html.H4(id = 'my-head1', children = "Match Results",
                                    className=" card-chart-container", style={"font-size": "1.5vw", "text-align" : "center", 'padding-top': '25px'}),
                                        dls.Roller(
                                            id="team-g2",
                                            children=[
                                    ], debounce=0
                                )
                            ]
                        )

                    ],
                    style={"min-height" :"20.25rem"}
                    )


@callback(
    Output("team-g2", "children"),
    Output("my-head1", "children"),
    [Input("query-team-select", "value")]
    #State("goals-df", "data"),
    #State("qualified-teams-df" , "data")
)

def update_figures(query_team):
    tstats = pd.read_csv('data/team_stats.csv')
    df = tstats.query(f"team_name=='{query_team}'")
    data = dict(labels=['Matches won', 'Matches lost', 'Matches tied'],
            values=[df.iloc[0][3], df.iloc[0][4], df.iloc[0][5]])
    colors = ['#89EF08', '#e53935',  '#00A4E4']
    fig = px.pie(data, values='values', names='labels',
                )
    fig.update_traces(marker=dict(colors=colors))
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    

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

    ) , f'Match results of {query_team}'