import plotly.express as px
import pandas as pd
from dash import html, dcc
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_loading_spinners as dls
from dash import callback
import plotly.graph_objects as go


WCteammin = html.Div(className="card-chart-container col-lg-15       md-6 sm-12",
                    children=[
                        html.Div(
                            className="card-chart",
                            children=[
                                html.H4(id = 'my-head3', children = "Match Results",
                                    className=" card-chart-container", style={"font-size": "1.7vw", "text-align" : "center", 'padding-top': '25px'}),
                                        dls.Roller(
                                            id="team-goals-count-per-tour",
                                            children=[
                                    ], debounce=0
                                )
                            ]
                        )

                    ],
                    style={"min-height" :"26.25rem"}
                    )


@callback(
    Output("team-goals-count-per-tour", "children"),
        Output("my-head3", "children"),
    [Input("query-team-select", "value")]
    #State("goals-df", "data"),
    #State("qualified-teams-df" , "data")
)

def update_figures(query_team):
    events = pd.read_csv("data/team_minute_stats.csv")
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

    events.drop(inplace=True, columns=['minute_label', 'minute_stoppage'])

    cats = ['penalty', 'goal', 'own goal', 'yellow card', 'second yellow',
    'red card']
    data = {cat: events.query(f"team_name=='{query_team}'").query(f"Event=='{cat}'") for
    cat in cats}

    goal_mins = data['goal']
    pen_mins = data['penalty']
    og_mins = data['own goal']
    yellow_mins = data['yellow card']
    secyel_mins = data['second yellow']
    red_mins = data['red card']
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=goal_mins["minute_regulation"], name='goal', nbinsx=30, opacity=0.65))
    fig.add_trace(go.Histogram(x=pen_mins["minute_regulation"], name='penalty', nbinsx=30, opacity=0.65))
    fig.add_trace(go.Histogram(x=og_mins["minute_regulation"], name='own goal', marker_color='light green', nbinsx=30, opacity=0.65)) # own goal is in favor that's why green
    fig.add_trace(go.Histogram(x=yellow_mins["minute_regulation"], name='yellow card', marker_color='yellow', nbinsx=30, opacity=0.65))
    fig.add_trace(go.Histogram(x=secyel_mins["minute_regulation"], name='second yellow card', marker_color='orange', nbinsx=30, opacity=0.65))
    fig.add_trace(go.Histogram(x=red_mins["minute_regulation"], name='red card', nbinsx=30, marker_color='red', opacity=0.65))
    #fig.update_layout(title=f'Minutewise stats of {query_team}')
    fig.update_xaxes(title="Minute of regulation")
    fig.update_yaxes(title="Counts")
    
    #figure = px.line(goals_df, x="year", y=["Goals Count","Matches Count"] , labels={"value":"Count" , "year":"Year", "variable":""}, color_discrete_sequence=theme.COLOR_PALLETE)
    
    

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

    ), f'Minute wise stats of {query_team}'