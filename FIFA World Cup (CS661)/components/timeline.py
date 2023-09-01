import plotly.express as px
import pandas as pd
from dash import html, dcc
import dash_bootstrap_components as dbc
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

def create_card(fig, class_name, title="Title"):
    return html.Div(
        html.Div(
            className="card-chart",
            children=[
                html.H4(title,
                        className="card-m-0 mt-1 pt-2 pb-3 text-center", style={"font-size": "1.8vw", "align-text": "center"}),
                dcc.Graph(
                    figure=fig.update_layout(
                        paper_bgcolor="rgb(0,0,0,0)",
                        plot_bgcolor="rgb(0,0,0,0)",
                        legend=dict(bgcolor = "#fff"),
                        font_family = "Public Sans, Amiri, Qatar2022, Poppins,",
                    ),
                    config={"displayModeBar": False},
                )
            ],
        ), className=class_name
    )

df_tour = pd.read_csv("data/tournaments.csv")

N = 22
c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

days = []
for index, row in df_tour.iterrows():
    d = (pd.to_datetime(row["end_date"]) - pd.to_datetime(row["start_date"]))
    e = d.days
    days.append(e)

days.append(28)
year = []
for index, row in df_tour.iterrows():
    year.append(int(row["tournament_id"][3:]))

year.append(2022)

df_check = pd.DataFrame()
df_check["Days"] = days
df_check["WC year"] = year


WCToursTimeline = create_card(class_name="card-chart-container col-lg-12 col-md-12 col-sm-12",
                    title="World Cup Timeline",
                    fig=px.bar(df_check,
                        x="WC year", y="Days", text=[f'WC {i}'for i in year],
                        height=350,
                        color_discrete_sequence=["#0084d6", "#F0E68C", "rgb(113,221,55)", "#8592a3",
                                                    "rgba(105, 108, 255, 0.85)", "rgba(3, 195, 236, 0.85)"],
                        labels={"host_country": "Host Country", "winner": "Hosting Times"}
                        ).update_xaxes(categoryorder="total ascending",
                        ).update_layout(margin={"r": 20, "t": 10}))

df_goals = pd.read_csv("data/goals.csv")
minutes = df_goals['minute_regulation']
hist_data = [minutes]
group_labels = ['goals scored']

WCGoalspermin = create_card(class_name="card-chart-container col-lg-12 col-md-12 col-sm-12",
                    title="Goals per Minute",
                    fig= px.histogram(minutes, labels={'value': 'Minutes'}, nbins=120,
                        color_discrete_sequence=["#0084d6", "#F0E68C", "rgb(113,221,55)", "#8592a3",
                                                    "rgba(105, 108, 255, 0.85)",
                            "rgba(3, 195, 236, 0.85)"]).update_layout(margin={"r": 20, "t": 10}, 
                        showlegend=False, height = 350).update_yaxes(title_text='Goals')
                    )