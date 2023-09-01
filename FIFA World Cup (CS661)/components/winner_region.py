import plotly.express as px
import pandas as pd
from dash import html, dcc
import dash_bootstrap_components as dbc
import numpy as np

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

WCWinnersBar = create_card(class_name="card-chart-container col-lg-12 col-md-12 col-sm-12",
                title="World Cup Holders",
                fig=px.bar(
                df_tour.groupby("winner", as_index=False).size(),
                    x="winner",
                    y="size",
                    height=350,
                    text="size",
                    color_discrete_sequence=["#0084d6", "#F0E68C", "rgb(113,221,55)", "#8592a3",
                                                    "rgba(105, 108, 255, 0.85)", "rgba(3, 195, 236, 0.85)"],
                    labels={"value": "Country",
                                "size": "", "winner": "Winner"},
                                ).update_xaxes(categoryorder="total descending",
                                ).update_layout(margin={"r": 20, "l": 30}))

region = pd.read_csv("data/region_winners.csv")


WCWinnerRegion = create_card(class_name="card-chart-container col-lg-12 col-md-12 col-sm-12",
                title="World Cup Holders By Region",
                fig=px.sunburst(
                    region,
                    path=["region_name", "team_name"],
                    values="size",
                    height=350,
                    color_discrete_sequence=["#0084d6" , "rgb(3,195,236)", "rgb(113,221,55)", "#8592a3",
                                                    "rgba(105, 108, 255, 0.85)", "rgba(3, 195, 236, 0.85)"],
                ).update_layout(
                    font_family="Public Sans, Amiri, Qatar2022, Poppins,",
                    margin={"t": 5, "l": 0, "r": 0, "b": 30}
                ))



df_temp = df_tour
df_temp["year"] = df_tour["year"].astype("str")

HostsCountriesBar = create_card(class_name="card-chart-container col-lg-12 col-md-12 col-sm-12",
                    title="World Cup Hosts",
                    fig=px.bar(df_temp.groupby("host_country", as_index=False).agg({"year": " , ".join, "winner": "size"}),
                        x="host_country", y="winner", text="year",
                        height=350,
                        color_discrete_sequence=["#0084d6", "#F0E68C", "rgb(113,221,55)", "#8592a3",
                                                    "rgba(105, 108, 255, 0.85)", "rgba(3, 195, 236, 0.85)"],
                        labels={"host_country": "Host Country", "winner": "Hosting Times"}
                        ).update_xaxes(categoryorder="total ascending",
                        ).update_layout(margin={"r": 20, "t": 10}))

