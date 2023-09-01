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
                        className="card-m-0 pt-2 pb-3 text-center", style={"font-size": "1.8vw", "align-text": "center"}),
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

VenuesAndCitiesBar = create_card(class_name="card-chart-container col-lg-12 col-md-12 col-sm-12",
                                title="Cities and Venues",
                                fig=px.histogram(df_tour, x=df_tour["year"],
                                y=["cities", "venues"],
                                barmode="group", labels={"year": "Year", "variable": ""},
                                color_discrete_sequence= ["#0084d6", "#F0E68C","rgb(113,221,55)", "#8592a3",
                                                    "rgba(105, 108, 255, 0.85)", "rgba(3, 195, 236, 0.85)"],
                                height= 350,
                                ).update_layout(yaxis_title="").update_xaxes(type='category'))


TotalAttendanceLine = create_card(class_name="card-chart-container col-lg-12 col-md-12 col-sm-12",
                    title="Total and Average Attendance",
                    fig=px.line(df_tour.rename(columns={"total_attendance": "Total", "avg_attendance": "Avg."}),
                    x="year", y=["Total", "Avg."],
                    labels={
                            "year": "Year", "value": "", "variable": ""},
                            color_discrete_sequence= ["#0084d6", "#F0E68C", "rgb(113,221,55)", "#8592a3",
                            "rgba(105, 108, 255, 0.85)", "rgba(3, 195, 236, 0.85)"],
                            height= 350,).update_xaxes(type='category'))
