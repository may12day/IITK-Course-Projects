import pandas as pd     
import plotly           
import plotly.io as pio
import plotly.express as px
from dash import html, dcc
import dash_bootstrap_components as dbc
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#read data 
df = pd.read_csv("data/racebar.csv")


dict_keys=['one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen',
        'fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty','twentyone','twentytwo']

years=[1930,1934,1938,1950,1954,1958,1962,1966,1970,1974,1978,1982,1986,1990,1994,1998,2002,2006,2010,2014,2018,2022]

n_frame={}
#make frames based on the each year
for y, d in zip(years, dict_keys):
    dataframe=df[(df['Year']==y)]
    dataframe=dataframe.nlargest(n=15,columns=['c_goal'])
    dataframe=dataframe.sort_values(by=['Year','c_goal'])

    n_frame[d]=dataframe

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

WCRacechart = create_card(class_name="card-chart-container col-lg-15 col-md-12 col-sm-12",
                                title="Highest Scoring Countries",
                                fig=go.Figure(
    data=[
        go.Bar(
        x=n_frame['one']['c_goal'], y=n_frame['one']['Team_names'],orientation='h',
        text=n_frame['one']['c_goal'], texttemplate='%{text:.3s}',
        textfont={'size':18}, textposition='inside', insidetextanchor='middle',
        width=0.9, marker={'color':n_frame['one']['color']})
    ],
    layout=go.Layout(
        height = 500,
        xaxis=dict(range=[0, 237], autorange=False, title=dict(text='Number of Goals',font=dict(size=18))),
        yaxis=dict(range=[-0.5, 15.5], autorange=False,tickfont=dict(size=14)),
        title=dict(text='World Cup Year: 1930',font=dict(size=28),x=0.5,xanchor='center'),
        # Add button
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                        method="animate",
                        args=[None,
                        {"frame": {"duration": 500, "redraw": True},
                        "transition": {"duration":250,
                        "easing": "linear"}}]
            )]
        )]
    ),
    frames=[
            go.Frame(
                data=[
                        go.Bar(x=value['c_goal'], y=value['Team_names'],
                        orientation='h',text=value['c_goal'],
                        marker={'color':value['color']})
                    ],
                layout=go.Layout(
                        xaxis=dict(range=[0, 237], autorange=False),
                        yaxis=dict(range=[-0.5, 15.5], autorange=False,tickfont=dict(size=14)),
                        title=dict(text='Total goals per team: '+str(value['Year'].values[0]),
                        font=dict(size=28))
                    )
            )
        for key, value in n_frame.items()
    ]
)
                                )