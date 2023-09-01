from dash import html
import dash_bootstrap_components as dbc

mapping = html.Div(className="col-md-12 col-lg-12 mb-md-0 mb-4 ps-2 mr-4 card-chart-container", children=[

    html.Div(className="card", children=[
        dbc.Row(html.Iframe(id = 'map', srcDoc= open("./assets/map/map.html", "r").read(), width= "1000", height= "550")
            , class_name="card pt-2 pb-2"),
    ])
])