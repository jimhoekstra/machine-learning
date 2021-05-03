import dash
from dash_app.layout import external_stylesheets


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
