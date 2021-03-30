import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px

from generate_data.regression import Line


external_stylesheets = [
    {
        'href': 'https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-eOJMYsd53ii+scO/bJGFsiCZc+5NDVN2yr8+0RDqr0Ql0h+rP48ckxlpbzKgwra6',
        'crossorigin': 'anonymous'
    }
]


graph_config = {'displayModeBar': False}


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.Div(className='container', children=[
        html.Br(),
        html.Div(className='row', children=[
            html.Div(className='col-8', children=[
                html.H1(children='Hello World!'),
            ]),
            html.Div(className='col-2', children=[
                html.Button('Generate Points', className='btn btn-primary btn-lg', id='points-button', n_clicks=0),
            ]),
            html.Div(className='col-2', children=[
                html.Button('Fit Line', className='btn btn-primary btn-lg', id='line-button', n_clicks=0),
            ]),
        ]),
        html.Br(),
        dcc.Graph(id='scatter-plot', config=graph_config)
    ])
])


@app.callback(
    Output('scatter-plot', 'figure'),
    Input('points-button', 'n_clicks'),
    Input('line-button', 'n_clicks'),
    prevent_initial_call=True
)
def update_scatter_plot(points_button, line_button):
    callback_context = dash.callback_context
    button_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'points-button':
        line_gen = Line(n_points=100, noise_std_ratio=0.2)
        x, y = line_gen.quadratic()
        fig = px.scatter(x=x.reshape(-1), y=y.reshape(-1))
        return fig

    if button_id == 'line_button':
        raise PreventUpdate
