import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash_app.settings import Colors


colors = Colors()
graph_layout = go.Layout(height=600, paper_bgcolor=colors.get_page_bg(), plot_bgcolor=colors.get_plot_bg(),
                         xaxis={'showgrid': False, 'zeroline': False}, yaxis={'showgrid': False, 'zeroline': False})


external_stylesheets = [
    {
        'href': 'https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-eOJMYsd53ii+scO/bJGFsiCZc+5NDVN2yr8+0RDqr0Ql0h+rP48ckxlpbzKgwra6',
        'crossorigin': 'anonymous'
    }
]


graph_config = {'displayModeBar': False}


layout = html.Div(className='container-fluid', style={'background-color': colors.get_page_bg(), 'height': '100vh'}, children=[
    html.Div(className='container', children=[
        html.Br(),
        html.Div(className='row', children=[
            html.Div(className='col-3', children=[
                html.H1(children='Hello World!'),
            ]),
            html.Div(className='col-2', children=[
                dcc.Dropdown(id='data-type-dropdown', options=[{'label': 'Quadratic', 'value': 'quadratic'},
                                                               {'label': 'Periodic', 'value': 'periodic'},
                                                               {'label': 'Circles', 'value': 'circles'}],
                             style={'height': '38px'}, optionHeight=38, value='quadratic'),
            ]),
            html.Div(className='col-2', children=[
                html.Button('Generate Points', className='btn btn-primary', id='points-button', n_clicks=0),
            ]),
            html.Div(className='col-2', children=[
                dcc.Input(id='poly-rank-input', type='number', className='form-control', value=2)
            ]),
            html.Div(className='col-2', children=[
                html.Button('Fit Line', className='btn btn-primary', id='line-button', n_clicks=0),
            ]),
            # html.Div(className='col-1', children=[
            #     dcc.Checklist(id='dark-mode-toggle', options=[{'label': ' Dark', 'value': 'dark'}], value=[]),
            # ]),
        ]),
        html.Br(),
        dcc.Graph(id='scatter-plot', config=graph_config)
    ])
])
