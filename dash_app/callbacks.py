import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import numpy as np

from data.regression import Line
from data.classification import Circles
from models.regression import PolynomialRegressor
from dash_app.layout import graph_layout
from dash_app.app import app


line_gen = Line(n_points=100, noise_std_ratio=0.2)
circle_gen = Circles(n_points_per_class=100, n_classes=2, noise_std=0.1)


@app.callback(
    Output('scatter-plot', 'figure'),
    Input('points-button', 'n_clicks'),
    Input('line-button', 'n_clicks'),
    State('data-type-dropdown', 'value'),
    State('poly-rank-input', 'value'),
)
def update_scatter_plot(points_button, line_button, data_type_value, poly_rank_value):
    callback_context = dash.callback_context
    button_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    fig = go.Figure(layout=graph_layout)

    if button_id == 'points-button':

        if data_type_value == 'quadratic':
            line_gen.generate_quadratic()
            x, y = line_gen.get_data()
            fig.add_trace(go.Scatter(x=x.reshape(-1), y=y.reshape(-1), mode='markers', marker={'size': 10, 'color': '#FF6347'}))
        elif data_type_value == 'periodic':
            line_gen.generate_periodic()
            x, y = line_gen.get_data()
            fig.add_trace(go.Scatter(x=x.reshape(-1), y=y.reshape(-1), mode='markers', marker={'size': 10, 'color': '#FF6347'}))
        elif data_type_value == 'circles':
            features, targets = circle_gen.generate()
            x = features[:, 0]
            y = features[:, 1]
            targets = targets.reshape(-1).astype(str)
            targets[np.argwhere(targets=='1.0')] = '#228B22'
            targets[np.argwhere(targets=='2.0')] = '#FF6347'
            fig.add_trace(go.Scatter(x=x.reshape(-1), y=y.reshape(-1), mode='markers', marker={'size': 10, 'color': targets}))

    if button_id == 'line-button':
        x, y = line_gen.get_data()

        quadratic_regressor = PolynomialRegressor(rank=poly_rank_value)
        quadratic_regressor.fit(x, y)
        y_pred = quadratic_regressor.predict(x)

        fig.add_trace(go.Scatter(x=x.reshape(-1), y=y.reshape(-1), mode='markers', marker={'size': 10, 'color': '#FF6347'}))
        fig.add_trace(go.Scatter(x=x.reshape(-1), y=y_pred.reshape(-1), mode='lines', line={'width': 3, 'color': '#0000CD'}))
        fig.update_layout(showlegend=False)

    return fig
