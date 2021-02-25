import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sample_app.app import app
import dash_bootstrap_components as dbc
import pandas as pd
import os.path
import datetime
import yfinance as yf
import numpy as np
import chart_studio.plotly as py
import matplotlib.pyplot as plt
import plotly.graph_objs as go

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "dat/constituents_csv.csv")
df = pd.read_csv(path)
stock_list = df.Symbol.to_list()

layout = html.Div(children=[

    dbc.Jumbotron(
        [
            html.H2("Time Series Model", className="display-3"),
            html.P(
                "Compare multiple models' predictive power ",
                className="lead",
            ),
            html.Hr(className="my-2"),
            html.P(
                "I used three models for now, which are LSTM, BSTS and MLP"
            ),
            html.P(dbc.Button("Learn more", color="primary"), className="lead"),
        ]
    ),

    dbc.Row(
        [
            dbc.Col(html.Div([
                html.H6("Select the equity list"),
                html.Div(["Input: ",
                          dcc.Dropdown(id='page_2_stock_choice', options=[
                              {'label': item1, 'value': item1} for item1 in stock_list
                          ], value='AMZN', multi=True, searchable=True)]), ]), ),

            dbc.Col(html.Div([
                html.H6("Select the Date Range"),
                html.Div([dcc.DatePickerRange(
                    id='page_2_date_range',
                    min_date_allowed=datetime.date(2017, 9, 19),
                    max_date_allowed=datetime.date(2021, 2, 20),
                    initial_visible_month=datetime.date(2021, 2, 10),
                    start_date=datetime.date(2020, 2, 20),
                    end_date=datetime.date(2021, 2, 20)
                )]), ]), ),
            dbc.Col(html.Div([
                html.H6("Fetch Price From Yahoo Finance"),
                html.Button(id='page_2_submit_1', n_clicks=0, children='Submit'), ]), ),

        ]
    ),
    html.Div(id='page_2_frontier_chart'),

]
)


@app.callback(
    Output("page_2_frontier_chart", "children"),
    Input('page_2_submit_1', 'n_clicks'),
    State(component_id='page_2_stock_choice', component_property='value'),
    State(component_id='page_2_date_range', component_property='start_date'),
    State(component_id='page_2_date_range', component_property='end_date'),
)
def fetch_price(n_clicks, ticker_list, start_date, end_date):
    # feel free to change this ticker to any other valid equity.
    all_close = {}
    for ticker in ticker_list:
        tickerData = yf.Ticker(ticker)

        # get the historical prices for this ticker
        tickerDf = tickerData.history(start=start_date, end=end_date, auto_adjust=True)
        all_close[ticker] = tickerDf['Close'].pct_change().to_list()

    return_vec = pd.DataFrame(all_close).dropna().to_numpy()
    n_portfolios = 2000
    means, stds = np.column_stack([
        random_portfolio(return_vec)
        for _ in range(n_portfolios)
    ])
    means = np.array(means).flatten()
    stds = np.array(stds).flatten()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
                x=stds,
                y=means,
                mode='markers',
                marker=dict(
                    color=np.random.randn(len(means)),
                    colorscale='Viridis',
                    line_width=1
                ),
                name='Random', ))
    frontier_graph = [
        dcc.Graph(figure={
            'data': [fig],
            'layout': {
                'title': 'Efficient Frontiers',
                'xaxis': {'title': 'Standard Deviation'},
                'yaxis': {'title': 'Expected Return'},
            }
        })
    ]
    return frontier_graph


def random_portfolio(returns):
    '''
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma


def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)
