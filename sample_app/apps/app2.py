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
import plotly.graph_objs as go
import scipy.optimize as sco

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "dat/constituents_csv.csv")
df = pd.read_csv(path)
stock_list = df.Symbol.to_list()

layout = html.Div(children=[

    dbc.Jumbotron(
        [
            html.H2("Portfolio Optimization", className="display-3"),
            html.P(
                "Test FNGU using Markowitz's Efficient Frontier Optimization Method ",
                className="lead",
            ),
            html.Hr(className="my-2"),
            html.P(
                "Method: Minimize Variance via Python's Scipy"
            ),
            html.P(dbc.Button("Learn more", color="primary"), className="lead"),
        ]
    ),

    dbc.Row(
        [
            dbc.Col(html.Div([
                html.H6("Create an Equity Portfolio"),
                html.Div([" ",
                          dcc.Dropdown(id='page_2_stock_choice', options=[
                              {'label': 'Amazon', 'value': 'AMZN'},
                              {'label': 'Twitter', 'value': 'TWTR'},
                              {'label': 'Apple', 'value': 'AAPL'},
                              {'label': 'Tesla', 'value': 'TSLA'},
                              {'label': 'Facebook', 'value': 'FB'},
                              {'label': 'Google', 'value': 'GOOG'},
                              {'label': 'Alibaba', 'value': 'BABA'},
                              {'label': 'Netflix', 'value': 'NFLX'},
                              {'label': 'Amazon', 'value': 'AMZN'},
                              {'label': 'Baidu', 'value': 'BIDU'},
                              {'label': 'Nvdia', 'value': 'NVDA'},
                          ], value=['AMZN', 'TWTR', 'AAPL',
                                    'TSLA', 'GOOG', 'FB', 'BABA',
                                    'NFLX', 'AMZN', 'BIDU', 'NVDA'],
                                       multi=True, searchable=True)]), ]), ),

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
                html.H6("Fetch Data From Yahoo Finance"),
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
        tickerDf = tickerData.history(start=start_date, end=end_date, auto_adjust=True)
        all_close[ticker] = tickerDf['Close'].to_list()

    stock_df = pd.DataFrame(all_close)
    risk_free_rate = 0.0178
    returns = stock_df.pct_change().dropna()
    mean_returns = returns.mean()

    cov_matrix = returns.cov()

    results, _ = random_portfolios(6000, mean_returns, cov_matrix, risk_free_rate)

    frontier_x = np.linspace(0.5, 1.8, num=100)

    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, frontier_x)
    frontier_y = [p['fun'] for p in efficient_portfolios]

    # create plot

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results[0, :],
        y=results[1, :],
        mode='markers',
        marker=dict(
            color=np.random.randn(len(results[0, :])),
            colorscale='Viridis',
            line_width=1
        ),
        name='Random', ))

    fig.add_trace(go.Scatter(x=frontier_y, y=frontier_x,
                             mode='lines+markers',
                             name='Efficient Frontier'))

    fig.update_layout(title='Markowitz Portfolio Efficient Frontier', )
    fig.update_layout(legend_title_text="Contestant")
    fig.update_xaxes(title_text="Annualized Risk", tickformat=".2%")
    fig.update_yaxes(title_text="Annualized Return", tickformat=".2%")

    frontier_graph = [
        dcc.Graph(figure=fig)
    ]
    return frontier_graph


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]


def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for _ in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets * [1. / num_assets, ], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

    return result


def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets * [1. / num_assets, ], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients


def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(10)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record
