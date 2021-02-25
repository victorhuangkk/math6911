import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from app import app
import yfinance as yf
import dash_table
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import numpy as np
import os.path

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "dat/finance_table.csv")
df = pd.read_csv(path)
layout = html.Div(children=[
    dbc.Jumbotron(
        [
            html.H2("Asset Exploration", className="display-1"),
            html.P(
                "Compare multiple models' predictive power ",
                className="lead",
            ),
            html.Hr(className="my-2"),
            html.P(
                "This is exploration part. Feel free to use this page to explore equity data for your own interest."
            ),
            html.P(dbc.Button("Learn more", color="primary"), className="lead"),
        ]
    ),

    dbc.Row(
        [dbc.Col(html.Div([
            html.H6("Select the equity of your own interest"),
            html.Div(["Input: ",
                      dcc.Dropdown(id='my-input', options=[
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
                      ], value='AMZN')]), ]), ),
            html.Br(),

            html.Br(),
            dbc.Col(html.Div([html.Br(), dash_table.DataTable(
                id='datatable-paging',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
            )])),

        ]),
    dbc.Row([dbc.Col(html.Div(dcc.Graph(id="my-output"), ), ), ]),
]
),


@app.callback(
    [Output("my-output", "figure"),
     Output('datatable-paging', 'data'), ],
    [Input(component_id='my-input', component_property='value')]
)
def yahoo_data(ticker):
    tickerData = yf.Ticker(ticker)

    # get the historical prices for this ticker
    tickerDf = tickerData.history(period='1d', start='2018-1-1',
                                  end=datetime.today().strftime('%Y-%m-%d'), auto_adjust=True)

    plot_df = tickerDf.reset_index()

    fig = go.Figure(data=[go.Candlestick(x=plot_df.Date,
                                         open=plot_df.Open, high=plot_df.High,
                                         low=plot_df.Low, close=plot_df.Close)])

    fig.update_layout(
        xaxis_rangeslider_visible='slider' in ticker
    )

    information = tickerData.info
    interest_metrics = ['industry', 'marketCap', 'averageVolume', 'beta',
                        'pegRatio', 'priceToBook']
    info_list = []
    for ind in interest_metrics:
        info_list.append(information.get(ind))
    plot_df = pd.DataFrame(info_list, index=interest_metrics).T

    return fig, plot_df.to_dict('records')
