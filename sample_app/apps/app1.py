import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from sample_app.app import app
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

path = os.path.join(my_path, "dat/IWM_holdings.csv")
df_2 = pd.read_csv(path)
stock_list = df_2.Ticker.to_list()
sector_list = list(set(df_2.Sector.to_list()))

layout = html.Div(children=[
    dbc.Jumbotron(
        [
            html.H2("Market Data", className="display-1"),
            html.P(
                "Customize Your Data",
                className="lead",
            ),
            html.Hr(className="my-1"),
            html.P(
                "Pick the Stock You Like and Choose the Data You Need"
            ),
            html.P(dbc.Button("Learn more", color="primary", href='https://schulich.yorku.ca/courses/math-6911-3-00/'),
                   className="lead"),
        ]
    ),

    dbc.Row([
        dbc.Col(html.Div([
            html.H6("Pick a Sector"),
            html.Div([dcc.Dropdown(id='page_1_sector', options=[
                {'label': item1, 'value': item1} for item1 in ['Energy', 'Industrials', 'Information Technology',
                                                               'Consumer Staples', 'Materials', 'Real Estate',
                                                               'Consumer Discretionary', 'Health Care', 'Financials',
                                                               'Utilities', 'Communication', 'Cash and/or Derivatives']
            ], value=['Information Technology'], multi=True, searchable=True)]), ]),
        ),
        html.Br(),
        dbc.Col(html.Div([
            html.H6("Pick a Stock from Russell 2000:"),
            html.Div([dcc.Dropdown(id='my-input', options=[
                {'label': item1, 'value': item1} for item1 in stock_list
            ], value='AMZN', searchable=True)]), ]),
        ),
        html.Br(),

    ]),
    html.Br(),
    dbc.Row([dbc.Col(html.Div([dash_table.DataTable(
        id='datatable-paging',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
    )])), ]),
    html.Br(),
    dbc.Row([dbc.Col(html.Div(dcc.Graph(id="my-output"), ), ), ]),
]
)


@app.callback(
    Output("my-input", "options"),
    [Input('page_1_sector', 'value')]
)
def list_update(value_list):
    df_temp = df_2[df_2.Sector.isin(value_list)]
    return [{'label': item1, 'value': item1} for item1 in df_temp.Ticker.to_list()]


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
    interest_metrics = df.columns.values
    info_list = [information.get('industry'), np.round(information.get('marketCap') / 1000000, 3),
                 np.round(information.get('averageVolume') / 1000000, 3), np.round(information.get('trailingPE'), 2),
                 np.round(information.get('forwardPE'), 2), np.round(information.get('trailingEps'), 2),
                 np.round(information.get('forwardEps'), 2), np.round(information.get('pegRatio'), 2),
                 np.round(information.get('beta'), 2)]
    print(info_list)
    plot_df = pd.DataFrame(info_list, index=interest_metrics).T

    return fig, plot_df.to_dict('records')