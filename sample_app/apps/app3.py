import os.path

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import json
import pandas as pd
import requests
from dash.dependencies import Input, Output, State
from newsapi import NewsApiClient

from sample_app.app import app

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "dat/news_content.csv")
record_df = pd.read_csv(path)

# load api
path = os.path.join(my_path, "dat/api_key.json")
f = open(path)
data = json.load(f)


layout = html.Div(children=[
    dbc.Jumbotron(
        [
            html.H2("News Impact", className="display-3"),
            html.P(
                "We would use neural network backed transformer model for embedding and RNN for prediction",
                className="lead",
            ),
            html.Hr(className="my-2"),
            html.P(
                "I used News API and a pre-built natural languages' sentiment analysis now. "
            ),
            html.P(dbc.Button("Learn more", color="primary"), className="lead"),
        ]
    ),

    dbc.Row(
        [
            dbc.Col(html.Div([
                html.H6("Select the equity of your own interest"),
                html.Div(["Input: ",
                          dcc.Dropdown(id='page_3_stock_choice', options=[
                              {'label': 'Amazon', 'value': 'amazon'},
                              {'label': 'Twitter', 'value': 'twitter'},
                              {'label': 'Apple', 'value': 'apple'},
                              {'label': 'Tesla', 'value': 'tesla'},
                              {'label': 'Facebook', 'value': 'facebook'},
                              {'label': 'Google', 'value': 'google'},
                              {'label': 'Alibaba', 'value': 'alibaba'},
                              {'label': 'Netflix', 'value': 'netflix'},
                              {'label': 'Amazon', 'value': 'amazon'},
                              {'label': 'Baidu', 'value': 'baidu'},
                              {'label': 'Nvdia', 'value': 'Nvidia'},
                          ], value='amazon')]), ]), ),

            dbc.Col(html.Div([
                html.H6("Select the News Category"),
                html.Div(["Source: ",
                          dcc.Dropdown(id='web_data_source', options=[
                              {'label': item1, 'value': item1} for item1 in ['business', 'entertainment', 'general',
                                                                             'health', 'science',
                                                                             'sports', 'technology']
                          ], value='business')]), ]), ),
            dbc.Col(html.Div([
                html.H6("Fetch News Data"),
                html.Button(id='page_3_submit_1', n_clicks=0, children='Submit'), ]), ),

        ]
    ),
    dbc.Row(
        html.Div([dash_table.DataTable(
            id='news_data_fetch',
            columns=[{"name": i, "id": i} for i in record_df.columns],
            data=record_df.to_dict('records'),
            editable=True
        )])
    ),

]
)

@app.callback(
    Output("news_data_fetch", "data"),
    Input('page_3_submit_1', 'n_clicks'),
    State(component_id='page_3_stock_choice', component_property='value'),
    State(component_id='web_data_source', component_property='value'),
)
def news_api_fetch(n_clicks, ticker, source):
    # you may add your own api key here
    news_api_key = data.get('news_api')
    deepai_key = data.get('deepai_key')

    newsapi = NewsApiClient(api_key=news_api_key)

    everything = newsapi.get_everything(q=ticker, sort_by='relevancy',
                                        page=2)

    content = everything.get("articles")
    title_list = []
    url_list = []
    date_list = []
    sentiment_list = []

    for ind in range(2):
        title_list.append(content[ind].get('title')[:100])
        url_list.append(content[ind].get('url'))
        date_list.append(content[ind].get('publishedAt')[:10])
        sentiment_list.append(sentiment_analysis(content[ind].get('content'), deepai_key))
    news_df = pd.DataFrame({"title": title_list,
                            "url": url_list,
                            "time": date_list,
                            "sentiment": sentiment_list})

    return news_df.to_dict('records')


def sentiment_analysis(text, deepai_key):
    r = requests.post(
        "https://api.deepai.org/api/sentiment-analysis",
        data={
            'text': text,
        },
        headers={'api-key': deepai_key}
    )
    return r.json().get('output')[0]
