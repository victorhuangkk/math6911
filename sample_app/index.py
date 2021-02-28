"""
This the controller page for the multiple page dash application

Pages are under apps
"""

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sample_app.apps import app1, app2, app3, app4
from sample_app.app import app

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Math Finance", className="display-4"),
        html.Hr(),
        html.P(
            "Quantitative Asset Exploration Dashboard", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Market Data", href="/page-1", id="page-1-link"),
                html.Hr(),
                dbc.NavLink("Portfolio Optimization", href="/page-2", id="page-2-link"),
                html.Hr(),
                dbc.NavLink("News Impact", href="/page-3", id="page-3-link"),
                # html.Hr(),
                # dbc.NavLink("Neural Network Price Prediction", href="/page-4", id="page-4-link"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1"]:
        return app1.layout
    elif pathname == "/page-2":
        return app2.layout
    elif pathname == "/page-3":
        return app3.layout
    # elif pathname == "/page-4":
    #     return app4.layout
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
    app.run_server(debug=True)
