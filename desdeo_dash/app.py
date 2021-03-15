import uuid

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from desdeo_dash.enautilus import layout as enautilus_layout
from desdeo_dash.lot_sizing import layout as lot_sizing_layout
# from desdeo_dash.nautilus_navigator import layout as navigator_layout
from desdeo_dash.server import app


def layout():
    session_id = str(uuid.uuid4())
    return html.Div(
        [
            html.Div(session_id, id="session-id", style={"display": "none"}),
            dcc.Location(id="url-main", refresh=False),
            html.Div(id="page-content-main"),
        ]
    )


app.layout = layout()


@app.callback(
    Output("page-content-main", "children"), [Input("url-main", "pathname")], [State("session-id", "children")]
)
def display_page(pathname, session_id):
    if pathname == "/":
        return html.Div(
            [
                html.H1("Interactive multiobjective optimization: a hands-on experience!"),
                html.H2("To begin, select a method:"),
                dcc.Link(
                    "NAUTILUS Navigator",
                    href="/navigator/",
                    style={"width": "100%", "height": "100%", "display": "block"},
                ),
                dcc.Link(
                    "E-NAUTILUS", href="/enautilus/", style={"width": "100%", "height": "100%", "display": "block"}
                ),
                html.P("If the links don't work, try refreshing the page."),
                html.A(
                    "What are these methods?", href="https://desdeo-mcdm.readthedocs.io/en/latest/background/index.html"
                ),
                html.P(),
                html.A("Source code for this website", href="https://github.com/gialmisi/desdeo-dash"),
            ]
        )
    elif pathname == "/navigator/":
        return lot_sizing_layout(session_id)
    elif pathname == "/enautilus/":
        return enautilus_layout(session_id)
    else:
        raise dash.exceptions.PreventUpdate()


if __name__ == "__main__":
    app.run_server(debug=True)
