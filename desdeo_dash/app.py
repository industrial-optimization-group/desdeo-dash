import uuid

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from desdeo_dash.enautilus import layout as enautilus_layout
from desdeo_dash.nautilus_navigator import layout as navigator_layout
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
                dcc.Link(
                    "Navigator NAUTILUS",
                    href="/navigator/",
                    style={"width": "100%", "height": "100%", "display": "block"},
                ),
                dcc.Link(
                    "E-NAUTILUS", href="/enautilus/", style={"width": "100%", "height": "100%", "display": "block"}
                ),
            ]
        )
    elif pathname == "/navigator/":
        return navigator_layout(session_id)
    elif pathname == "/enautilus/":
        return enautilus_layout(session_id)
    else:
        raise dash.exceptions.PreventUpdate()


if __name__ == "__main__":
    app.run_server(debug=True)
