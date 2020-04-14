from typing import Union
from collections import OrderedDict
import uuid

import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from desdeo_mcdm.interactive.NautilusNavigator import (
    NautilusNavigator,
    NautilusNavigatorRequest,
)
from dash.dependencies import Input, Output, State, ALL

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "NAUTILUS Navigator"


class SessionManager:
    MAX_METHOD_STORE = None
    METHOD_CACHE = OrderedDict()
    REQUEST_CACHE = OrderedDict()

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "SessionManager should not be instantiated; it is purely a static class!"
        )

    @staticmethod
    def config(method_cache_size: int):
        SessionManager.MAX_METHOD_STORE = method_cache_size

    @staticmethod
    def add_method(method: NautilusNavigator, uid: str) -> bool:
        """Add a method to the internal cache with an uuid identifier. Return
        True if successfull, otherwise False (for example, a method instance
        is already stored for a given uuid)

        """
        if uid in SessionManager.METHOD_CACHE:
            # already in cache, do nothing
            return False

        if len(SessionManager.METHOD_CACHE) > SessionManager.MAX_METHOD_STORE:
            # remove last element if cache is full (first in, first out)
            SessionManager.METHOD_CACHE.popitem(False)

        SessionManager.METHOD_CACHE[uid] = method
        return True

    @staticmethod
    def get_method(uid: str) -> Union[NautilusNavigator, None]:
        """Return the method identified by the given UUID. If the given uuid
        does not exist, return None.
        
        """
        if uid in SessionManager.METHOD_CACHE:
            return SessionManager.METHOD_CACHE[uid]
        else:
            # could not find mathcing uuid
            return None

    @staticmethod
    def add_request(request: NautilusNavigatorRequest, uid: str) -> bool:
        # check key
        if not uid in SessionManager.METHOD_CACHE:
            return False

        if not uid in SessionManager.REQUEST_CACHE:
            SessionManager.REQUEST_CACHE[uid] = []

        SessionManager.REQUEST_CACHE[uid].append(request)

        return True

    @staticmethod
    def get_request(uid: str) -> Union[NautilusNavigatorRequest, None]:
        if not uid in SessionManager.REQUEST_CACHE:
            return None

        return SessionManager.REQUEST_CACHE[uid][-1]


def create_method():
    f1 = np.linspace(1, 100, 50)
    f2 = f1[::-1] ** 2

    pfront = np.stack((f1, f2)).T
    ideal = np.min(pfront, axis=0)
    nadir = np.max(pfront, axis=0)

    method = NautilusNavigator(
        pfront,
        ideal,
        nadir,
        objective_names=["price", "pain"],
        maximize=[-1, -1],
    )

    return method


def update_fig(request, fig):
    content = request.content
    lower_bound = content["reachable_lb"]
    upper_bound = content["reachable_ub"]
    steps_taken = content["step_number"]

    # response = request.response
    # aspiration_levels = response["reference_point"]

    for i in range(content["ideal"].shape[0]):
        for j in range(2):
            fig["data"][3 * i + j]["x"] += (steps_taken,)

        fig["data"][3 * i]["y"] += (lower_bound[i],)
        fig["data"][3 * i + 1]["y"] += (upper_bound[i],)
        # fig["data"][3 * i + 2]["y"] += (aspiration_levels[i],)

    return fig


def make_fig(request):
    n_objectives = request.content["ideal"].shape[0]
    steps_remaining = request.content["steps_remaining"]

    fig = make_subplots(rows=n_objectives, cols=1, shared_xaxes=True)
    fig.update_xaxes(title_text="step", row=n_objectives, col=1)
    fig.update_xaxes(range=[0, steps_remaining + 1])
    for i in range(n_objectives):
        # lower bound
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name="Lower bound of reachable solutions",
                mode="lines",
                line_color="green",
                fillcolor="green",
            ),
            row=i + 1,
            col=1,
        )
        # upper bound
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name="Upper bound of reachable solutions",
                mode="lines",
                line_color="red",
                fillcolor="red",
            ),
            row=i + 1,
            col=1,
        )
        # aspiration levels
        fig.add_trace(
            go.Scatter(
                x=list(range(1, 101)),
                # y=[request.content["ideal"][i]] * 100,
                y=[],
                name=f"Preference",
                mode="lines",
                line_color="black",
                fillcolor="black",
            ),
            row=i + 1,
            col=1,
        )

    fig = update_fig(request, fig)
    return fig


def create_app_layout():
    session_id = str(uuid.uuid4())

    method = create_method()

    n_objectives = method._ideal.shape[0]
    ideal = method._ideal
    nadir = method._nadir
    is_maximize = method._maximize
    objective_names = method._objective_names

    request, _ = method.start()

    response = {
        "reference_point": ideal,
        "speed": 5,
        "go_to_previous": False,
        "stop": False,
    }
    request.response = response

    fig = make_fig(request)

    SessionManager.add_method(method, session_id)
    SessionManager.add_request(request, session_id)

    i = 1
    html_page = html.Div(
        [
            html.Div(
                [
                    html.Div(session_id, "session-id"),
                    dcc.Graph(id="last-navigation-graph", figure=fig),
                ],
                id="storage-div",
                style={"display": "none"},
            ),
            html.H1(f"Session id: {session_id}", id="h1"),
            html.Div(
                html.H6(
                    "Current aspiration levels: "
                    + ", ".join(
                        [
                            f"(f{i+1}){objective_names[i]}: {ideal[i]}"
                            for i in range(n_objectives)
                        ]
                    )
                ),
                "preference-display-div",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Slider(
                                        id={
                                            "type": "preference-slider",
                                            "index": i,
                                        },
                                        min=ideal[i] * -is_maximize[i],
                                        max=nadir[i] * -is_maximize[i],
                                        value=ideal[i] * -is_maximize[i],
                                        step=abs(nadir[i] - ideal[i]) / 100,
                                        updatemode="drag",
                                        vertical=True,
                                    ),
                                    html.Div(
                                        f"({'MIN' if not is_maximize[i] == 1 else 'MAX'})f{i+1}",
                                        id=f"text-f{i}",
                                    ),
                                ],
                                className="three columns",
                            )
                            for i in range(n_objectives)
                        ],
                        id="preference-sliders",
                        className="two columns",
                    ),
                    dcc.Graph(
                        id="navigation-graph",
                        figure=fig,
                        className="ten columns",
                    ),
                    dcc.Interval(
                        id="stepper",
                        interval=1 / response["speed"] * 1000,
                        n_intervals=0,
                        max_intervals=0,
                    ),
                    html.Button(
                        "Start",
                        id="start-button",
                        n_clicks=0,
                        className="row",
                        style={
                            "width": "40%",
                            "margin-left": "30%",
                            "margin-right": "30%",
                        },
                    ),
                ],
                id="navigation-div",
                className="row ",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                id="speed-slider-output",
                                className="two columns",
                                # style={"margin-top": "2.5%"},
                            ),
                            dcc.Slider(
                                id="speed-slider",
                                min=min(method._allowed_speeds),
                                max=max(method._allowed_speeds),
                                step=1,
                                value=1,
                                className="ten columns",
                            ),
                        ],
                        className="six columns",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        "Go to previous step?",
                                        style={"display": "inline-block"},
                                    ),
                                    dcc.RadioItems(
                                        id="previous-point-selection",
                                        options=[
                                            {"label": "Yes", "value": "yes"},
                                            {"label": "No", "value": "no"},
                                        ],
                                        value="no",
                                        labelStyle={"display": "inline-block"},
                                        style={"display": "inline-block"},
                                    ),
                                ],
                                className="six columns",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Select previous point:",
                                        id="previous-point-text",
                                        style={"display": "none"},
                                        className="six columns",
                                    ),
                                    dcc.Input(
                                        id="previous-point-input",
                                        type="number",
                                        placeholder=f"previous step",
                                        style={"display": "none"},
                                        className="six columns",
                                    ),
                                ],
                                id="previous-input-div",
                                className="six columns",
                            ),
                        ],
                        className="six columns",
                    ),
                ],
                id="controls-div",
                className="row",
                style={"margin-top": "2.5%"},
            ),
        ]
    )

    return html_page


@app.callback(
    [Output("preference-display-div", "children")],
    [Input({"type": "preference-slider", "index": ALL}, "value")],
    [State("session-id", "children")],
)
def update_preferences(values, uid):
    method = SessionManager.get_method(uid)
    objective_names = method._objective_names
    n_objectives = method._ideal.shape[0]
    is_maximize = method._maximize

    print(is_maximize)

    res = "Current aspiration levels: " + ", ".join(
        [
            f"(f{i+1}){objective_names[i]}: {values[i]*-is_maximize[i]}"
            for i in range(n_objectives)
        ]
    )

    return [res]


@app.callback(
    [
        Output("previous-point-text", "style"),
        Output("previous-point-input", "style"),
    ],
    [Input("previous-point-selection", "value")],
)
def show_input(value):
    if value == "yes":
        return {"display": "block"}, {"display": "block"}
    else:
        return {"display": "none"}, {"display": "none"}


@app.callback(
    [
        Output("navigation-graph", "figure"),
        Output("last-navigation-graph", "figure"),
    ],
    [
        Input("session-id", "children"),
        Input("stepper", "n_intervals"),
        Input({"type": "preference-slider", "index": ALL}, "value"),
    ],
    [State("stepper", "interval"), State("last-navigation-graph", "figure")],
)
def update_navigation_graph(uid, n_intervals, values, interval, fig):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    method = SessionManager.get_method(uid)
    if trigger_id == "stepper":
        if n_intervals == 0:
            return fig, fig

        last_request = SessionManager.get_request(uid)

        print(last_request.content["steps_remaining"])
        if last_request.content["steps_remaining"] <= 1:
            # stop
            return fig, fig

        response = {
            "reference_point": np.array(values),
            "speed": int(1000 / interval),
            "go_to_previous": False,
            "stop": False,
        }

        last_request.response = response

        new_request, _ = method.iterate(last_request)

        new_fig = update_fig(new_request, fig)

        SessionManager.add_request(new_request, uid)

        return new_fig, new_fig

    else:
        for (i, value) in enumerate(values):
            fig["data"][3 * i + 2]["y"] = fig["data"][3 * i + 2]["y"][
                : method._step_number - 1
            ] + method._steps_remaining * [value]

        return fig, fig


@app.callback(
    [
        Output("stepper", "max_intervals"),
        Output("start-button", "children"),
        Output("stepper", "interval"),
        Output("previous-input-div", "style"),
        Output({"type": "preference-slider", "index": ALL}, "disabled"),
    ],
    [Input("start-button", "n_clicks")],
    [State("speed-slider", "value"), State("preference-sliders", "children")],
)
def start_navigating(n, value, slider_children):
    if n % 2 == 1:
        return (
            -1,
            "Stop",
            (1 / value) * 1000,
            {"display": "none"},
            [True] * len(slider_children),
        )
    else:
        return (
            0,
            "Start",
            (1 / value) * 1000,
            {"display": "inline-block"},
            [False] * len(slider_children),
        )


@app.callback(
    dash.dependencies.Output("speed-slider-output", "children"),
    [dash.dependencies.Input("speed-slider", "value")],
)
def update_output(value):
    return f"Selected speed {value}"


def main():
    SessionManager.config(10)

    app.layout = create_app_layout
    # False to prevent doble loading
    app.run_server(debug=True, use_reloader=True)


if __name__ == "__main__":
    main()
