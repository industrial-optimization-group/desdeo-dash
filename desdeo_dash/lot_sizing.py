import base64
import datetime
import json
import uuid
from collections import OrderedDict
from typing import Union

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import ALL, Input, Output, State
from desdeo_mcdm.interactive.NautilusNavigator import (
    NautilusNavigator, NautilusNavigatorRequest)
from plotly.subplots import make_subplots

from desdeo_dash.server import app

# np.set_printoptions(precision=2)

# external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

MANAGER_SIZE = 100


class SessionManager:
    MAX_METHOD_STORE = 100
    METHOD_CACHE = OrderedDict()
    REQUEST_CACHE = OrderedDict()

    def __init__(self, *args, **kwargs):
        raise TypeError("SessionManager should not be instantiated; it is purely a static class!")

    @staticmethod
    def config(method_cache_size: int):
        SessionManager.MAX_METHOD_STORE = method_cache_size

    @staticmethod
    def add_method(method: NautilusNavigator, uid: str) -> bool:
        """Add a method to the internal cache with an uuid identifier. Return
        True if successfull, otherwise False (for example, a method instance
        is already stored for a given uuid)

        """
        # if uid in SessionManager.METHOD_CACHE:
        #     # already in cache, do nothing
        #     return False

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
    def add_request(request: NautilusNavigatorRequest, uid: str, index: int = -1) -> bool:
        # check key
        if not uid in SessionManager.METHOD_CACHE:
            return False

        if not uid in SessionManager.REQUEST_CACHE:
            SessionManager.REQUEST_CACHE[uid] = []

        if index == -1:
            SessionManager.REQUEST_CACHE[uid].append(request)
        else:
            SessionManager.REQUEST_CACHE[uid] = SessionManager.REQUEST_CACHE[uid][:index] + [request]

        return True

    @staticmethod
    def get_request(uid: str, index: int = -1) -> Union[NautilusNavigatorRequest, None]:
        if not uid in SessionManager.REQUEST_CACHE:
            return None

        return SessionManager.REQUEST_CACHE[uid][index]


SessionManager.config(MANAGER_SIZE)


def update_fig(request, fig, minimize):
    content = request.content
    lower_bound = content["reachable_lb"] * minimize
    upper_bound = content["reachable_ub"] * minimize
    steps_taken = content["step_number"]

    # response = request.response
    # aspiration_levels = response["reference_point"]

    for i in range(content["ideal"].shape[0]):
        for j in range(2):
            fig["data"][3 * i + j]["x"] += (steps_taken,)

        # fig["data"][3 * i]["y"] += (lower_bound[i] * minimize[i],)
        # fig["data"][3 * i + 1]["y"] += (upper_bound[i] * minimize[i],)

        fig["data"][3 * i]["y"] += (lower_bound[i],) if minimize[i] == 1 else (upper_bound[i],)
        fig["data"][3 * i + 1]["y"] += (upper_bound[i],) if minimize[i] == 1 else (lower_bound[i],)

        # fig["data"][3 * i + 2]["y"] += (aspiration_levels[i],)

    return fig


def make_fig(request, minimize, objective_names, multipliers):
    n_objectives = request.content["ideal"].shape[0]
    steps_remaining = request.content["steps_remaining"]

    sub_plot_names = []
    for i, name in enumerate(objective_names):
        sub_plot_names.append(name + " (MIN)" if multipliers[i] == 1 else name + " (MAX)")

    fig = make_subplots(
        rows=n_objectives,
        cols=1,
        shared_xaxes=True,
        x_title="Step n",
        y_title="Obj. value",
        subplot_titles=sub_plot_names,
    )
    fig.update_xaxes(row=n_objectives, col=1)
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
                # fillcolor="green",
                showlegend=True if i == 0 else False,
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
                fillcolor="rgba(52, 235, 70, 0.5)",
                fill="tonexty",
                showlegend=True if i == 0 else False,
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
                showlegend=True if i == 0 else False,
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout({"title": "Navigation", "height": 800})
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="right", x=1))
    # fig.update_layout({"plot_bgcolor": "red"})
    #     dict(
    #         shapes=[
    #             {
    #                 "type": "line",
    #                 "x0": 0,
    #                 "x1": 1,
    #                 "xref": "paper",
    #                 "y0": 3,
    #                 "y1": 3,
    #                 "yref": "y",
    #                 "line": {"width": 4, "color": "rgb(30, 30, 30)"},
    #             }
    #         ]
    #     )
    # )
    fig = update_fig(request, fig, minimize)
    return fig


def layout(session_id: str = None, material_id: str = None):
    if not session_id:
        session_id = str(uuid.uuid4())
    if not material_id:
        material_id = "31726010"
    return html.Div(
        [
            html.Div(session_id, id="session-id", style={"display": "none"}),
            html.Div(material_id, id="material-id", style={"display": "none"}),
            dcc.Location(id="url", refresh=False, pathname="/navigator/index"),
            html.Div(id="page-content"),
        ]
    )


def index(uid, material_id):
    return html.Div(
        [
            dcc.Store(id="upload-data-storage"),
            dcc.ConfirmDialog(id="alert-bad-upload"),
            html.Div(uid, id="session-id", style={"display": "none"}),
            html.Div(material_id, id="material-id", style={"display": "none"}),
            html.Img(
                src=app.get_asset_url("nautilus_navigator_logo.png"),
                style={"width": "18%", "position": "absolute", "top": "0px", "right": "2.5%"},
            ),
            html.H2(
                "NAUTILUS Navigator data-based interactive multiobjective optimization demonstration",
                style={"width": "75%"},
            ),
            html.H3("For Lot Sizing Problem"),
            html.P(
                (
                    "Select one material for the decision making process"
                ),
                style={
                    "width": "75%",
                    # "white-space": "nowrap",
                    # "overflow": "hidden",
                    # "text-overflow": "ellipsis",
                },
            ),
            html.Br(),
            html.Label('Material list'),
            dcc.RadioItems(
                options=[
                    {'label': '31726010 -- RING GEAR 82', 'value': '31726010'},
                    {'label': '37636500 -- COVER SFS-EN_1561 - EN-GJL-250 5PS', 'value': '37636500'},
                    {'label': 'ACW179192B -- ASSEMBLY TOOLBOX M1-18VT S3-18VT', 'value': 'ACW179192B'}
                ],
                value='',
                id='material_list'
            ),
            html.Br(),
            html.Div(
                html.Button(
                    dcc.Link(
                        "Start navigation",
                        href="/navigator/navigate",
                        style={"width": "100%", "height": "100%", "display": "block"},
                    ),
                    style={
                        "width": "80%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "solid",
                        "borderRadius": "10px",
                        "textAlign": "center",
                        "margin": "10px",
                        "color": "black",
                        "background-color": "#e7e7e7",
                    },
                    id="start_navigation"
                ),
                id="start-navigating-button-div",
                style={"text-align": "center", "display": "block"},
            ),
            html.Br(),
        ],
        style={"left": "2.5%", "right": "2.5%"},
    )


def navigation_layout(session_id, material):

    objective_names = list(['Purchashing and Ordering Cost (POC)', 'Holding Cost (HC)', 'Cycle Service Level (CSL)', 'Probability of Product Unavailability (PPU)', 'Inventory Turn Over (ITO)'])
    short_objective_names = list(['POC', 'HC', 'CSL', 'PPU', 'ITO'])
    multiplier = list([1,1,-1,1,-1])
    objective_values_ = np.genfromtxt(f"./data/{material}_f_data.csv", delimiter=",")
    # check for dominated solutions
    tmp = np.zeros(objective_values_.shape)

    # assume all to be minimized, drop dominated solutions
    for (i, e) in enumerate(objective_values_ * multiplier):
        condition = np.any(np.all(e > objective_values_, axis=1))
        if not condition:
            tmp[i] = e
        else:
            tmp[i] = np.nan

    objective_values = (tmp[~np.isnan(tmp).any(axis=1)] * multiplier).tolist()
    objective_values = np.array(objective_values) * multiplier

    ideal = np.min(objective_values, axis=0)
    nadir = np.max(objective_values, axis=0)

    # choose correct method here!
    method = NautilusNavigator(objective_values, ideal, nadir, objective_names, multiplier)

    SessionManager.add_method(method, session_id)

    n_objectives = method._ideal.shape[0]
    is_minimize = method._minimize

    request = method.start()

    response = {"reference_point": ideal, "speed": 5, "go_to_previous": False, "stop": False}
    request.response = response

    fig = make_fig(request, is_minimize, objective_names, is_minimize)

    SessionManager.add_request(request, session_id)

    html_page = html.Div(
        [
            html.Div(
                [
                    html.Div(session_id, "session-id"), 
                    html.Div(material, "material-id"), 
                    dcc.Graph(id="last-navigation-graph", figure=fig)
                ],
                id="storage-div",
                style={"display": "none"},
            ),
            html.H2(f"Material {material}", id="material"),
            html.H4("Use the sliders or input preference manually", className="row"),
            html.Div(
                [
                    item
                    for sublist in [
                        [
                            html.P(f" {short_objective_names[i]}:", style={"display": "inline"}),
                            dcc.Input(
                                id={"type": "preference-manual-input", "index": i},
                                type="number",
                                min=ideal[i] if is_minimize[i] == 1 else -nadir[i],
                                max=nadir[i] if is_minimize[i] == 1 else -ideal[i],
                                placeholder=f"{short_objective_names[i]} aspiration",
                            ),
                        ]
                        for i in range(n_objectives)
                    ]
                    for item in sublist
                ],
                id="preference-manual-input-div",
                className="row",
            ),
            html.Div(
                [
                    html.Button(
                        "Ok",
                        id="manual-preference-ok-button",
                        n_clicks=0,
                        # style={"width": "40%", "margin-left": "30%", "margin-right": "30%"},
                    )
                ],
                id="manual-preference-ok-button",
                className="row",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Slider(
                                        id={"type": "preference-slider", "index": i},
                                        min=ideal[i] if is_minimize[i] == 1 else -nadir[i],
                                        max=nadir[i] if is_minimize[i] == 1 else -ideal[i],
                                        value=ideal[i] if is_minimize[i] == 1 else -ideal[i],
                                        step=abs(nadir[i] - ideal[i]) / 1000,
                                        updatemode="drag",
                                        vertical=True,
                                    ),
                                    html.Div(f"{short_objective_names[i]}", id=f"text-f{i}"),
                                    html.Div(f"({'MIN' if is_minimize[i] == 1 else 'MAX'})", id=f"min-f{i}"),
                                ],
                                className="three columns",
                                style={"width": "15%"},
                            )
                            for i in range(n_objectives)
                        ],
                        id="preference-sliders",
                        className="two columns",
                    ),
                    dcc.Graph(
                        id="navigation-graph",
                        figure=fig,
                        # config={"edits": {"shapePosition": True}},
                        className="eight columns",
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.P(
                                    "Current best reachable value:\n"
                                    + ";\n".join(
                                        [
                                            f"{short_objective_names[i]}: {ideal[i]*is_minimize[i]}"
                                            for i in range(n_objectives)
                                        ]
                                    )
                                ),
                                "best-reachable-display-div",
                            ),
                            html.Div(
                                html.P(
                                    "Current worst reachable worst:\n"
                                    + ";\n".join(
                                        [
                                            f"{short_objective_names[i]}: {nadir[i]*is_minimize[i]}"
                                            for i in range(n_objectives)
                                        ]
                                    )
                                ),
                                "worst-reachable-display-div",
                            ),
                        ],
                        className="two columns",
                        style={"margin-top": "7%"},
                    ),
                    dcc.Interval(id="stepper", interval=1 / response["speed"] * 1000, n_intervals=0, max_intervals=0),
                    html.Button(
                        "Start",
                        id="start-button",
                        n_clicks=0,
                        className="row",
                        style={"width": "40%", "margin-left": "30%", "margin-right": "30%"},
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
                                    html.Div("Go to previous step?", style={"display": "inline-block"}),
                                    dcc.RadioItems(
                                        id="previous-point-selection",
                                        options=[{"label": "Yes", "value": "yes"}, {"label": "No", "value": "no"}],
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
                                    html.Button("Ok", id="previous-point-ok-button", style={"display": "none"}),
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
            # Home button
            html.Div(dcc.Link("Back to method index", href="/")),
        ]
    )

    return html_page


@app.callback(
    [
        Output("best-reachable-display-div", "children"),
        Output("worst-reachable-display-div", "children"),
        Output({"type": "preference-manual-input", "index": ALL}, "value"),
    ],
    [
        Input({"type": "preference-slider", "index": ALL}, "value"),
        Input("stepper", "n_intervals"),
        Input("previous-point-ok-button", "n_clicks"),
    ],
    [
        State("session-id", "children"),
        State("material-id", "children")
    ],
)
def update_preferences(values, _, prev_input_clicks, uid, material):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "previous-point-ok-button":
        if prev_input_clicks == 0:
            raise dash.exceptions.PreventUpdate

    method = SessionManager.get_method(uid)
    minimize = method._minimize
    objective_names = method._objective_names
    short_objective_names = list(['POC', 'HC', 'CSL', 'PPU', 'ITO'])
    n_objectives = method._ideal.shape[0]
    request = SessionManager.get_request(uid)
    content = request.content

    lower_bounds = content["reachable_lb"] * minimize
    upper_bounds = content["reachable_ub"] * minimize

    res_best = ['Current best reachable values:', html.Br(),'POC : ', lower_bounds[0], html.Br(),
                'HC : ', lower_bounds[1], html.Br(), 'CSL : ', lower_bounds[2], html.Br(), 
                'PPU : ', lower_bounds[3], html.Br(), 'ITO : ', lower_bounds[4]
#        [f"{short_objective_names[i]}: {lower_bounds[i]}", html.Br(), for i in range(n_objectives)]
    ]

    res_worst = ['Current worst reachable values:', html.Br(),'POC : ', upper_bounds[0], html.Br(),
                'HC : ', upper_bounds[1], html.Br(), 'CSL : ', upper_bounds[2], html.Br(), 
                'PPU : ', upper_bounds[3], html.Br(), 'ITO : ', upper_bounds[4]
#        [f"{short_objective_names[i]}: {upper_bounds[i]}" for i in range(n_objectives)]
    ]

    if request.content["steps_remaining"] <= 1:
        i_final = method._projection_index + 1
        res_final = ['Final objective function values:', html.Br(),'POC : ', lower_bounds[0], html.Br(),
                    'HC : ', lower_bounds[1], html.Br(), 'CSL : ', lower_bounds[2], html.Br(), 
                    'PPU : ', lower_bounds[3], html.Br(), 'ITO : ', lower_bounds[4]
        ]
        x_data = np.genfromtxt(f"data/{material}_x_data.csv", delimiter=",")
        q_data = x_data[:, :41]
        ss_data = x_data[:, 41]
        st_data = x_data[:, 42]
        decision_final = ['Final solution : ', html.Br(), 'Q : [', ", ".join([str(int(q)) for q in q_data[i_final]]), ']', html.Br(),
                         'SS : ', ss_data[i_final], html.Br(), 'SLT : ', st_data[i_final]
                         ]
        return res_final, decision_final, values

    return res_best, res_worst, values


@app.callback(
    [Output({"type": "preference-slider", "index": ALL}, "value")],
    [Input("manual-preference-ok-button", "n_clicks")],
    [State({"type": "preference-manual-input", "index": ALL}, "value")],
)
def manual_preference_input(n_clicks, values):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    return [values]


@app.callback(
    [
        Output("previous-point-text", "style"),
        Output("previous-point-input", "style"),
        Output("previous-point-ok-button", "style"),
    ],
    [Input("previous-point-selection", "value")],
)
def show_input(value):
    if value == "yes":
        return {"display": "block"}, {"display": "block"}, {"display": "block"}
    else:
        return {"display": "none"}, {"display": "none"}, {"display": "none"}


@app.callback(
    [
        Output("navigation-graph", "figure"),
        Output("last-navigation-graph", "figure"),
        Output("previous-point-selection", "value"),
    ],
    [
        Input("stepper", "n_intervals"),
        Input({"type": "preference-slider", "index": ALL}, "value"),
        Input("previous-point-ok-button", "n_clicks"),
    ],
    [
        State("session-id", "children"),
        State("previous-point-selection", "value"),
        State("previous-point-input", "value"),
        State("stepper", "interval"),
        State("last-navigation-graph", "figure"),
    ],
)
def update_navigation_graph(
    n_intervals, values, prev_input_clicks, uid, go_to_previous, previous_point, interval, fig
):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    method = SessionManager.get_method(uid)
    if trigger_id == "stepper":
        if n_intervals == 0:
            return fig, fig, "no"

        last_request = SessionManager.get_request(uid)

        if last_request.content["steps_remaining"] <= 1:
            # stop
            return (
                fig,
                fig,
                "no"
#                f"Solution navigated to {method._pareto_front[method._projection_index]*method._minimize}",
            )

        response = {
            "reference_point": np.array(values) * method._minimize,
            "speed": int(1000 / interval),
            "go_to_previous": False,
            "stop": False,
        }

        last_request.response = response

        new_request = method.iterate(last_request)

        SessionManager.add_request(new_request, uid)

        new_fig = update_fig(new_request, fig, method._minimize)
        return (new_fig, new_fig, "no")
        # else:
        #     raise dash.exceptions.PreventUpdate
        #     # for i in range(method._ideal.shape[0]):
        #     #     fig["data"][3 * i + 0]["y"] = fig["data"][3 * i + 0]["y"][:step]
        #     #     fig["data"][3 * i + 0]["x"] = fig["data"][3 * i + 0]["x"][:step]
        #     #     fig["data"][3 * i + 1]["y"] = fig["data"][3 * i + 1]["y"][:step]
        #     #     fig["data"][3 * i + 1]["x"] = fig["data"][3 * i + 1]["x"][:step]
        #     #     fig["data"][3 * i + 2]["y"] = fig["data"][3 * i + 2]["y"][
        #     #         : method._step_number - 1
        #     #     ] + method._steps_remaining * [values[i]]

        #     # return fig, fig, "no", solution_reached

    elif trigger_id == "previous-point-ok-button":
        if prev_input_clicks == 0:
            raise dash.exceptions.PreventUpdate

        step = int(previous_point)
        if step > 0 and step < method._step_number:
            last_request = SessionManager.get_request(uid, step - 1)
        else:
            raise dash.exceptions.PreventUpdate

        response = {
            "reference_point": np.array(values) * method._minimize,
            "speed": int(1000 / interval),
            "go_to_previous": True,
            "stop": False,
        }

        last_request.response = response

        new_request = method.iterate(last_request)

        SessionManager.add_request(new_request, uid, step - 1)

        for i in range(method._ideal.shape[0]):
            fig["data"][3 * i + 0]["y"] = fig["data"][3 * i + 0]["y"][:step]
            fig["data"][3 * i + 0]["x"] = fig["data"][3 * i + 0]["x"][:step]
            fig["data"][3 * i + 1]["y"] = fig["data"][3 * i + 1]["y"][:step]
            fig["data"][3 * i + 1]["x"] = fig["data"][3 * i + 1]["x"][:step]
            # fig["data"][3 * i + 2]["y"] = 100 * [values[i]]
            fig["data"][3 * i + 2]["y"] = fig["data"][3 * i + 2]["y"][
                : method._step_number - 1
            ] + method._steps_remaining * [values[i]]

        return fig, fig, "no"

    else:
        for (i, value) in enumerate(values):
            fig["data"][3 * i + 2]["y"] = fig["data"][3 * i + 2]["y"][
                : method._step_number - 1
            ] + method._steps_remaining * [value]
            # fig["data"][3 * i + 2]["y"] = 100 * [value]
        return fig, fig, go_to_previous


@app.callback(
    [
        Output("stepper", "max_intervals"),
        Output("start-button", "children"),
        Output("stepper", "interval"),
        Output("previous-input-div", "style"),
        Output({"type": "preference-slider", "index": ALL}, "disabled"),
        Output("speed-slider", "disabled"),
        Output("previous-point-selection", "options"),
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
            True,
            [{"disabled": True, "label": "Yes", "value": "yes"}, {"disabled": True, "label": "No", "value": "no"}],
        )
    else:
        return (
            0,
            "Start",
            (1 / value) * 1000,
            {"display": "inline-block"},
            [False] * len(slider_children),
            False,
            [{"disabled": False, "label": "Yes", "value": "yes"}, {"disabled": False, "label": "No", "value": "no"}],
        )


@app.callback(
    dash.dependencies.Output("speed-slider-output", "children"), [dash.dependencies.Input("speed-slider", "value")]
)
def update_output(value):
    return f"Selected speed {value}"


@app.callback(Output("page-content", "children"), [Input("url", "pathname"), Input("session-id", "children"),Input("material-id", "children")])
def display_page(pathname, uid, material):
    if pathname == "/navigator/navigate":
        return navigation_layout(uid, material)
    elif pathname == "/navigator/index":
        return index(uid, material)
    else:
        raise dash.exceptions.PreventUpdate

@app.callback(
    Output("material-id", "children"),
    [Input(component_id='material_list', component_property='value')]
)
def update_material(material_selected):
#    print(material_selected)
    return material_selected

def main():
    # False to prevent doble loading
    app.title = "Navigator NAUTILUS"
    app.config.suppress_callback_exceptions = True
    app.layout = layout()
    app.run_server(debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
