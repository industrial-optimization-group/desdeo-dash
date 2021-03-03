import base64
import datetime
import json
import uuid
from collections import OrderedDict
from typing import Union

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
from dash.dependencies import Input, Output, State
from desdeo_mcdm.interactive.ENautilus import ENautilus, ENautilusInitialRequest, ENautilusRequest, ENautilusStopRequest
from sklearn.preprocessing import MinMaxScaler

from desdeo_dash import Plotter
from desdeo_dash.server import app

MANAGER_SIZE = 100


class SessionManagerENautilus:
    MAX_METHOD_STORE = 100
    METHOD_CACHE = OrderedDict()
    LAST_REQUEST_CACHE = OrderedDict()
    LAST_CANDIDATE_CACHE = OrderedDict()
    PLOTTER_CACHE = OrderedDict()

    def __init__(self, *args, **kwargs):
        raise TypeError("SessionManager should not be instantiated; it is purely a static class!")

    @staticmethod
    def config(method_cache_size: int):
        SessionManagerENautilus.MAX_METHOD_STORE = method_cache_size

    @staticmethod
    def add_method(method: ENautilus, uid: str) -> bool:
        """Add a method to the internal cache with an uuid identifier. Return
        True if successfull, otherwise False (for example, a method instance
        is already stored for a given uuid)

        """

        if len(SessionManagerENautilus.METHOD_CACHE) > SessionManagerENautilus.MAX_METHOD_STORE:
            # remove last element if cache is full (first in, first out)
            SessionManagerENautilus.METHOD_CACHE.popitem(False)
            SessionManagerENautilus.LAST_CANDIDATE_CACHE.popitem(False)
            SessionManagerENautilus.PLOTTER_CACHE.popitem(False)
            SessionManagerENautilus.LAST_REQUEST_CACHE.popitem(False)

        SessionManagerENautilus.METHOD_CACHE[uid] = method
        SessionManagerENautilus.LAST_CANDIDATE_CACHE[uid] = None
        SessionManagerENautilus.PLOTTER_CACHE[uid] = None
        SessionManagerENautilus.LAST_REQUEST_CACHE[uid] = None

        return True

    @staticmethod
    def get_method(uid: str) -> Union[ENautilus, None]:
        """Return the method identified by the given UUID. If the given uuid
        does not exist, return None.
        
        """
        if uid in SessionManagerENautilus.METHOD_CACHE:
            return SessionManagerENautilus.METHOD_CACHE[uid]
        else:
            # could not find matching uuid
            return None

    @staticmethod
    def update_last_candidate(uid: str, candidate: np.ndarray) -> bool:
        if uid in SessionManagerENautilus.LAST_CANDIDATE_CACHE:
            SessionManagerENautilus.LAST_CANDIDATE_CACHE[uid] = candidate
            return True

        else:
            return False

    @staticmethod
    def get_last_candidate(uid: str) -> Union[np.ndarray, None]:
        if uid in SessionManagerENautilus.LAST_CANDIDATE_CACHE:
            return SessionManagerENautilus.LAST_CANDIDATE_CACHE[uid]
        else:
            return None

    @staticmethod
    def update_last_request(uid: str, request):
        if uid in SessionManagerENautilus.LAST_REQUEST_CACHE:
            SessionManagerENautilus.LAST_REQUEST_CACHE[uid] = request
            return True

        else:
            return False

    def get_last_request(uid: str):
        if uid in SessionManagerENautilus.LAST_REQUEST_CACHE:
            return SessionManagerENautilus.LAST_REQUEST_CACHE[uid]

        else:
            return False

    @staticmethod
    def add_plotter(plotter: Plotter, uid: str) -> bool:
        if uid in SessionManagerENautilus.METHOD_CACHE:
            SessionManagerENautilus.PLOTTER_CACHE[uid] = plotter
            return True

        else:
            return False

    @staticmethod
    def get_plotter(uid: str) -> Union[Plotter, bool]:
        if uid in SessionManagerENautilus.PLOTTER_CACHE:
            return SessionManagerENautilus.PLOTTER_CACHE[uid]

        else:
            return False


def parse_file_contents(contents, filename, date):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string).decode("utf-8").splitlines()
    objective_names = list(map(lambda x: x.strip(), decoded[0].split(sep=",")))
    multiplier = list(map(lambda x: int(x.strip()), decoded[1].split(sep=",")))
    multiplier_ = np.array(multiplier)
    objective_values_ = np.genfromtxt(decoded[2:], delimiter=",")
    # check for dominated solutions
    tmp = np.zeros(objective_values_.shape)

    # assume all to be minimized, drop dominated solutions
    for (i, e) in enumerate(objective_values_ * multiplier_):
        condition = np.any(np.all(e > objective_values_, axis=1))
        if not condition:
            tmp[i] = e
        else:
            tmp[i] = np.nan

    objective_values = (tmp[~np.isnan(tmp).any(axis=1)] * multiplier_).tolist()

    mod_date = datetime.datetime.fromtimestamp(date)

    parsed_contents = {
        "objective_names": objective_names,
        "multiplier": multiplier,
        "objective_values": objective_values,
    }

    return parsed_contents, content_type, mod_date


def layout(session_id: str = None):
    if not session_id:
        session_id = str(uuid.uuid4())
    return html.Div(
        [
            html.Div(session_id, id="session-id", style={"display": "none"}),
            dcc.Location(id="url", refresh=False, pathname="/enautilus/index/"),
            html.Div(id="page-content-enautilus"),
        ]
    )


def index(uid: str):
    return html.Div(
        [
            dcc.Store(id="enautilus-upload-data-storage"),
            dcc.ConfirmDialog(id="enautilus-alert-bad-upload"),
            html.Div(uid, id="session-id", style={"display": "none"}),
            html.H2(
                "E-NAUTILUS data-based interactive multiobjective optimization demonstration", style={"width": "75%"}
            ),
            html.H3("Data upload"),
            html.P(
                (
                    "To begin, upload a file. The file should contain "
                    "objective values separeted by commas on its columns (a "
                    "CSV file is fine). The first row should contain the objective names. "
                    "The values of the second row "
                    "should indicate if an objective is to be minimized or maximized: "
                    "'1' indicates minimization and '-1' indicates maximization. "
                    "Dominated solutions will be eliminated from the data. "
                    "Provide also a desired number of intermediate points to be shown and the number of iterations to "
                    "to be carried out."
                ),
                style={
                    "width": "75%",
                    # "white-space": "nowrap",
                    # "overflow": "hidden",
                    # "text-overflow": "ellipsis",
                },
            ),
            html.Br(),
            dcc.Markdown(
                """
            Example of file contents (min, max, min):
            ```
            price quality time
            1 -1 1
            5.2, 3.3, 10.1
            3.2, 2.2, 11.1
            4.2, 1.1, 9.8
            ```
            """
            ),
            html.P("Number of iterations:"),
            dcc.Input(id="enautilus-niterations", type="number", placeholder=10, value=10, style={"width": "80%"}),
            html.P("Number of intermediate points to be shown:"),
            dcc.Input(id="enautilus-npoints", type="number", placeholder=5, value=5, style={"width": "80%"}),
            dcc.Upload(
                id="enautilus-upload-data",
                children=html.Div(["Drag and Drop or ", html.A("Select File")]),
                style={
                    "width": "80%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                },
                # Do not allow multiple files to be uploaded
                multiple=False,
            ),
            html.Div(
                [
                    html.Button(
                        dcc.Link(
                            "Start",
                            href="/enautilus/optimize/",
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
                    )
                ],
                id="enautilus-start-div",
                style={"text-align": "center", "display": "none"},
            ),
            html.Div(
                html.Button(
                    "UPLOAD",
                    n_clicks=0,
                    id="enautilus-upload-data-button",
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
                        "background-color": "#ffffd1",
                    },
                ),
                id="enautilus-upload-data-button-div",
                style={"text-align": "center", "display": "none"},
            ),
            dcc.Markdown("", id="enautilus-uploaded-data-preview"),
            # Home button
            html.Div(dcc.Link("Back to method index", href="/")),
        ],
        style={"left": "2.5%", "right": "2.5%"},
    )


def optimization_layout(uid: str):
    method = SessionManagerENautilus.get_method(uid)
    plotter = SessionManagerENautilus.get_plotter(uid)

    columns, data = plotter.make_table(
        zs=np.array([method._nadir, method._ideal]), names=method._objective_names, labels=["nadir", "ideal"]
    )
    return html.Div(
        [
            html.Div([html.Div(uid, "session-id")], id="enautilus-storage-div", style={"display": "none"}),
            # First row
            html.Div(
                [
                    html.H3(
                        f"E-NAUTILUS: Iterations left {method._n_iterations_left}",
                        id="enautilus-title",
                        className="row",
                    ),
                    html.P(
                        (
                            "To start, click 'ITERATE'. Spider plots will be generated after the first iteration. "
                            " The red path in the value paths shows the nadir point.",
                        ),
                        className="six columns",
                        style={"margin": 0},
                        id="enautilus-info",
                    ),
                    dcc.RadioItems(
                        id="enautilus-candidate-selection",
                        options=[],
                        value=-1,
                        labelStyle={"display": "inline-block"},
                        className="three columns",
                    ),
                    html.Button(
                        id="enautilus-iterate-button", n_clicks=0, children="ITERATE", className="three columns"
                    ),
                ],
                className="row",
            ),
            html.Hr(),
            # Second row
            html.Div(
                [
                    # First column
                    html.Div(
                        [
                            html.H4("Spider plots", style={"margin": 0}),
                            dcc.Graph(id="enautilus-spider-plots", figure=plotter.spider_plot_candidates(np.array([]))),
                        ],
                        className="six columns",
                        style={"border": "1px grey solid", "padding": "1em"},
                    ),
                    # Second column
                    html.Div(
                        [
                            html.H4("Value paths. Current selection in red.", style={"margin": 0}),
                            dcc.Graph(
                                id="enautilus-value-paths",
                                figure=plotter.value_path_plot_candidates(
                                    np.array([method._nadir, method._ideal]),
                                    method._objective_names,
                                    labels=["nadir", "ideal"],
                                ),
                            ),
                        ],
                        className="six columns",
                        style={"border": "1px grey solid", "padding": "1em"},
                    ),
                ],
                className="row",
            ),
            # Third row
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Tabled candidate objective values"),
                            dash_table.DataTable(id="enautilus-table", columns=columns, data=data),
                        ],
                        className="row",
                    ),
                    html.Div(
                        [
                            html.H4("Tabled candidate best reachable values"),
                            dash_table.DataTable(id="enautilus-table-best"),
                        ],
                        className="row",
                    ),
                ],
                className="row",
            ),
            # Home button
            html.Div(dcc.Link("Back to method index", href="/")),
        ]
    )


@app.callback(Output("page-content-enautilus", "children"), [Input("url", "pathname"), Input("session-id", "children")])
def display_page(pathname, uid):
    if pathname == "/enautilus/index/":
        return index(uid)
    elif pathname == "/enautilus/optimize/":
        return optimization_layout(uid)
    else:
        raise dash.exceptions.PreventUpdate


@app.callback(
    [
        Output("enautilus-uploaded-data-preview", "children"),
        Output("enautilus-upload-data-button-div", "style"),
        Output("enautilus-upload-data-storage", "data"),
    ],
    [Input("enautilus-upload-data", "contents")],
    [State("enautilus-upload-data", "filename"), State("enautilus-upload-data", "last_modified")],
)
def update_data_preview(content, file_name, mod_date):
    if not content:
        raise dash.exceptions.PreventUpdate

    try:
        _contents, _file_name, _mod_date = parse_file_contents(content, file_name, mod_date)
        nl = "\n\n"
        n = "\n"
        res = (
            f"## Data preview{nl}"
            f"Content type: `{_file_name}`{nl}"
            f"Last modified: `{_mod_date}`{nl}"
            f"Content:{nl}"
            f"```{nl}"
            f"#{', '.join(_contents['objective_names'])}{n}"
            f"{', '.join(['Min' if i == 1 else 'Max' for i in _contents['multiplier']])}{n}"
            f"{n.join([', '.join(list(map(lambda x: str(x), content))) for content in _contents['objective_values'][:10]])}"
            f"{nl}```"
        )
        return res, {"display": "inline"}, json.dumps(_contents)
    except Exception as e:
        return (f"An exception occurred when reading the file: {str(e)}", {"display": "none"}, "")


@app.callback(
    [
        Output("enautilus-start-div", "style"),
        Output("enautilus-alert-bad-upload", "displayed"),
        Output("enautilus-alert-bad-upload", "message"),
    ],
    [Input("enautilus-upload-data-button", "n_clicks")],
    [
        State("session-id", "children"),
        State("enautilus-upload-data-storage", "data"),
        State("enautilus-npoints", "value"),
        State("enautilus-niterations", "value"),
    ],
)
def upload_and_make_method(n, uid, json_data, n_points, n_iterations):
    if n < 1:
        raise dash.exceptions.PreventUpdate

    try:
        dict_data = json.loads(json_data)
        multiplier = dict_data["multiplier"]
        multiplier_bool = [True if x == -1 else False for x in multiplier]
        objective_values = np.array(dict_data["objective_values"])
        objective_names = dict_data["objective_names"]

        scaler = MinMaxScaler((-1, 1))
        scaler.fit(np.where(multiplier_bool, -objective_values, objective_values))
        objective_values_norm = scaler.transform(np.where(multiplier_bool, -objective_values, objective_values))

        ideal = np.min(objective_values_norm, axis=0)
        nadir = np.max(objective_values_norm, axis=0)

        method = ENautilus(objective_values_norm, ideal, nadir, objective_names, multiplier)

        request = method.start()
        request.response = {"n_points": n_points, "n_iterations": n_iterations}

        first_request = method.iterate(request)
        SessionManagerENautilus.add_method(method, uid)
        SessionManagerENautilus.update_last_request(uid, first_request)

        plotter = Plotter(nadir, ideal, scaler, multiplier_bool)
        SessionManagerENautilus.add_plotter(plotter, uid)

        return {"display": "inline"}, False, ""

    except Exception as e:
        return {"display": "none"}, True, str(e)


@app.callback(
    [
        Output("enautilus-candidate-selection", "options"),
        Output("enautilus-title", "children"),
        Output("enautilus-table", "columns"),
        Output("enautilus-table", "data"),
        Output("enautilus-table-best", "columns"),
        Output("enautilus-table-best", "data"),
        Output("enautilus-candidate-selection", "value"),
        Output("enautilus-info", "children"),
    ],
    [Input("enautilus-iterate-button", "n_clicks")],
    [State("session-id", "children"), State("enautilus-candidate-selection", "value")],
)
def update_candidates(n_clicks, uid, candidate_index):
    method = SessionManagerENautilus.get_method(uid)

    if method._n_iterations_left == 0 or n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    last_request = SessionManagerENautilus.get_last_request(uid)

    if n_clicks == 1:
        # first iteration, do not interact
        intermediate_points = last_request.content["points"]
        intermediate_ranges = last_request.content["lower_bounds"]

        msg = (
            "First iteration. Explore the candidates, and select the most preferred one using the radio buttons."
            "To continue iterating after selecting a candidate, click on 'ITERATE'."
        )

    else:
        last_request.response = {"preferred_point_index": candidate_index}

        new_request = method.iterate(last_request)

        if type(new_request) is ENautilusRequest:
            SessionManagerENautilus.update_last_request(uid, new_request)
            SessionManagerENautilus.update_last_candidate(uid, last_request.content["points"][candidate_index])
            intermediate_points = np.atleast_2d(new_request.content["points"])
            intermediate_ranges = np.atleast_2d(new_request.content["lower_bounds"])
            msg = "Select the most preferred candidate and continue iterating."

        elif type(new_request) is ENautilusStopRequest:
            SessionManagerENautilus.update_last_request(uid, new_request)
            SessionManagerENautilus.update_last_candidate(uid, last_request.content["points"][candidate_index])
            intermediate_points = np.atleast_2d(new_request.content["solution"])
            intermediate_ranges = np.atleast_2d(new_request.content["solution"])
            # TODO: Fix iteration count in desdeo-mcdm
            method._n_iterations_left = 0
            msg = "Final solution reached."

    default_candidate = 0

    options = [
        {"label": f"Candidate {ind + 1}", "value": val} for (ind, val) in enumerate(range(len(intermediate_points)))
    ]
    if method._n_iterations_left > 1:
        title = "E-NAUTILUS: Iterations left {}".format(method._n_iterations_left)

    elif method._n_iterations_left == 1:
        title = "Select the final solution."

    else:
        title = "Done. Final solution displayed"
        default_candidate = candidate_index
        intermediate_points = np.atleast_2d(intermediate_points)
        intermediate_ranges = np.atleast_2d(intermediate_ranges)
        options = []

    plotter = SessionManagerENautilus.get_plotter(uid)

    columns, data = plotter.make_table(zs=intermediate_points, names=method._objective_names)

    columns_best, data_best = plotter.make_table(
        zs=intermediate_ranges, names=method._objective_names, row_name=["Best reachable"]
    )

    return options, title, columns, data, columns_best, data_best, default_candidate, msg


@app.callback(
    [
        Output("enautilus-table", "style_data_conditional"),
        Output("enautilus-table-best", "style_data_conditional"),
        Output("enautilus-spider-plots", "figure"),
        Output("enautilus-value-paths", "figure"),
    ],
    [Input("enautilus-candidate-selection", "value")],
    [State("session-id", "children")],
)
def highlight_table_row(candidate_index, uid):
    if candidate_index == -1:
        raise dash.exceptions.PreventUpdate

    method = SessionManagerENautilus.get_method(uid)
    last_request = SessionManagerENautilus.get_last_request(uid)
    plotter = SessionManagerENautilus.get_plotter(uid)
    previous_best = SessionManagerENautilus.get_last_candidate(uid)

    if type(last_request) is ENautilusRequest:
        intermediate_points = last_request.content["points"]
        intermediate_ranges = last_request.content["lower_bounds"]

    elif type(last_request) is ENautilusStopRequest:
        intermediate_points = np.atleast_2d(last_request.content["solution"])
        intermediate_ranges = np.atleast_2d(last_request.content["solution"])

    # just one solution
    if intermediate_points.ndim == 1:
        candidate_index = 0

    style = [{"if": {"row_index": candidate_index}, "backgroundColor": "#0000FF", "color": "white"}]

    if len(intermediate_points) == 0:
        zs = np.array([])
    else:
        zs = intermediate_points

    spider_plots = plotter.spider_plot_candidates(
        zs, names=method._objective_names, best=intermediate_ranges, previous=previous_best, selection=candidate_index
    )

    value_paths = plotter.value_path_plot_candidates(zs, method._objective_names, selection=candidate_index)

    return style, style, spider_plots, value_paths


def main():
    # False to prevent double loading
    app.title = "E-NAUTILUS"
    app.config.suppress_callback_exceptions = True
    app.layout = layout()
    app.run_server(debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
