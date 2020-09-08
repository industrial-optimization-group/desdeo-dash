import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import ALL, Input, Output, State
from desdeo_mcdm.interactive.ENautilus import ENautilus, ENautilusInitialRequest, ENautilusRequest, ENautilusStopRequest
from sklearn.preprocessing import MinMaxScaler

from desdeo_dash import Plotter
from desdeo_dash.server import app

global method
global plotter
global intermediate_solutions
global current_best_solutions
global last_request
global intermediate_choose_from


def layout():
    global method
    global plotter
    global intermediate_solutions

    n_objectives = method._problem.n_of_objectives

    initial_request = method.request_classification()[0]

    objective_values_str = "; ".join(
        [
            f"{name}: {value:.3f}"
            for name, value in zip(method._problem.objective_names, initial_request.content["objective_values"])
        ]
    )

    ideal = method._ideal
    nadir = method._nadir
    is_minimize = method._problem._max_multiplier

    _initial_and_bounds = np.stack((initial_request.content["objective_values"], ideal, nadir))
    figure = plotter.spider_plot_candidates(
        plotter.scaler.transform(_initial_and_bounds), selection=0, labels=["Current best", "Ideal", "Nadir"]
    )

    return html.Div(
        [
            html.Div(
                [
                    html.H3("Classify each of the objectives:"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P(
                                        f"{method._problem.objective_names[i]} is to be "
                                        f"{'minimized' if is_minimize[i] == 1 else 'maximized'}, and may:"
                                    ),
                                    dcc.Dropdown(
                                        options=[
                                            {"label": "Improve", "value": "<"},
                                            {"label": "Improve until", "value": "<="},
                                            {"label": "Stay as it is", "value": "="},
                                            {"label": "Change freely", "value": "0"},
                                            {"label": "Worsen until", "value": ">="},
                                        ],
                                        value="0",
                                        id={"type": "dropdown-class", "index": i},
                                    ),
                                    dcc.Input(
                                        type="hidden",
                                        min=ideal[i] if is_minimize[i] == 1 else -nadir[i],
                                        max=nadir[i] if is_minimize[i] == 1 else -ideal[i],
                                        # value=ideal[i] if is_minimize[i] == 1 else -ideal[i],
                                        placeholder=(
                                            f"{ideal[i] if is_minimize[i] == 1 else -nadir[i]} "
                                            f"to {nadir[i] if is_minimize[i] == 1 else -ideal[i]}"
                                        ),
                                        id={"type": "bound-class", "index": i},
                                    ),
                                ]
                            )
                            for i in range(n_objectives)
                        ]
                        + [
                            html.P("Number of desired solutions?"),
                            dcc.Input(
                                type="number", min=1, max=4, value=1, placeholder="Between 1 and 4", id="num-des-sol"
                            ),
                            html.P(),
                            html.Button("Classifications OK", id="classification-ok-btn", n_clicks=0),
                        ],
                        id="dropdown-classses",
                    ),
                    html.H3("Current solution"),
                    html.P(objective_values_str, id="current-solution"),
                ],
                id="nimbus-classification",
            ),
            html.Div("", style={"display": "none"}, id="classification-trigger-div"),
            html.Div(
                html.Div(
                    [
                        html.H3("Choose alternatives to save to the archive"),
                        dcc.Checklist(id="alternative-checklist"),
                        html.Button("Save", id="save-ok-btn", n_clicks=0),
                    ]
                ),
                id="nimbus-newsolutions",
                style={"display": "block"},
            ),
            html.Div("", style={"display": "none"}, id="newsolutions-trigger-div"),
            html.Div(
                [
                    html.H3(
                        "Either choose two solutions between which to generate a derided number of intermediate solutions or continue"
                    ),
                    html.Div(
                        [
                            html.P("First solution"),
                            dcc.RadioItems(
                                options=[
                                    {"label": "option 1", "value": 1},
                                    {"label": "option 2", "value": 2},
                                    {"label": "option 3", "value": 3},
                                    {"label": "option 4", "value": 4},
                                ],
                                id="inter-first-alternatives",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.P("Second solution"),
                            dcc.RadioItems(
                                options=[
                                    {"label": "option 1", "value": 1},
                                    {"label": "option 2", "value": 2},
                                    {"label": "option 3", "value": 3},
                                    {"label": "option 4", "value": 4},
                                ],
                                id="inter-second-alternatives",
                            ),
                        ]
                    ),
                    dcc.Input(
                        type="number",
                        min=1,
                        step=1,
                        placeholder="Number of intermediate solutions",
                        id="input-intermediate",
                    ),
                    html.Button("Calculate intermediate solutions", id="calculate-intermediate-btn", n_clicks=0),
                    html.Button("Continue", id="continue-btn", n_clicks=0),
                    html.P(
                        "The two chosen alternatives cannot be the same!",
                        style={"display": "none"},
                        id="error-same-choice",
                    ),
                ],
                id="nimbus-intermediate",
                style={"display": "none"},
            ),
            html.Div("", style={"display": "none"}, id="intermediate-trigger-div"),
            html.Div(
                [
                    html.H3(
                        (
                            "Select a preferred solution among the displayed solutions, and either continue to "
                            "classification or end and display final the final solution chosen."
                        )
                    ),
                    dcc.RadioItems(
                        options=[
                            {"label": "option 1", "value": 1},
                            {"label": "option 2", "value": 2},
                            {"label": "option 3", "value": 3},
                            {"label": "option 4", "value": 4},
                        ],
                        id="select-preferred",
                    ),
                    html.Button("Classify", id="select-preferred-btn", n_clicks=0),
                    html.Button("End and display", id="end-btn", n_clicks=0),
                ],
                id="nimbus-preferred",
            ),
            html.Div("", style={"display": "none"}, id="preferred-trigger-div"),
            html.Div([dcc.Graph(figure=figure, id="nimbus-figure")], style={"display": "block"}, id="nimbus-display"),
            html.Div("", style={"display": "none"}, id="restart-trigger-div"),
            html.H3("", id="final-solution"),
            html.P("", id="decision-variables"),
            html.P("", id="objective-values"),
        ],
        id="nimbus-page",
    )


@app.callback(
    [Output({"type": "bound-class", "index": ALL}, "type")], [Input({"type": "dropdown-class", "index": ALL}, "value")]
)
def activate_bound_inputs(dropdown_values):
    # handle the dropdown selections, activate input if dropdown selection mandates one...
    typeof = ["hidden" if v not in ["<=", ">="] else "number" for v in dropdown_values]

    return [typeof]


@app.callback(
    Output("classification-trigger-div", "children"),
    [Input("classification-ok-btn", "n_clicks")],
    [
        State("num-des-sol", "value"),
        State({"type": "dropdown-class", "index": ALL}, "value"),
        State({"type": "bound-class", "index": ALL}, "value"),
    ],
)
def classification_ok(n_clicks, num_of_solutions, classifications, bounds):
    if n_clicks == 0 or num_of_solutions is None:
        raise dash.exceptions.PreventUpdate

    if all([True if c not in ["<=", ">=", "0"] else False for c in classifications]):
        # At least one of the classifications should be to improve to worsen an objective value
        raise dash.exceptions.PreventUpdate

    if all([True if c == "=" else False for c in classifications]):
        # All objectives cannot stay as they are...
        raise dash.exceptions.PreventUpdate

    if any([True if c in ["<=", ">="] and v is None else False for (c, v) in zip(classifications, bounds)]):
        # check that each classifications requiring a bound is given one
        raise dash.exceptions.PreventUpdate

    global method
    global intermediate_solutions
    global last_request

    req = method.request_classification()[0]

    response = {}
    response["classifications"] = classifications
    response["levels"] = bounds
    response["number_of_solutions"] = num_of_solutions
    req.response = response
    last_request = method.iterate(req)[0]

    intermediate_solutions = np.atleast_2d(last_request.content["objectives"])
    # options = [{"label": f"Alternative {i+1}", "value": i} for (i, _) in enumerate(intermediate_solutions)]

    return "Triggered!"


@app.callback(
    [Output("alternative-checklist", "options"), Output("alternative-checklist", "value")],
    [Input("classification-trigger-div", "children"), Input("intermediate-trigger-div", "children")],
)
def update_saveable(_1, _2):
    global last_request

    solutions = np.atleast_2d(last_request.content["objectives"])

    options = [{"label": f"Alternative {i+1}", "value": i} for (i, _) in enumerate(solutions)]

    return options, []


@app.callback(
    Output("newsolutions-trigger-div", "children"),
    [Input("save-ok-btn", "n_clicks")],
    [State("alternative-checklist", "value")],
)
def save_ok(n_clicks, values):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    global last_request
    global method

    print(last_request.content)

    print(values)

    res = {"indices": values}
    last_request.response = res

    last_request = method.iterate(last_request)[0]

    return "Triggered!"


@app.callback(
    [Output("inter-first-alternatives", "options"), Output("inter-second-alternatives", "options")],
    [Input("newsolutions-trigger-div", "children"), Input("intermediate-trigger-div", "children")],
)
def update_intermediate_alternatives(_1, _2):
    global method
    global last_request

    solutions = np.atleast_2d(last_request.content["objectives"])

    options = [{"label": f"Alternative {i+1}", "value": i} for (i, _) in enumerate(solutions)]

    return [options, options]


@app.callback(
    [Output("intermediate-trigger-div", "children"), Output("error-same-choice", "style")],
    [Input("calculate-intermediate-btn", "n_clicks")],
    [
        State("inter-first-alternatives", "value"),
        State("inter-second-alternatives", "value"),
        State("input-intermediate", "value"),
    ],
)
def calculate_intermediate_solutions(n_clicks, first_choice, second_choice, n_solutions):
    global method
    global last_request

    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    if first_choice == second_choice:
        return ["", {"display": "block"}]

    resp = {}
    resp["indices"] = [first_choice, second_choice]
    resp["number_of_desired_solutions"] = n_solutions

    last_request.response = resp

    new_req = method.iterate(last_request)[0]
    last_request = new_req

    return ["Triggered!", {"display": "none"}]


@app.callback(
    [Output("preferred-trigger-div", "children"), Output("select-preferred", "options")],
    [Input("continue-btn", "n_clicks")],
)
def select_preferred(n_clicks):
    global method
    global last_request

    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    resp = {}
    resp["indices"] = []
    resp["number_of_desired_solutions"] = 0

    last_request.response = resp

    new_req = method.iterate(last_request)[0]
    last_request = new_req

    solutions = new_req.content["objectives"]

    return "Triggered!", [{"label": f"Alternative {i+1}", "value": i} for (i, _) in enumerate(solutions)]


@app.callback(
    [Output("restart-trigger-div", "children"), Output("current-solution", "children")],
    [Input("select-preferred-btn", "n_clicks")],
    [State("select-preferred", "value")],
)
def confirm_preferred(n_clicks, value):
    global method
    global last_request

    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    resp = {}
    resp["index"] = value
    resp["continue"] = True

    last_request.response = resp

    new_req = method.iterate(last_request)[0]

    last_request = new_req

    objective_values_str = "; ".join(
        [
            f"{name}: {value:.3f}"
            for name, value in zip(method._problem.objective_names, last_request.content["objective_values"])
        ]
    )

    return ["Triggered!", objective_values_str]


@app.callback(
    [
        Output("final-solution", "children"),
        Output("decision-variables", "children"),
        Output("objective-values", "children"),
    ],
    [Input("end-btn", "n_clicks")],
    [State("select-preferred", "value")],
)
def end(n_clicks, value):
    global method
    global last_request

    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    resp = {}
    resp["index"] = value
    resp["continue"] = False

    last_request.response = resp

    new_req = method.iterate(last_request)[0]

    title = "Final solution"

    decision_vars = f"Decision variable values " f"{method._problem.variable_names}: {new_req.content['solution']}"

    objective_vals = f"Objective values {method._problem.objective_names}: {new_req.content['objective']}"

    return [title, decision_vars, objective_vals]


@app.callback(
    [
        Output("nimbus-classification", "style"),
        Output("nimbus-newsolutions", "style"),
        Output("nimbus-intermediate", "style"),
        Output("nimbus-preferred", "style"),
    ],
    [
        Input("classification-trigger-div", "children"),
        Input("newsolutions-trigger-div", "children"),
        Input("intermediate-trigger-div", "children"),
        Input("preferred-trigger-div", "children"),
        Input("restart-trigger-div", "children"),
        Input("classification-ok-btn", "n_clicks"),
    ],
)
def manage_shown_div(_1, _2, _3, _4, _5, n_clicks):
    context = dash.callback_context

    if len(context.triggered) > 1:
        if "classification-trigger-div.children" in context.triggered[1]["prop_id"]:
            # show save view, hide others
            return [{"display": "none"}, {"display": "block"}, {"display": "none"}, {"display": "none"}]

    if "newsolutions-trigger-div.children" in context.triggered[0]["prop_id"]:
        # show ??? view, hide others
        return [{"display": "none"}, {"display": "none"}, {"display": "block"}, {"display": "none"}]

    if "intermediate-trigger-div.children" in context.triggered[0]["prop_id"]:
        # show ??? view, hide others
        return [{"display": "none"}, {"display": "block"}, {"display": "none"}, {"display": "none"}]

    if "preferred-trigger-div.children" in context.triggered[0]["prop_id"]:
        # show ??? view, hide others
        return [{"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "block"}]

    if "restart-trigger-div.children" in context.triggered[0]["prop_id"]:
        # show ??? view, hide others
        return [{"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}]

    if n_clicks > 0:
        # only fall to down if page load
        raise dash.exceptions.PreventUpdate

    # show classification view, hide others
    return [{"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}]


@app.callback(
    Output("nimbus-figure", "figure"),
    [
        Input("classification-trigger-div", "children"),
        Input("newsolutions-trigger-div", "children"),
        Input("intermediate-trigger-div", "children"),
        Input("preferred-trigger-div", "children"),
        Input("restart-trigger-div", "children"),
    ],
)
def plot_new_solutions(_1, _2, _3, _4, _5):
    global plotter
    global last_request

    if "objectives" in last_request.content:
        solutions = np.atleast_2d(last_request.content["objectives"])
    else:
        solutions = np.atleast_2d(last_request.content["objective_values"])

    figure = plotter.spider_plot_candidates(
        plotter.scaler.transform(solutions),
        selection=0,
        labels=[f"Alternative {i+1}" for (i, _) in enumerate(solutions)],
    )

    return figure


def main():
    # False to prevent double loading
    app.title = "NIMBUS"
    app.config.suppress_callback_exceptions = True
    app.layout = layout()
    app.run_server(debug=True, use_reloader=False)


if __name__ == "__main__":
    from desdeo_problem.Problem import MOProblem
    from desdeo_problem.Objective import _ScalarObjective
    from desdeo_problem.Variable import variable_builder
    from desdeo_problem.Constraint import ScalarConstraint

    from desdeo_mcdm.interactive.NIMBUS import NIMBUS

    # create the problem
    def f_1(x):
        res = 4.07 + 2.27 * x[:, 0]
        return -res

    def f_2(x):
        res = 2.60 + 0.03 * x[:, 0] + 0.02 * x[:, 1] + 0.01 / (1.39 - x[:, 0] ** 2) + 0.30 / (1.39 - x[:, 1] ** 2)
        return -res

    def f_3(x):
        res = 8.21 - 0.71 / (1.09 - x[:, 0] ** 2)
        return -res

    def f_4(x):
        res = 0.96 - 0.96 / (1.09 - x[:, 1] ** 2)
        return -res

    def f_5(x):
        return np.max([np.abs(x[:, 0] - 0.65), np.abs(x[:, 1] - 0.65)], axis=0)

    def c_1(x, f=None):
        x = x.squeeze()
        return (x[0] + x[1]) - 0.5

    f1 = _ScalarObjective(name="f1", evaluator=f_1)
    f2 = _ScalarObjective(name="f2", evaluator=f_2)
    f3 = _ScalarObjective(name="f3", evaluator=f_3)
    f4 = _ScalarObjective(name="f4", evaluator=f_4)
    f5 = _ScalarObjective(name="f5", evaluator=f_5)
    varsl = variable_builder(
        ["x_1", "x_2"], initial_values=[0.5, 0.5], lower_bounds=[0.3, 0.3], upper_bounds=[1.0, 1.0]
    )
    c1 = ScalarConstraint("c1", 2, 5, evaluator=c_1)

    # pre-calculated ideal and nadir
    # ideal = np.array([-6.3397, -3.4310, -7.4998, 5.7610e-7, 0.0])
    # nadir = np.array([-4.7520, -2.8690, -0.3431, 9.6580, 0.3500])
    ideal = np.array([-7.3397, -4.4310, -8.4998, 0, 0.0])
    nadir = np.array([-3.7520, -1.8690, 0.3431, 10.6580, 0.35])

    problem = MOProblem(variables=varsl, objectives=[f1, f2, f3, f4, f5], constraints=[c1], nadir=nadir, ideal=ideal)
    # problem = MOProblem(variables=varsl, objectives=[f1, f2, f3, f4, f5], constraints=[c1])

    max_bool = list(map(lambda x: True if x < 0 else False, problem._max_multiplier))

    # GLOBAL
    global method
    method = NIMBUS(problem, scalar_method="scipy_de")

    # GLOBAL
    global plotter
    scaler = MinMaxScaler((-1, 1))
    scaler.fit(np.stack((ideal, nadir)))

    plotter = Plotter(method._nadir, method._ideal, scaler, max_bool)

    # reqs = method.request_classification()[0]

    # response = {}
    # response["classifications"] = ["<", "<=", "=", ">=", "0"]
    # response["levels"] = [-6, -3, -5, 8, 0.349]
    # response["number_of_solutions"] = 3
    # reqs.response = response

    # res_1 = method.iterate(reqs)[0]
    # res_1.response = {"indices": [0, 1, 2]}

    # res_2 = method.iterate(res_1)[0]
    # response = {}
    # response["indices"] = []
    # response["number_of_desired_solutions"] = 0
    # res_2.response = response

    # res_3 = method.iterate(res_2)[0]
    # response_pref = {}
    # response_pref["index"] = 1
    # response_pref["continue"] = True
    # res_3.response = response_pref

    # res_4 = method.iterate(res_3)

    # display the app
    main()
