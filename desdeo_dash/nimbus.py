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

    is_minimize = method._problem._max_multiplier

    objective_values_str = "; ".join(
        [
            f"{name}: {value:.3f}"
            for name, value in zip(
                method._problem.objective_names, initial_request.content["objective_values"] * is_minimize
            )
        ]
    )

    ideal = method._ideal
    nadir = method._nadir

    _initial_and_bounds = np.stack((initial_request.content["objective_values"], ideal, nadir))

    figure = plotter.spider_plot_candidates(
        plotter.scaler.transform(_initial_and_bounds),
        selection=0,
        labels=["Current best", "Ideal", "Nadir"],
        names=method._problem.objective_names,
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
                            html.Button("Classifications OK", id="classification-ok-btn", n_clicks=0),
                            html.P(
                                "Check the classifications! At least one objective should be set to improve and one allowed to change freely or detoriate!",
                                id="bad-class-warn",
                                style={"display": "none"},
                            ),
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
                        "Either choose two solutions between which to generate a desidered number of intermediate solutions or continue"
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
                            "classification or end and display the final solution chosen."
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


@app.callback(Output("bad-class-warn", "style"), [Input({"type": "dropdown-class", "index": ALL}, "value")])
def check_classification(classifications):
    able_to_improve = any([True if c in ["<", "<="] else False for c in classifications])
    able_to_worsen = any([True if c in [">=", "0"] else False for c in classifications])

    if able_to_improve and able_to_worsen:
        return {"display": "none"}

    return {"display": "block"}


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

    able_to_improve = any([True if c in ["<", "<="] else False for c in classifications])
    able_to_worsen = any([True if c in [">=", "0"] else False for c in classifications])

    if not (able_to_worsen and able_to_improve):
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
    response["levels"] = [
        bound * method._problem._max_multiplier[i] if bound else None for (i, bound) in enumerate(bounds)
    ]
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

    objective_vals = f"Objective values {method._problem.objective_names}: {new_req.content['objective'] * method._problem._max_multiplier}"

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
        names=method._problem.objective_names,
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

    ### create the problem
    # define the objective functions
    def f_1(x):
        # min x_1 + x_2 + x_3
        res = x[:, 0] + x[:, 1] + x[:, 2]
        return res

    def f_2(x):
        # max x_1 + x_2 + x_3
        res = x[:, 0] + x[:, 1] + x[:, 2]
        return res

    def f_3(x):
        # max x_1 + x_2 - x_3
        res = -x[:, 0] - x[:, 1] - x[:, 2]
        return res

    f1 = _ScalarObjective(name="Price", evaluator=f_1, maximize=False)
    f2 = _ScalarObjective(name="Quality", evaluator=f_2, maximize=True)
    f3 = _ScalarObjective(name="Size", evaluator=f_3, maximize=True)

    objl = [f1, f2, f3]

    # define the variables, bounds -5 <= x_1 and x_2 <= 5
    varsl = variable_builder(
        ["x_1", "x_2", "x_3"], initial_values=[0.5, 0.5, 0.5], lower_bounds=[-5, -5, -5], upper_bounds=[5, 5, 5]
    )

    # define constraints
    def c_1(x, f=None):
        x = x.squeeze()
        # x_1 < 2
        return x[0] + 2

    # name of constraints, num variables, num objectives, evaluator
    c1 = ScalarConstraint("c1", len(varsl), len(objl), evaluator=c_1)

    problem = MOProblem(variables=varsl, objectives=objl, constraints=[c1])
    # problem = MOProblem(variables=varsl, objectives=[f1, f2, f3, f4, f5], constraints=[c1])

    max_bool = list(map(lambda x: True if x < 0 else False, problem._max_multiplier))
    max_multiplier = problem._max_multiplier

    # pre computed ideal and nadir, defined AS IF minimizing all objectives!
    ideal = max_multiplier * np.array([-15, 15, 15])
    nadir = max_multiplier * np.array([15, -15, -15])

    problem.ideal = ideal
    problem.nadir = nadir

    # GLOBAL
    global method
    method = NIMBUS(problem, scalar_method="scipy_de")

    # because we did not supply the ideal and nadir, the NIMBUS method will approximate these value using a payoff table
    ideal = method._ideal
    nadir = method._nadir

    # GLOBAL
    global plotter
    idealnadir = np.stack((ideal, nadir))
    scaler = MinMaxScaler((-1, 1))
    scaler.fit(idealnadir)

    plotter = Plotter(method._nadir, method._ideal, scaler, max_bool)

    # run the app
    main()
