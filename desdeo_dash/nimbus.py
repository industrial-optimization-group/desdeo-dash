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

global method
global plotter


def layout():
    global method
    global plotter

    n_objectives = method._problem.n_of_objectives

    initial_request = method.request_classification()[0]

    objective_values_str = "; ".join(
        [
            f"{name}: {value}"
            for name, value in zip(method._problem.objective_names, initial_request.content["objective_values"])
        ]
    )

    figure = plotter.spider_plot_candidates(
        plotter.scaler.transform(np.atleast_2d(initial_request.content["objective_values"])), selection=0
    )

    ideal = method._ideal
    nadir = method._nadir
    is_minimize = method._problem._max_multiplier

    return html.Div(
        [
            html.H3("Classify each of the objectives:"),
            html.Div(
                [
                    html.Div(
                        [
                            html.P(
                                f"{method._problem.objective_names[i]}: {initial_request.content['objective_values'][i]}"
                            ),
                            dcc.Dropdown(
                                options=[{"label": "Improve", "value": "<"}, {"label": "Improve until", "value": "<="}],
                                value="<",
                                id={"type": "dropdown-class", "index": i},
                            ),
                            dcc.Input(
                                type="number",
                                min=ideal[i] if is_minimize[i] == 1 else -nadir[i],
                                max=nadir[i] if is_minimize[i] == 1 else -ideal[i],
                                value=ideal[i] if is_minimize[i] == 1 else -ideal[i],
                                id={"type": "bound-class", "index": i},
                            ),
                        ]
                    )
                    for i in range(n_objectives)
                ],
                id="dropdown-classses",
            ),
            html.H3("Initial solution"),
            html.P(objective_values_str),
            dcc.Graph(figure=figure),
        ],
        id="nimbus-contents",
    )


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
    ideal = np.array([-6.3397, -3.4310, -7.4998, 5.7610e-7, 0.0])
    nadir = np.array([-4.7520, -2.8690, -0.3431, 9.6580, 0.3500])

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
