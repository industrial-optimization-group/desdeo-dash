import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash_table
from desdeo_dash import Plotter
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys

from desdeov2.problem.Problem import ScalarDataProblem
from desdeov2.methods.Nautilus import ENautilus


xs = np.genfromtxt("./data/decision_result.csv", delimiter=",")
fs = np.genfromtxt("./data/objective_result.csv", delimiter=",")
objective_names = ["obj{}".format(i + 1) for i in range(fs.shape[1])]
is_max = [True, True, False, False, False]
fs = np.where(is_max, -fs, fs)

# scale the data
scaler = MinMaxScaler((-1, 1))
scaler.fit(fs)
fs_norm = scaler.transform(fs)

# create the problem
problem = ScalarDataProblem(xs, fs_norm)
enautilus = ENautilus(problem)
total_iters = 5
points_shown = 4

ideal, nadir = enautilus.initialize(total_iters, points_shown)
plotter = Plotter(nadir, ideal, scaler, is_max)

# this is bad!
intermediate_points = []
intermediate_ranges = []
current_best_idx = 0
previous_best = None
###

# fot the parallel axes
columns, data = plotter.make_table(
    np.array([nadir, ideal]), objective_names, ["nadir", "ideal"]
)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.Div(
            [
                html.H3(
                    "E-NAUTILUS: Iterations left {}".format(enautilus.ith),
                    id="title",
                ),

                html.H4(
                    "Select the best candidate and iterate. "
                    "Or just iterate if first iteration."
                ),

                dcc.RadioItems(
                    id="candidate-selection",
                    options=[
                        {"label": "Candidate {}".format(ind + 1), "value": val}
                        for (ind, val) in enumerate(
                            range(len(intermediate_points))
                        )
                    ],
                    value=-1,
                    labelStyle={"display": "inline-block"},
                ),

                html.Button(
                    id="iterate-button", n_clicks=0, children="ITERATE"
                ),

                dcc.Graph(
                    id="spider-plots",
                    figure=plotter.spider_plot_candidates(np.array([])),
                ),
            ],
            style={
                "columnCount": 1,
                "width": "99%",
                "display": "inline-block",
            },
        ),
        html.Div(
            [

                html.H5("Value paths"),
                dcc.Graph(
                    id="value-paths",
                    figure=plotter.value_path_plot_candidates(
                        np.array([nadir, ideal]),
                        objective_names,
                        labels=["nadir", "ideal"],
                    ),
                ),
                html.H5("Tabled candidate objective values"),
                dash_table.DataTable(id="table", columns=columns, data=data),
                html.H5("Tabled candidate best reachable values"),
                dash_table.DataTable(id="table-best"),
            ],
            style={
                "columnCount": 1,
                "width": "99%",
                "display": "inline-block",
            },
        ),
    ],
    style={"columnCount": 2, "width": "100%"},
)


@app.callback(
    [Output("table", "style_data_conditional"),
     Output("table-best", "style_data_conditional")],
    [Input("candidate-selection", "value")],
)
def highlight_table_row(candidate_index):
    style =  [
        {
            "if": {"row_index": candidate_index},
            "backgroundColor": "#0000FF",
            "color": "white",
        }
    ]

    return style, style


@app.callback(
    [
        Output("spider-plots", "figure"),
        Output("candidate-selection", "options"),
        Output("title", "children"),
        Output("table", "columns"),
        Output("table", "data"),
        Output("table-best", "columns"),
        Output("table-best", "data"),
        Output("value-paths", "figure"),
    ],
    [Input("iterate-button", "n_clicks")],
    [State("candidate-selection", "value")],
)
def update_candidates(n_clicks, candidate_index):
    global intermediate_points
    global intermediate_ranges
    global current_best_idx
    global previous_best

    if n_clicks == 0:
        raise PreventUpdate
    if n_clicks == 1:
        # first iteratino, do not interact
        zs, best = enautilus.iterate()
    else:
        current_best_idx = candidate_index
        enautilus.interact(
            intermediate_points[current_best_idx],
            intermediate_ranges[current_best_idx],
        )
        zs, best = enautilus.iterate()

    intermediate_points = zs
    intermediate_ranges = best

    spider_plots = plotter.spider_plot_candidates(
        zs,
        names=objective_names,
        best=intermediate_ranges,
        previous=previous_best,
    )

    previous_best = zs[current_best_idx]

    options = [
        {"label": "Candidate {}".format(ind + 1), "value": val}
        for (ind, val) in enumerate(range(len(intermediate_points)))
    ]
    title = "E-NAUTILUS: Iterations left {}".format(enautilus.ith)

    columns, data = plotter.make_table(zs, objective_names)

    columns_best, data_best = plotter.make_table(best, objective_names, row_name=["Best reachable"])

    value_paths = plotter.value_path_plot_candidates(zs, objective_names)

    return spider_plots, options, title, columns, data, columns_best, data_best, value_paths


def main():
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
