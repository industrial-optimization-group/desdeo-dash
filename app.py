import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from desdeo_dash import Plotter
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from desdeov2.problem.Problem import ScalarDataProblem
from desdeov2.methods.Nautilus import ENautilus

# notes: thousands separator, lot size 48th and safety stock 49th, value paths are nice, max text with arrow, days on hand, 254/inventory turnover (days on hand)


data = np.genfromtxt("./data/first_data.csv", delimiter=",")
xs = data[:, :50]
fs = data[:, 50:]

objective_names = [
    "Purchasing and ordering cost",
    "Holding cost",
    "Cycle service level",
    "Probability of product availability",
    "Inventory turnoever",
]
variable_names = ["x{}".format(i + 1) for i in range(xs.shape[1])]
is_max = [False, False, True, True, True]

# scale the data
scaler = MinMaxScaler((-1, 1))
scaler.fit(np.where(is_max, -fs, fs))
fs_norm = scaler.transform(np.where(is_max, -fs, fs))

# create the problem
problem = ScalarDataProblem(xs, fs_norm)
enautilus = ENautilus(problem)
total_iters = 5
points_shown = 4

nadir, ideal = enautilus.initialize(total_iters, points_shown)
plotter = Plotter(nadir, ideal, scaler, is_max)

# this is bad!
intermediate_points = []
intermediate_ranges = []
current_best_idx = 0
previous_best = None
###

# fot the parallel axes
columns, data = plotter.make_table(
    zs=np.array([nadir, ideal]),
    names=objective_names,
    labels=["nadir", "ideal"],
)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        # First row
        html.Div(
            [
                html.H3(
                    "E-NAUTILUS: Iterations left {}".format(enautilus.ith),
                    id="title",
                    className="row",
                ),
                html.H4(
                    (
                        "Select the best candidate and iterate. "
                        "Or just iterate if first iteration."
                    ),
                    className="six columns",
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
                    className="three columns",
                ),
                html.Button(
                    id="iterate-button",
                    n_clicks=0,
                    children="ITERATE",
                    className="three columns",
                ),
            ],
            className="row",
        ),
        # Second row
        html.Div(
            [
                # First column
                html.Div(
                    [
                        html.H5("Spider plots"),
                        html.P("info", id="info"),
                        dcc.Graph(
                            id="spider-plots",
                            figure=plotter.spider_plot_candidates(
                                np.array([])
                            ),
                        ),
                    ],
                    className="six columns",
                ),
                # Second column
                html.Div(
                    [
                        html.H5(
                            "Value paths (double click on first axis to show all paths)"
                        ),
                        html.P("Final lot size and safety stocks will be shown "
                               "here when solutions are reached.", id="final"),
                        dcc.Graph(
                            id="value-paths",
                            figure=plotter.value_path_plot_candidates(
                                np.array([nadir, ideal]),
                                objective_names,
                                labels=["nadir", "ideal"],
                            ),
                        ),
                    ],
                    className="six columns",
                ),
            ],
            className="row",
        ),
        # Third row
        html.Div(
            [
                html.Div(
                    [
                        html.H5("Tabled candidate objective values"),
                        dash_table.DataTable(
                            id="table", columns=columns, data=data
                        ),
                    ],
                    className="row",
                ),
                html.Div(
                    [
                        html.H5("Tabled candidate best reachable values"),
                        dash_table.DataTable(id="table-best"),
                    ],
                    className="row",
                ),
            ],
            className="row",
        ),
    ]
)


@app.callback(
    [
        Output("table", "style_data_conditional"),
        Output("table-best", "style_data_conditional"),
        Output("spider-plots", "figure"),
        Output("value-paths", "figure"),
        Output("info", "children"),
    ],
    [Input("candidate-selection", "value")],
)
def highlight_table_row(candidate_index):
    global intermediate_points
    global intermediate_ranges
    global current_best_idx
    global previous_best

    if candidate_index == -1:
        raise PreventUpdate

    # just one solution
    if intermediate_points.ndim == 1:
        candidate_index = 0

    style = [
        {
            "if": {"row_index": candidate_index},
            "backgroundColor": "#0000FF",
            "color": "white",
        }
    ]

    if len(intermediate_points) == 0:
        zs = np.array([])
    else:
        zs = intermediate_points

    spider_plots = plotter.spider_plot_candidates(
        zs,
        names=objective_names,
        best=intermediate_ranges,
        previous=previous_best,
        selection=candidate_index,
    )

    value_paths = plotter.value_path_plot_candidates(
        zs, objective_names, selection=candidate_index
    )

    # original scale zs
    if zs.ndim == 1:
        orig_zs = scaler.inverse_transform(zs.reshape(1, -1))
    else:
        orig_zs = scaler.inverse_transform(zs)
    orig_zs = np.where(is_max, -orig_zs, orig_zs)
    info = ("Inventory turnover for currently selected candidate: {:,.2f}").format(
        254 / orig_zs[candidate_index, 4]
    )

    return style, style, spider_plots, value_paths, info


@app.callback(
    [
        Output("candidate-selection", "options"),
        Output("title", "children"),
        Output("table", "columns"),
        Output("table", "data"),
        Output("table-best", "columns"),
        Output("table-best", "data"),
        Output("candidate-selection", "value"),
        Output("final", "children")
    ],
    [Input("iterate-button", "n_clicks")],
    [State("candidate-selection", "value")],
)
def update_candidates(n_clicks, candidate_index):
    global intermediate_points
    global intermediate_ranges
    global current_best_idx
    global previous_best

    if enautilus.ith == 0:
        raise PreventUpdate

    if n_clicks == 0:
        raise PreventUpdate
    if n_clicks == 1:
        # first iteration, do not interact
        zs, best = enautilus.iterate()
    else:
        previous_best = intermediate_points[current_best_idx]
        current_best_idx = candidate_index
        x = enautilus.interact(
            intermediate_points[current_best_idx],
            intermediate_ranges[current_best_idx],
        )
        print(x)
        zs, best = enautilus.iterate()

    intermediate_points = zs
    intermediate_ranges = best

    default_candidate = 0

    options = [
        {"label": "Candidate {}".format(ind + 1), "value": val}
        for (ind, val) in enumerate(range(len(intermediate_points)))
    ]
    if enautilus.ith > 1:
        title = "E-NAUTILUS: Iterations left {}".format(enautilus.ith)
        final = ""

    elif enautilus.ith == 1:
        title = "Select the final solution."
        final = ""
    else:
        title = "Done. Final solution displayed"
        final = "Lot size: {:,.2f} Safety stock: {:,.2f}".format(x[0][48], x[0][49])
        default_candidate = current_best_idx
        intermediate_points = zs[current_best_idx]
        intermediate_ranges = best[current_best_idx]
        previous_best = None
        options = []

    columns, data = plotter.make_table(zs=intermediate_points, names=objective_names)

    columns_best, data_best = plotter.make_table(
        zs=intermediate_ranges, names=objective_names, row_name=["Best reachable"]
    )

    return (options, title, columns, data, columns_best, data_best, default_candidate, final)


def main():
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
