import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple
from plotly.graph_objs._figure import Figure
import pandas as pd


class Plotter:
    """A class to contain different plotters.

    Args:
        nadir (np.ndarray): The nadir, or worst, values of the problem's
        objective functions.
        ideal (np.ndarray): The ideal, or best, values of the problem's
        objective functions.
        scaler (sklean.preprocessing,data.MinMaxScaler): A scaler to scale the
        data back to the original values.
        is_max (List[bool]): An array to indicate whether an objective is to be
        maximized or minimized.

    Note:
        The nadir and ideal are assumed to be already normalized.
    """

    def __init__(
        self,
        nadir: np.ndarray,
        ideal: np.ndarray,
        scaler: "sklearn.preprocessing.data.MinMaxScaler",
        is_max: List[bool],
    ):
        self.nadir = nadir
        self.ideal = ideal
        self.scaler = scaler
        self.is_max = is_max

    def spider_plot_candidates(
        self,
        zs: np.ndarray,
        names: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        best: Optional[np.ndarray] = None,
        previous: Optional[np.ndarray] = None,
        dims: Optional[Tuple[int, int]] = (2, 2),
    ) -> Figure:
        """Plots multiple solution candiates with the best reachable values and
        the previous candidate chosen shown, if defined.

        Args:
            zs (np.ndarray): A 2D array with each candidate on its' rows.
            names (List[str], optional): A list of the objective names.
            labels (List[str], optioanl): A list of the solution names.
            best (np.ndarray, optional): A 2D array with the best reachable
            solution from each candidate on its' rows.
            previous (np.ndarray, optional): An array with the previously
            chosen candidate.
            dims (Tuple[int, int], optional): How many solutions should be
            displayed on each row and column

        """
        if zs.size == 0:
            fig = make_subplots(specs=[[{"type": "polar"}]])
            fig["layout"]["autosize"] = True
            return {}

        if zs.ndim == 1:
            # reshape single solution to function as 2D array
            zs = zs.reshape(1, -1)

        zs_original = self.scaler.inverse_transform(zs)
        zs_original = np.where(self.is_max, -zs_original, zs_original)
        zs = np.where(self.is_max, -zs, zs)

        if names is None:
            names = ["Obj {}".format(i + 1) for i in range(zs.shape[1])]
        names = [
            "{} ({})".format(name, val)
            for (val, name) in zip(
                ["max" if m is True else "min" for m in self.is_max], names
            )
        ]

        if best is not None and best.ndim == 1:
            best = best.reshape(1, -1)
        if best is not None:
            best_original = self.scaler.inverse_transform(best)
            best_original = np.where(
                self.is_max, -best_original, best_original
            )
            best = np.where(self.is_max, -best, best)

        if previous is not None and previous.ndim == 1:
            previous = previous.reshape(1, -1)
        if previous is not None:
            previous_original = self.scaler.inverse_transform(previous)
            previous_original = np.where(
                self.is_max, -previous_original, previous_original
            )
            previous = np.where(self.is_max, -previous, previous)

        rows, cols = dims

        if labels is None:
            titles = ["Candidate {}".format(i + 1) for i in range(zs.shape[1])]
        else:
            titles = labels

        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"type": "polar"}] * cols] * rows,
            subplot_titles=titles,
        )

        fig["layout"]["width"] = cols * 400
        fig["layout"]["height"] = rows * 400
        fig["layout"]["autosize"] = True
        polars = ["polar"] + [
            "polar{}".format(i + 1) for i in range(1, len(zs))
        ]

        dicts = dict(
            zip(
                polars,
                # TODO range from scaler
                [dict(radialaxis=dict(visible=False, range=[-1, 1]))]
                * len(polars),
            )
        )

        fig.update_layout(
            **dicts,
            title=go.layout.Title(
                text=(
                    "Candidate solutions in Blue,\nbest reachable values in "
                    "red,\nprevious solution in green."
                ),
                xref="container",
                x=0.5,
                xanchor="center",
            )
        )

        def index_generator():
            for i in range(1, rows + 1):
                for j in range(1, cols + 1):
                    yield i, j

        gen = index_generator()

        for (z_i, z) in enumerate(zs):
            try:
                i, j = next(gen)
            except StopIteration:
                break

            if best is not None:
                fig.add_trace(
                    go.Scatterpolar(
                        r=best[z_i],
                        opacity=0.25,
                        theta=names,
                        name="best",
                        fillcolor="red",
                        fill="toself",
                        showlegend=False,
                        hovertext=best_original[z_i],
                        hoverinfo="name+theta+text",
                        line={"color": "red"},
                    ),
                    row=i,
                    col=j,
                )

            if previous is not None:
                fig.add_trace(
                    go.Scatterpolar(
                        r=previous[0],
                        opacity=1,
                        theta=names,
                        name="previous",
                        fillcolor="green",
                        fill="none",
                        showlegend=False,
                        hovertext=previous_original,
                        hoverinfo="name+theta+text",
                        line={"dash": "dot", "color": "green"},
                    ),
                    row=i,
                    col=j,
                )

            fig.add_trace(
                go.Scatterpolar(
                    r=z,
                    opacity=1,
                    theta=names,
                    showlegend=False,
                    name="Candidate {}".format(z_i + 1),
                    fill="none",
                    fillcolor="blue",
                    hovertext=zs_original[z_i],
                    hoverinfo="name+theta+text",
                    line={"color": "blue"},
                ),
                row=i,
                col=j,
            )

        for annotation in fig["layout"]["annotations"]:
            annotation["yshift"] = 20

        return fig

    def value_path_plot_candidates(
        self,
        zs: np.ndarray,
        names: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> Figure:
        """Plots multiple solution candidates as an parallel axis plot.

        Args:
            zs (np.ndarray): A 2D array with each candidate on its' rows.
            names (List[str], optional): A list of the objective names.
        """
        if zs.ndim == 1:
            zs = zs.reshape(1, -1)
        zs_original = self.scaler.inverse_transform(zs)
        zs_original = np.where(self.is_max, -zs_original, zs_original)

        nadir_original = self.scaler.inverse_transform(
            self.nadir.reshape(1, -1)
        )[0]
        nadir_original = np.where(self.is_max, -nadir_original, nadir_original)

        ideal_original = self.scaler.inverse_transform(
            self.ideal.reshape(1, -1)
        )[0]
        ideal_original = np.where(self.is_max, -ideal_original, ideal_original)

        if names is None:
            names = ["Obj {}".format(i + 1) for i in range(zs.shape[1])]

        names = [
            "{} ({})".format(name, val)
            for (val, name) in zip(
                ["max" if m is True else "min" for m in self.is_max], names
            )
        ]

        rows = [list(row) for row in zs_original.T]
        fig = go.Figure(
            data=go.Parcoords(
                dimensions=list(
                    # stupid hack to "label" each solution on the first axis
                    [
                        dict(
                            range=[1, len(zs_original)],
                            label="Candidate",
                            values=list(range(1, len(zs_original) + 1)),
                            tickvals=list(range(1, len(zs_original) + 1)),
                        )
                    ]
                    + [
                        dict(range=[low, up], label=name, values=vals)
                        for (low, up, name, vals) in zip(
                            nadir_original, ideal_original, names, rows
                        )
                    ]
                )
            )
        )

        return fig

    def make_table(
        self,
        zs: np.ndarray,
        names: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        row_name: Optional[List[str]] = None,
    ):
        """Make a dash table out of a numpy array

        Args:
            zs (np.ndarray): A 2D array with the solutions to table on each
            row.
            names (List[str], optional): List of the objective_names

        """
        if zs.ndim == 1:
            zs = zs.reshape(1, -1)
        zs_original = self.scaler.inverse_transform(zs)
        zs_original = np.where(self.is_max, -zs_original, zs_original)

        if names is None:
            names = [
                "Obj {}".format(i + 1) for i in range(zs_original.shape[1])
            ]
        if row_name is None:
            row_name = ["Candidate"]

        names = [
            "{} ({})".format(name, val)
            for (val, name) in zip(
                ["max" if m is True else "min" for m in self.is_max], names
            )
        ]
        names = row_name + names

        labeled = np.zeros((zs_original.shape[0], zs_original.shape[1] + 1))
        labeled[:, 0] = np.linspace(
            1, len(zs_original), num=len(zs_original), dtype=int
        )
        labeled[:, 1:] = zs_original
        df = pd.DataFrame(
            data=labeled, index=list(range(len(labeled))), columns=names
        )
        if labels:
            df["Candidate"] = labels

        columns = [{"name": i, "id": i} for i in df.columns]
        data = df.to_dict("records")

        return columns, data
