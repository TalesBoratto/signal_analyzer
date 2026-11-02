from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def extract_mat_sensor_names(value, n_sensors: int) -> list[str] | None:
    """
    Extrai nomes de sensores de formatos comuns do MATLAB:
    cell arrays de strings, vetores de strings e matrizes de caracteres.

    Retorna None quando não encontra um nome por sensor.
    """
    def to_text(item) -> str:
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace").strip()

        if isinstance(item, str):
            return item.strip()

        array = np.asarray(item)

        if array.dtype.kind in {"U", "S"}:
            parts = [
                element.decode("utf-8", errors="replace")
                if isinstance(element, bytes)
                else str(element)
                for element in array.ravel()
            ]
            return "".join(parts).strip()

        if array.size == 1 and array.dtype.kind == "O":
            return to_text(array.item())

        return ""

    labels = np.asarray(value)

    if labels.dtype.kind == "O":
        names = [to_text(item) for item in labels.ravel()]

    elif labels.dtype.kind in {"U", "S"}:
        squeezed = labels.squeeze()

        # Vetor com uma string completa por sensor.
        if squeezed.size == n_sensors:
            names = [
                to_text(item)
                for item in squeezed.ravel()
            ]

        # Matriz de caracteres: uma linha por sensor.
        elif labels.ndim == 2 and labels.shape[0] == n_sensors:
            names = [to_text(row) for row in labels]

        elif n_sensors == 1:
            names = [to_text(labels)]

        else:
            return None

    else:
        return None

    if len(names) != n_sensors or not all(names):
        return None

    return names


def line_figure(x, y, title, x_label, y_label, color):
    fig = go.Figure(
        go.Scattergl(
            x=x,
            y=y,
            mode="lines",
            line={"color": color, "width": 1},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=380,
        margin={"l": 20, "r": 20, "t": 60, "b": 40},
        dragmode="zoom",
    )

    return fig


def error_figure(
    points,
    unit: str,
    error_type: str = "absolute",
):
    """
    Compara a aceleração com os erros dos dois métodos.

    absolute:
        abs(referência - estimativa), no eixo da aceleração.

    signed:
        referência - estimativa, no eixo da aceleração.

    relative:
        (referência - estimativa) / referência,
        no eixo secundário, sem percentual.
    """
    valid_types = {"absolute", "signed", "relative"}

    if error_type not in valid_types:
        raise ValueError(
            f"error_type deve ser um destes valores: {valid_types}"
        )

    use_secondary_axis = error_type == "relative"

    fig = make_subplots(
        specs=[[{"secondary_y": use_secondary_axis}]]
    )

    fig.add_trace(
        go.Scattergl(
            x=points["time_s"],
            y=points["acc_dc"],
            name="Acceleration after DC removal",
            mode="lines",
            line={
                "color": "black",
                "width": 1,
            },
            hovertemplate=(
                "Time: %{x:.4f} s"
                "<br>Acceleration: %{y:.6g}"
                "<extra>%{fullData.name}</extra>"
            ),
        ),
        secondary_y=False,
    )

    methods = {
        "fft": {
            "label": "FFT differentiation",
            "color": "royalblue",
        },
        "time": {
            "label": "Finite differences",
            "color": "darkorange",
        },
    }

    for method, settings in methods.items():
        if error_type == "absolute":
            values = np.abs(points[f"error_{method}"].to_numpy())
            error_label = "Absolute error"

        elif error_type == "signed":
            values = points[f"error_{method}"].to_numpy()
            error_label = "Signed error"

        else:
            values = points[f"relative_error_{method}"].to_numpy()
            error_label = "Relative error"

        fig.add_trace(
            go.Scattergl(
                x=points["time_s"],
                y=values,
                name=f"{error_label} — {settings['label']}",
                mode="lines",
                connectgaps=False,
                line={
                    "color": settings["color"],
                    "width": 1,
                },
                hovertemplate=(
                    "Time: %{x:.4f} s"
                    "<br>Error: %{y:.6g}"
                    "<extra>%{fullData.name}</extra>"
                ),
            ),
            secondary_y=use_secondary_axis,
        )

    fig.update_xaxes(title_text="Time [s]")

    fig.update_yaxes(
        title_text=(
            f"Acceleration [{unit}]"
            if use_secondary_axis
            else f"Acceleration and error [{unit}]"
        ),
        secondary_y=False,
    )

    if use_secondary_axis:
        fig.update_yaxes(
            title_text="Relative error [dimensionless]",
            showgrid=False,
            secondary_y=True,
        )

    titles = {
        "absolute": "Acceleration and absolute reconstruction errors",
        "signed": "Acceleration and signed reconstruction errors",
        "relative": "Acceleration and pointwise relative errors",
    }

    fig.update_layout(
        title=titles[error_type],
        height=500,
        dragmode="zoom",
        hovermode="x",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        margin={
            "l": 30,
            "r": 30,
            "t": 110,
            "b": 40,
        },
    )

    return fig

