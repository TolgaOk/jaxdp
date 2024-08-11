
import plotly.graph_objects as go
import jax.numpy as jnp


def make_data_table(results, percentile: int = 25):
    table = []

    for name, info in results.items():
        for metric in info._fields:
            values = getattr(info, metric)
            percentiles = jnp.nanpercentile(
                values, q=jnp.array([percentile, 50, 100 - percentile]), axis=0)
            not_nan_indices = jnp.argwhere(1 - jnp.isnan(percentiles.sum(0))).flatten()

            table.append(
                {"algo": name,
                 "env": "",
                 "metric": metric,
                 "x": jnp.arange(values.shape[-1])[not_nan_indices].tolist(),
                 "y_low": percentiles[0][not_nan_indices].tolist(),
                 "y_median": percentiles[1][not_nan_indices].tolist(),
                 "y_high": percentiles[2][not_nan_indices].tolist(),
                 }
            )
    return table


def add_figure_data(data, color, fig, row=None, col=None, visible=True):
    fig.add_trace(
        go.Scatter(
            x=data["x"],
            y=data["y_low"],
            mode="lines",
            showlegend=False,
            visible=visible,
            legendgroup=data["algo"],
            opacity=0.75,
            line={"color": color, "width": 0}
        ),
        row=row,
        col=col
    )
    fig.add_trace(
        go.Scatter(
            x=data["x"],
            y=data["y_high"],
            mode="lines",
            showlegend=False,
            visible=visible,
            fill="tonexty",
            legendgroup=data["algo"],
            opacity=0.75,
            line={"color": color, "width": 0}
        ),
        row=row,
        col=col
    )
    fig.add_trace(
        go.Scatter(
            x=data["x"],
            y=data["y_median"],
            mode="lines",
            name=data["algo"],
            showlegend=True,
            visible=visible,
            legendgroup=data["algo"],
            marker={"color": color},
            line={"color": color}
        ),
        row=row,
        col=col
    )

    return fig


def figure(title, table, colors):
    fig = go.FigureWidget()
    metrics = set([(item["metric"], item["env"]) for item in table])
    metric_map = {name: index for index, name in enumerate(metrics)}
    buttons = [
        {
            "label": metric,
            "method": "update",
            "args": [{"visible": [False] * (len(table) * 3)},
                     {"yaxis": {"title": metric.replace("_", " "), "showgrid": True}},
                     {"title": {"text": env, "x": 0.5}}]
        }
        for metric, env in metric_map.keys()
    ]

    for index, data in enumerate(table):
        buttons[metric_map[(data["metric"], data["env"])]]["args"][0]["visible"][(
            index * 3):(index * 3) + 3] = [True] * 3
        add_figure_data(data, colors[data["algo"]], fig, visible=False)
    fig.update_layout(
        template="simple_white",
        width=700,
        height=500,
        title={"text": title, "x": 0.5},
        xaxis={"title": "Iteration", "showgrid": True},
        yaxis={"title": "", "showgrid": True},
        showlegend=True,
        updatemenus=[go.layout.Updatemenu(
            active=1,
            buttons=buttons,
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0,
            xanchor="left",
            y=1.2,
            yanchor="top")
        ]
    )
    return fig
