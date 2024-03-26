
import plotly.graph_objects as go
import jax.numpy as jnp


def make_data_table(results, percentile: int = 25):
    table = []

    for name, info in results.items():
        for metric in info["metric"]._fields:
            values = getattr(info["metric"], metric)
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
                 "color": info["color"],
                 }
            )
    return table


def figure(title, table):
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

    fig = go.FigureWidget()
    for index, data in enumerate(table):
        buttons[metric_map[(data["metric"], data["env"])]]["args"][0]["visible"][(
            index * 3):(index * 3) + 3] = [True] * 3

        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["y_low"],
                mode="lines",
                showlegend=False,
                visible=False,
                legendgroup=data["algo"],
                opacity=0.75,
                line={"color": data["color"], "width": 1}
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["y_high"],
                mode="lines",
                showlegend=False,
                visible=False,
                fill="tonexty",
                legendgroup=data["algo"],
                opacity=0.75,
                line={"color": data["color"], "width": 1}
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["y_median"],
                mode="lines",
                name=data["algo"],
                showlegend=True,
                visible=False,
                legendgroup=data["algo"],
                marker={"color": data["color"]},
                line={"color": data["color"], "dash": "dash"}
            )
        )
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
