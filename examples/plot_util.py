import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, HTML


display(
    HTML(
        """<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>"""
    )
)


def add_traces(fig, name, df, color, row, col, showlegend):
    x_axis = df.index.get_level_values("STEP").values

    non_nan = df.low.notna().values

    fig.add_trace(
        go.Scatter(
            x=x_axis[non_nan],
            y=df.low.values[non_nan],
            mode="lines",
            showlegend=False,
            visible=True,
            legendgroup=name,
            opacity=0.75,
            line={"color": color, "width": 0}
        ),
        row=row,
        col=col
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis[non_nan],
            y=df.high.values[non_nan],
            mode="lines",
            showlegend=False,
            visible=True,
            fill="tonexty",
            legendgroup=name,
            opacity=0.75,
            line={"color": color, "width": 0}
        ),
        row=row,
        col=col
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis[non_nan],
            y=df.med.values[non_nan],
            mode="lines",
            name=name,
            showlegend=showlegend,
            visible=True,
            legendgroup=name,
            marker={"color": color},
            line={"color": color}
        ),
        row=row,
        col=col
    )


def make_figure(df):
    algos = df.index.get_level_values("ALG").unique()

    fig = go.FigureWidget()
    metrics = df.columns.get_level_values("METRIC").unique().values
    fig = make_subplots(rows=1, cols=len(metrics))

    for algo in algos:
        for grid_x, metric in enumerate(metrics):
            add_traces(fig, algo, getattr(df.loc[algo], metric),
                       "blue", row=1, col=grid_x + 1, showlegend=(grid_x == 0))
            fig.update_yaxes(
                title=metric.replace("_", " ").capitalize(),
                type="linear",
                row=1,
                col=1 + grid_x,
            )

    fig.update_yaxes(
        type="linear",
        exponentformat="power",
        showline=True,
        linecolor="gray",
        linewidth=2,
        mirror=True,
        gridcolor="white",
        gridwidth=3,
    )
    fig.update_xaxes(
        type="linear",
        exponentformat="power",
        showline=True,
        linecolor="gray",
        linewidth=2,
        mirror=True,
        title="Iteration",
        gridcolor="white",
        gridwidth=3,
    )
    fig.update_layout(
        legend={
            "x": 0.025,
            "y": 1.25,
            "font": {"size": 25},
            "orientation": "h",
            "visible": True,
        },
        font=dict(size=15, color="black"),
        plot_bgcolor="#f5f6f7",
        width=550 * len(metrics),
        height=500,
    )
    return fig
