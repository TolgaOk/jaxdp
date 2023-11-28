from typing import Union, Tuple, Optional, Dict, List, Any
from abc import abstractmethod
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Float, Array


import jaxdp
from jaxdp.mdp import MDP


class BaseRunner():

    value_expr: str = r"v"
    init_dist_expr: str = r"\rho"
    policy_expr: str = r"\pi"

    @property
    def policy_evaluation_expr(self) -> str:
        return (fr"${self.init_dist_expr}^T (\mathrm{{I}}_n -"
                fr" \gamma P^{{{self.policy_expr}_k}})^{{-1}}r^"
                fr"{{{self.policy_expr}_k}}$")

    @property
    def bellman_error_expr(self) -> str:
        return fr"$ \|{self.value_expr}_k - T({self.value_expr}_k)\|_\infty$"

    @property
    def value_delta_expr(self) -> str:
        return fr"$\|{self.value_expr}_{{k+1}} - {self.value_expr}_k\|_\infty$"

    @property
    def policy_norm_expr(self) -> str:
        return fr"$\|{self.policy_expr}_{{k+1}} - {self.policy_expr}_k \|_0$"

    def render_metrics(self, metrics: Dict[str, Float[Array, "..."]], **layout: Dict[str, Any]) -> "FigureWidget":
        import plotly.graph_objects as go
        from IPython.display import display, HTML

        # plotly.offline.init_notebook_mode()
        display(HTML(
            ('<script type="text/javascript" async '
             'src="https://cdnjs.cloudflare.com/ajax/libs'
             '/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'))
        )
        fig = go.FigureWidget()
        for name, values in metrics.items():
            legend_name = getattr(self, f"{name}_expr", name)
            fig.add_trace(
                go.Scatter(
                    y=values,
                    name=legend_name,
                    mode="lines+markers"
                )
            )
        fig.update_layout(
            template="simple_white",
            xaxis={"type": "log", "title": "steps", "showgrid": True},
            yaxis={"type": "linear", "title": None, "showgrid": True},
            title_x=0.5,
            font={"family": "Times New Roman", "size": 14},
            legend=dict(
                title=None,
                orientation="v",
                y=1,
                yanchor="top",
                x=1,
                xanchor="left",
                bgcolor="White",
                bordercolor="Gray",
                borderwidth=0.5,
            ),
            width=600,
            height=400,
        )
        fig.update_layout(layout)
        return fig
