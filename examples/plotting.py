from typing import Tuple, List, Dict, Union
from IPython.display import display


def vega_notebook_render(schema) -> None:
    display(
        {"application/vnd.vegalite.v5+json": schema},
        raw=True
    )


def line_plot(data: List[Dict[str, Union[int, float, bool]]],
              x_key: str,
              title: str
              ) -> None:
    names = list(data[0].keys())
    vega_notebook_render({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Multi seed line plot",
        "width": 500,
        "height": 300,
        "data": {"values": data},
        "params": [
            {
                "name": "selected_legend",
                "value": "y",
                "bind": {"input": "select", "options": names, "name": "Choose y axis:  "}
            },
            {
                "name": "bandwidth",
                "value": 0.01,
                "bind": {"input": "range", "min": 0.01, "max": 1, "step": 0.01, "name": "Smoothness:  "}
            },
            {
                "name": "show_region",
                "value": False,
                "bind": {"input": "checkbox", "name": "Show error bound:"}
            }
        ],
        "title": {
            "text": title,
            "fontSize": 20,
            "fontWeight": 300,
        },
        "transform": [
            {"fold": names, "as": ["legend_name", "legend_value"]},
            {"filter": {"field": "legend_name", "equal": {"expr": "selected_legend"}}},
        ],
        "encoding": {
            "x": {"field": x_key,
                  "type": "quantitative",
                  "axis": {
                      "offset": 10,
                      "titleFontSize": 14,
                      "titleFontWeight": 500,
                      "title": x_key,
                  }
                  },
        },
        "layer": [
            {
                "mark": {"type": "errorband",
                         "interpolate": "basis",
                         "extent": "iqr",
                         "borders": {
                             "opacity": 0.2,
                             "strokeDash": [1, 1],
                             "color": "gray"
                         }},
                "encoding": {
                    "y": {"field": "legend_value",
                          "type": "quantitative",
                          "axis": {"title": {"signal": "selected_legend"},
                                   "offset": 10,
                                   "titleFontSize": 14,
                                   "titleFontWeight": 500}
                          },
                    "opacity": {"condition": {"param": "show_region", "value": 0.7}, "value": 0},
                },
            },
            {
                "mark": {"type": "line", "interpolate": "basis"},
                "encoding": {
                    "y": {
                        "aggregate": "mean",
                        "field": "legend_value",
                    },
                    "color": {"field": "legend_name", "type": "nominal", "axis": {"title": "Traces"}},
                },
                "transform": [{"loess": "legend_value", "on": x_key, "bandwidth": {"signal": "bandwidth"}}]
            },
            {
                "mark": "point",
                "encoding": {
                    "y": {
                        "field": "legend_value",
                        "type": "quantitative"
                    },
                    "x": {
                        "field": x_key,
                        "type": "quantitative",
                        "title": x_key,
                    },
                    "color": {"field": "legend_name", "type": "nominal", "axis": {"title": "Traces"}},
                    "opacity": {"value": 0.9},
                },
            },
            {
                "mark": {"type": "line", "opacity": 1.0, "strokeDash": [2, 1], "interpolate": "basis"},
                "encoding": {
                    "y": {
                        "aggregate": "mean",
                        "field": "legend_value",
                    },
                    "color": {"field": "legend_name", "type": "nominal", "axis": {"title": "Traces"}},
                    "opacity": {"condition": {"param": "show_region", "value": 1.0}, "value": 0},
                },
            },
            {
                "mark": {"type": "line", "color": "gray", "strokeDash": [2, 3], "interpolate": "basis"},
                "encoding": {
                    "y": {
                        "aggregate": "max",
                        "field": "legend_value",
                    },
                    "color": {"field": "legend_name", "type": "nominal", "axis": {"title": "Traces"}},
                    "opacity": {"condition": {"param": "show_region", "value": 0.5}, "value": 0},
                },
            },
            {
                "mark": {"type": "line", "color": "gray", "strokeDash": [2, 3], "interpolate": "basis"},
                "encoding": {
                    "y": {
                        "aggregate": "min",
                        "field": "legend_value",
                    },
                    "color": {"field": "legend_name", "type": "nominal", "axis": {"title": "Traces"}},
                    "opacity": {"condition": {"param": "show_region", "value": 0.5}, "value": 0},
                },
            },
        ]
    })
