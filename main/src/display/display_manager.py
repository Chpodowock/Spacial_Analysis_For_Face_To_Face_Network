# === Standard Library ===
import math
from collections import Counter
from itertools import cycle, chain

# === Third-Party Libraries ===
import numpy as np
import pandas as pd
import networkx as nx

from shapely.geometry import Point, Polygon
from scipy.ndimage import gaussian_filter

# === Plotly and Dash ===
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback_context

# === Jupyter Utilities ===
from IPython.display import display, Markdown

# === Project Visualizers ===
from network_visualizer import NetworkVisualizer
from plot_visualizer import PlotVisualizer
from map_visualizer import MapVisualizer
from matrix_visualizer import MatrixVisualizer


__all__ = [
    # Standard Library
    "math", "Counter", "cycle", "chain",

    # Third-Party Libraries
    "np", "pd", "nx",
    "Point", "gaussian_filter", "Polygon",

    # Plotly and Dash
    "go", "px", "pio", "make_subplots",
    "dash", "dcc", "html",

    # Jupyter
    "display", "Markdown",

    # Project Visualizers
    "NetworkVisualizer", "PlotVisualizer", "MapVisualizer", "MatrixVisualizer",
]



class PlotStyler:
    def __init__(self, width=1000, height=600, font_size=12, title_size=16, axis_title_size=14, tick_size=12):
        self.width = width
        self.height = height
        self.font_size = font_size
        self.title_size = title_size
        self.axis_title_size = axis_title_size
        self.tick_size = tick_size

    def apply(self, fig, margin=dict(l=60, r=20, t=60, b=50)):
        fig.update_layout(
            autosize=False,
            width=self.width,
            height=self.height,
            margin=margin,
            font=dict(size=self.font_size),
            title_font=dict(size=self.title_size),
            xaxis=dict(
                title_font=dict(size=self.axis_title_size),
                tickfont=dict(size=self.tick_size)
            ),
            yaxis=dict(
                title_font=dict(size=self.axis_title_size),
                tickfont=dict(size=self.tick_size)
            ),
            template="plotly_white"
        )
        return fig


class DisplayManager:
    def __init__(self, world ,width=1000, height=600, font_size=12, title_size=16, axis_title_size=14, tick_size=12):
        self.world = world

        # Initialize visualizer components
        self.plotter = PlotVisualizer(world)
        self.matrixer = MatrixVisualizer()
        self.networker = NetworkVisualizer()
        self.styler = PlotStyler(width, height, font_size, title_size, axis_title_size, tick_size)
        
        print("✅ DisplayManager initialized with Plotter, Matrixer, and Networker.")

        if hasattr(world, 'plans') and world.plans:
            self.mapper = MapVisualizer(world)
            print(f"✅ Mapper initialized with {len(world.plans)} plans.")
        else:
            self.mapper = None
            print("⚠️ No plans found. Mapper not initialized.")

    def run_plotly_dash_export_app(
        self,
        plot_func,
        plot_args=None,
        plot_kwargs=None,
        output_basename="plot_export",
        port=8050,
        debug=True,
        styler=None
    ):
        plot_args = plot_args or ()
        plot_kwargs = plot_kwargs or {}
    
        app = dash.Dash(__name__)
        server = app.server
    
        # Generate figure once and store as dict
        initial_figure = plot_func(*plot_args, **plot_kwargs)
        if styler:
            styler.apply(initial_figure)
    
        app.layout = html.Div([
            dcc.Graph(id="plot"),
            dcc.Store(id="initial-figure", data=initial_figure.to_dict()),
            dcc.Store(id="zoom-store"),
            html.Div([
                html.Button("Download PDF", id="btn-pdf", n_clicks=0),
                html.Button("Download SVG", id="btn-svg", n_clicks=0),
            ], style={"margin": "10px 0"}),
            html.Div(id="export-status", style={"marginBottom": "20px", "color": "green"}),
        ])
    
        @app.callback(
            Output("zoom-store", "data"),
            Input("plot", "relayoutData"),
            prevent_initial_call=True
        )
        def store_zoom(relayout_data):
            return relayout_data
    
        @app.callback(
            Output("plot", "figure"),
            Input("initial-figure", "data"),
            State("zoom-store", "data")
        )
        def update_figure(initial_fig_dict, zoom_data):
            fig = go.Figure(initial_fig_dict)
            if zoom_data:
                axis_ranges = {}
                for key, value in zoom_data.items():
                    if ".range[" in key:
                        axis, index = key.split(".range[")
                        index = int(index.strip("]"))
                        axis_entry = axis_ranges.setdefault(axis, [None, None])
                        axis_entry[index] = value
                for axis, (start, end) in axis_ranges.items():
                    if start is not None and end is not None:
                        fig.update_layout({axis: dict(range=[start, end])})
            return fig
    
        @app.callback(
            Output("export-status", "children"),
            Input("btn-pdf", "n_clicks"),
            Input("btn-svg", "n_clicks"),
            State("zoom-store", "data"),
            State("initial-figure", "data"),
            prevent_initial_call=True
        )
        def export_figure(pdf_clicks, svg_clicks, zoom_data, fig_dict):
            ctx = callback_context
            if not ctx.triggered:
                return "No action triggered"
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            fmt = "pdf" if button_id == "btn-pdf" else "svg"
            return _export(fmt, zoom_data, fig_dict)
    
        def _export(fmt, zoom_data, fig_dict):
            fig = go.Figure(fig_dict)
            if zoom_data:
                axis_ranges = {}
                for key, value in zoom_data.items():
                    if ".range[" in key:
                        axis, index = key.split(".range[")
                        index = int(index.strip("]"))
                        axis_entry = axis_ranges.setdefault(axis, [None, None])
                        axis_entry[index] = value
                for axis, (start, end) in axis_ranges.items():
                    if start is not None and end is not None:
                        fig.update_layout({axis: dict(range=[start, end])})
    
            if styler:
                styler.apply(fig)
    
            filename = f"{output_basename}.{fmt}"
            try:
                pio.write_image(fig, filename, format=fmt)
                return f"✅ {fmt.upper()} saved as {filename}"
            except Exception as e:
                return f"❌ Export failed: {e}"
    
        app.run(debug=debug, port=port)
