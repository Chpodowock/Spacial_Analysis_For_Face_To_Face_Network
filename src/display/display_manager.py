# === Standard Library ===
import math
from collections import Counter
from itertools import cycle, chain

# === Third-Party Libraries ===
import numpy as np
import pandas as pd
import networkx as nx

from shapely.geometry import Point
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
    "Point", "gaussian_filter",

    # Plotly and Dash
    "go", "px", "pio", "make_subplots",
    "dash", "dcc", "html",

    # Jupyter
    "display", "Markdown",

    # Project Visualizers
    "NetworkVisualizer", "PlotVisualizer", "MapVisualizer", "MatrixVisualizer"
]


class DisplayManager:
    def __init__(self, world):
        self.world = world

        # Initialize visualizer components
        self.plotter = PlotVisualizer(world)
        self.matrixer = MatrixVisualizer()
        self.networker = NetworkVisualizer()

        print("✅ DisplayManager initialized with Plotter, Matrixer, and Networker.")

        # Conditional mapper setup based on plan availability
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
        title="Plot Viewer",
        output_basename="plot_export",
        port=8050,
        debug=True
    ):
        """
        Launch a Dash app for interactive Plotly viewing and vector export.
        Supports multiple subplots by applying all axis zoom states found in relayoutData.
    
        Args:
            plot_func (callable): Function that returns a Plotly Figure.
            plot_args (tuple): Positional arguments for plot_func.
            plot_kwargs (dict): Keyword arguments for plot_func.
            title (str): Title to display in the Dash app.
            output_basename (str): Base filename for PDF/SVG exports.
            port (int): Port for Dash server.
            debug (bool): Whether to run Dash in debug mode.
        """
        plot_args = plot_args or ()
        plot_kwargs = plot_kwargs or {}
    
        app = dash.Dash(__name__)
        server = app.server
    
        def generate_figure():
            return plot_func(*plot_args, **plot_kwargs)
    
        app.layout = html.Div([
            html.H2(title),
            dcc.Graph(id="plot", figure=generate_figure()),
            html.Div([
                html.Button("Download PDF", id="btn-pdf", n_clicks=0),
                html.Button("Download SVG", id="btn-svg", n_clicks=0),
            ], style={"margin": "10px 0"}),
            html.Div(id="export-status", style={"marginBottom": "20px", "color": "green"}),
            dcc.Store(id="zoom-store")
        ])
    
        @app.callback(
            Output("zoom-store", "data"),
            Input("plot", "relayoutData"),
            prevent_initial_call=True
        )
        def store_zoom(relayout_data):
            return relayout_data
    
        @app.callback(
            Output("export-status", "children"),
            Input("btn-pdf", "n_clicks"),
            Input("btn-svg", "n_clicks"),
            State("zoom-store", "data"),
            prevent_initial_call=True
        )
        def export_figure(pdf_clicks, svg_clicks, zoom_data):
            ctx = callback_context
            if not ctx.triggered:
                return "No action"
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            fmt = "pdf" if button_id == "btn-pdf" else "svg"
            return _export(fmt, zoom_data)
    
        def _export(fmt, zoom_data):
            fig = generate_figure()
        
            if zoom_data:
                # Collect all axis range updates
                axis_ranges = {}
        
                for key, value in zoom_data.items():
                    if ".range[" in key:
                        axis, range_part = key.split(".range[")
                        index = int(range_part.strip("]"))
                        axis_entry = axis_ranges.setdefault(axis, [None, None])
                        axis_entry[index] = value
        
                # Apply all valid axis ranges
                for axis, (start, end) in axis_ranges.items():
                    if start is not None and end is not None:
                        fig.update_layout({axis: dict(range=[start, end])})
        
            filename = f"{output_basename}.{fmt}"
            try:
                pio.write_image(fig, filename, format=fmt)
                return f"✅ {fmt.upper()} saved as {filename}"
            except Exception as e:
                return f"❌ Export failed: {e}"

    
        app.run(debug=debug, port=port)
