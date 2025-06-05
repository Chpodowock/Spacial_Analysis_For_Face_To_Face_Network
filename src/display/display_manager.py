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
from dash import dcc, html, Input, Output

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
            
            