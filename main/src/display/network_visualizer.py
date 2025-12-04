from display_manager import dash, nx, go, np, dcc, html, Input, Output, px, pd, make_subplots
 

class NetworkVisualizer:

    def __init__(self):
        self.app = dash.Dash(__name__)
        self.graph = None
        self.layout_options = ['spring', 'circular', 'kamada_kawai']

    def _get_layout(self, G, layout_name='spring'):
        if layout_name == 'circular':
            return nx.circular_layout(G)
        elif layout_name == 'kamada_kawai':
            return nx.kamada_kawai_layout(G)
        return nx.spring_layout(G, seed=42)
    
    def network_display(self, graph, experiment_id=None, layout_options=None, title="Area Transition Network", **kwargs):
        import numpy as np
        import networkx as nx
        import plotly.graph_objects as go
    
        if layout_options is None:
            layout_options = ['spring', 'circular', 'kamada_kawai']
    
        def compute_layout(G, layout_name):
            layouts = {
                "spring": lambda G: nx.spring_layout(G, seed=42),
                "circular": nx.circular_layout,
                "kamada_kawai": nx.kamada_kawai_layout
            }
            return layouts.get(layout_name, layouts["spring"])(G)
    
        def build_figure(G, layout_name="spring", min_weight=0):
            if not isinstance(G, nx.DiGraph) or G.number_of_nodes() == 0:
                raise ValueError("Invalid or empty directed graph.")
    
            pos = compute_layout(G, layout_name)
    
            # === Nodes ===
            node_x, node_y, node_text, node_colors, node_labels = [], [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                name = G.nodes[node].get("name", str(node))
                node_x.append(x)
                node_y.append(y)
                node_labels.append(name)
                in_deg = G.in_degree(node, weight='weight')
                out_deg = G.out_degree(node, weight='weight')
                node_text.append(f"<b>{name}</b><br>In: {in_deg}<br>Out: {out_deg}")
                node_colors.append(G.nodes[node].get('color', 'lightgray'))
    
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=node_labels,
                textposition="top center",
                hovertext=node_text,
                hoverinfo='text',
                marker=dict(size=40, color=node_colors, line=dict(width=2)),
                showlegend=False
            )
    
            # === Edges ===
            edge_traces = []
            annotations = []
            weights = [data.get("weight", 1) for _, _, data in G.edges(data=True)]
            max_w = max(weights) if weights else 1
    
            def scale_weight(w): return 1 + 5 * w / max_w
    
            def bezier_control(x0, y0, x1, y1, curvature=0.2):
                dx, dy = x1 - x0, y1 - y0
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                norm = np.hypot(dx, dy)
                return (mx - dy * curvature, my + dx * curvature) if norm else (mx, my)
    
            for src, dst, data in G.edges(data=True):
                weight = data.get("weight", 1)
                if weight < min_weight:
                    continue
    
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                cx, cy = bezier_control(x0, y0, x1, y1)
                width = scale_weight(weight)
    
                t = np.linspace(0, 1, 20)
                bez_x = (1 - t)**2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
                bez_y = (1 - t)**2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
    
                edge_traces.append(go.Scatter(
                    x=bez_x,
                    y=bez_y,
                    mode='lines',
                    line=dict(width=width, color='gray'),
                    hoverinfo='text',
                    text=f"{G.nodes[src].get('name', src)} → {G.nodes[dst].get('name', dst)}<br>Weight: {weight}",
                    opacity=0.6,
                    showlegend=False
                ))
    
                annotations.append(dict(
                    ax=bez_x[-2], ay=bez_y[-2],
                    x=bez_x[-1], y=bez_y[-1],
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True, arrowhead=2,
                    arrowsize=2, arrowwidth=width,
                    arrowcolor='gray', opacity=0.9
                ))
    
            fig = go.Figure(data=edge_traces + [node_trace])
            fig.update_layout(
                title=title,
                annotations=annotations,
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor="white",
                margin=dict(l=20, r=20, t=40, b=20),
                height=600
            )
            return fig
    
        return build_figure(graph)

    
        # --- Dash App ---
        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H1(title),
            html.Div([
                html.Label("Layout:"),
                dcc.Dropdown(
                    id='layout-dropdown',
                    options=[{'label': l.title(), 'value': l} for l in layout_options],
                    value='spring',
                    clearable=False,
                    style={"width": "200px"}
                ),
                html.Label("Min Edge Weight:"),
                dcc.Slider(
                    id='weight-slider',
                    min=0, max=10, step=1,
                    value=0,
                    marks={i: str(i) for i in range(11)},
                    tooltip={"placement": "bottom"}
                )
            ], style={'marginBottom': 20}),
            dcc.Graph(id='network-graph')
        ])
    
        @app.callback(
            Output('network-graph', 'figure'),
            Input('layout-dropdown', 'value'),
            Input('weight-slider', 'value')
        )
        def update_graph(layout_name, min_weight):
            return build_figure(graph, layout_name, min_weight)
    
        app.run(jupyter_mode="inline", debug=False)

    
                
    def plot_area_transition_sankey(self, G, world):
        """
        Plot a Sankey diagram of area transitions using area names as node labels.
        
        Args:
            G (networkx.DiGraph): Transition graph.
            world (WorldModel): Reference to world containing Area objects.
        """
        # Step 1: Extract node labels and assign indices
        area_ids = list(G.nodes)
        label_to_index = {area_id: i for i, area_id in enumerate(area_ids)}
    
        # Step 2: Resolve area names and colors
        area_labels = []
        node_colors = []
        for area_id in area_ids:
            area = world.areas.get(area_id)
            label = getattr(area, "name", str(area_id)) if area else str(area_id)
            color = getattr(area, "color", "lightgray") if area else "lightgray"
            area_labels.append(label)
            node_colors.append(color)
    
        # Step 3: Extract transitions
        sources = []
        targets = []
        values = []
        for src, dst, data in G.edges(data=True):
            if src in label_to_index and dst in label_to_index:
                sources.append(label_to_index[src])
                targets.append(label_to_index[dst])
                values.append(data.get("weight", 1))
    
        # Step 4: Build Sankey figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=area_labels,
                color=node_colors
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(100, 100, 200, 0.4)"
            )
        )])
    
        fig.update_layout(title_text="Area Transition Sankey Diagram", font_size=14)
        return fig



                
    def plot_temporal_area_graph_slider(self, temporal_graphs, layout_name='spring', min_weight=0):
        """
        Visualize temporal area transition graphs with a slider using curved edges and
        node size proportional to in+out degree. Node labels and edge tooltips use area names.
        Edges are colored based on the source area color.
    
        Args:
            temporal_graphs: List of (label, nx.DiGraph)
            layout_name: Layout algorithm name ('spring', 'circular', etc.)
            min_weight: Minimum edge weight to include
        """
        import numpy as np
        import networkx as nx
        import plotly.graph_objects as go
    
        def get_layout(G, layout_name):
            if layout_name == 'spring':
                return nx.spring_layout(G, seed=42)
            elif layout_name == 'circular':
                return nx.circular_layout(G)
            elif layout_name == 'kamada_kawai':
                return nx.kamada_kawai_layout(G)
            else:
                raise ValueError(f"Unsupported layout: {layout_name}")
    
        def scale_weight(w, max_w):
            return 1 + 5 * w / max_w if max_w else 1
    
        def bezier_control_point(x0, y0, x1, y1, curvature=0.2):
            dx, dy = x1 - x0, y1 - y0
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            norm = np.sqrt(dx**2 + dy**2)
            if norm == 0:
                return mx, my
            off_x = -dy * curvature
            off_y = dx * curvature
            return mx + off_x, my + off_y
    
        base_graph = temporal_graphs[0][1]
        pos = get_layout(base_graph, layout_name)
        frames = []
    
        for label, G in temporal_graphs:
            edge_traces, annotations = [], []
            weights = [data.get("weight", 1) for _, _, data in G.edges(data=True)]
            max_w = max(weights) if weights else 1
    
            for src, dst, data in G.edges(data=True):
                weight = data.get("weight", 1)
                if weight < min_weight:
                    continue
    
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                cx, cy = bezier_control_point(x0, y0, x1, y1)
                width = scale_weight(weight, max_w)
    
                t_vals = np.linspace(0, 1, 20)
                bez_x = (1 - t_vals)**2 * x0 + 2 * (1 - t_vals) * t_vals * cx + t_vals**2 * x1
                bez_y = (1 - t_vals)**2 * y0 + 2 * (1 - t_vals) * t_vals * cy + t_vals**2 * y1
    
                # Use area name labels and source color
                src_area = self.areas.get(src)
                dst_area = self.areas.get(dst)
                src_label = getattr(src_area, "name", str(src))
                dst_label = getattr(dst_area, "name", str(dst))
                edge_color = getattr(src_area, "color", "gray")
    
                edge_traces.append(go.Scatter(
                    x=bez_x,
                    y=bez_y,
                    mode='lines',
                    line=dict(width=width, color=edge_color),
                    hoverinfo='text',
                    text=f"{src_label} → {dst_label}<br>Weight: {weight}",
                    opacity=0.6,
                    showlegend=False
                ))
    
                annotations.append(dict(
                    ax=bez_x[-2], ay=bez_y[-2],
                    x=bez_x[-1], y=bez_y[-1],
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=2,
                    arrowwidth=width,
                    arrowcolor=edge_color,
                    opacity=0.9
                ))
    
            node_x, node_y, node_sizes, node_texts, node_colors = [], [], [], [], []
    
            for node in G.nodes():
                x, y = pos[node]
                area = self.areas.get(node)
                label = getattr(area, "name", str(node))
                color = getattr(area, "color", "lightgray")
                in_deg = G.in_degree(node, weight='weight')
                out_deg = G.out_degree(node, weight='weight')
                total = in_deg + out_deg
                size = 10 + total * 3
    
                node_x.append(x)
                node_y.append(y)
                node_sizes.append(size)
                node_colors.append(color)
                node_texts.append(f"<b>{label}</b><br>In: {in_deg}<br>Out: {out_deg}")
    
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=node_texts,
                textposition="top center",
                hoverinfo='text',
                marker=dict(size=node_sizes, color=node_colors, line=dict(width=2)),
                showlegend=False
            )
    
            frame = go.Frame(data=edge_traces + [node_trace], name=label, layout=go.Layout(annotations=annotations))
            frames.append(frame)
    
        # Base figure
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="📊 Temporal Area Transitions",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                updatemenus=[dict(type="buttons", showactive=False,
                    buttons=[dict(label="▶ Play", method="animate", args=[None])])]
            ),
            frames=frames
        )
    
        # Slider control
        fig.update_layout(
            sliders=[dict(
                steps=[dict(method="animate", args=[[f.name], {"frame": {"duration": 700, "redraw": True}}], label=f.name)
                       for f in frames],
                transition=dict(duration=0),
                x=0, y=0,
                currentvalue=dict(font=dict(size=16), prefix="Period: ", visible=True),
                len=1.0
            )],
            height=650,
            plot_bgcolor="white"
        )
    
        return fig




    def plot_node_edge_distribution_per_area(self, world, areas=None):
        """
        For each area, plot (log-log) number of edges vs number of nodes per time window,
        with theoretical bounds: N/2 (linear) and N(N-1)/2 (complete).
    
        Args:
            world: WorldModel
            areas: optional list of area IDs to include (default: all)
        """
        areas = areas or list(world.areas.keys())
        n_cols = 2
        n_rows = int(np.ceil(len(areas) / n_cols))
    
        fig = make_subplots(rows=n_rows, cols=n_cols,
                            subplot_titles=[world.areas[aid].name for aid in areas],
                            horizontal_spacing=0.12, vertical_spacing=0.15)
    
        for idx, area_id in enumerate(areas):
            area = world.areas[area_id]
            graphs = area.temporal_agent_graphs
            row = idx // n_cols + 1
            col = idx % n_cols + 1
    
            # Data points
            x_vals = []
            y_vals = []
            time_labels = []
    
            for time, G in graphs:
                n = G.number_of_nodes()
                m = G.number_of_edges()
                if n > 0 and m > 0:
                    x_vals.append(n)
                    y_vals.append(m)
                    time_labels.append(str(time))
    
            # Plot data points
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='markers',
                text=time_labels, name=f"{area.name}",
                marker=dict(size=7, opacity=0.7, color=area.color),
                showlegend=False
            ), row=row, col=col)
                
            # Plot lower bound (N/2) and upper bound (N(N-1)/2)
            n_range = np.unique(x_vals)
            lower_bound = n_range / 2
            upper_bound = n_range * (n_range - 1) / 2
    
            fig.add_trace(go.Scatter(
                x=n_range, y=lower_bound,
                mode='lines', line=dict(dash='dash', color='gray'),
                name='Lower bound (N/2)', showlegend=(idx == 0)
            ), row=row, col=col)
    
            fig.add_trace(go.Scatter(
                x=n_range, y=upper_bound,
                mode='lines', line=dict(dash='dot', color='black'),
                name='Upper bound (N(N-1)/2)', showlegend=(idx == 0)
            ), row=row, col=col)
    
            # Set axis to log-log
            fig.update_xaxes(type="log", title_text="Nodes", row=row, col=col)
            fig.update_yaxes(type="log", title_text="Edges", row=row, col=col)
    
        fig.update_layout(
            height=500 * n_rows,
            width=900,
            title="Log-Log Distribution of Nodes vs Edges by Area",
            template="plotly_white",
            margin=dict(t=50)
        )

        return fig


