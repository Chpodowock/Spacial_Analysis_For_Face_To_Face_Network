from display_manager import go, np, pd, make_subplots, math, Point, gaussian_filter


class MapVisualizer:
    
    def __init__(self, world):
        self.plans = world.plans
        self.signatures = world.signatures
        self.world = world

    def display(self, scale=1.0, show_outline=False, activity_threshold=0.005):
        """
        Display plans with agent signature surfaces, area colors, and interactive legend.
        Signature polygons have opacity proportional to their activity.
    
        Args:
            scale (float): Scaling factor for figure size.
            show_outline (bool): Whether to show black polygon outlines.
            activity_threshold (float): Minimum opacity to apply to active signatures.
        """
    
        def get_max_signature_activity():
            return max(
                (np.sum(sig.activity) if hasattr(sig, "activity") and sig.activity is not None else 0)
                for sig in self.signatures.values()
            )
    
        max_activity = get_max_signature_activity()
    
        num_plans = len(self.plans)
        columns = math.ceil(math.sqrt(num_plans))
        rows = math.ceil(num_plans / columns)
    
        max_width = max(plan.size[0] for plan in self.plans.values())
        max_height = max(plan.size[1] for plan in self.plans.values())
    
        fig = make_subplots(
            rows=rows,
            cols=columns,
            subplot_titles=[plan.name for plan in self.plans.values()],
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
        )
    
        layout_images = []
        used_area_ids = set()
    
        for idx, plan in enumerate(self.plans.values(), start=1):
            row, col = divmod(idx - 1, columns)
            row += 1
            col += 1
            width, height = plan.size
    
            # Plot readers
            reader_x = [x_rel * width for x_rel, _ in plan.readers.values()]
            reader_y = [y_rel * height for _, y_rel in plan.readers.values()]
    
            fig.add_trace(
                go.Scatter(
                    x=reader_x,
                    y=reader_y,
                    mode="markers",
                    marker=dict(size=14, color="red", symbol="x"),
                    hovertext=list(plan.readers.keys()),
                    hoverinfo="text",
                    name="Readers",
                    showlegend=(idx == 1),
                    legendgroup="readers"
                ),
                row=row,
                col=col,
            )
    
            # Plot signature polygons
            for sig_obj in self.signatures.values():
                if sig_obj.dominant_plan != plan.name:
                    continue
    
                polygon = getattr(sig_obj, "dominant_plan_polygon", None)
                if not polygon or not hasattr(polygon, "exterior"):
                    continue
    
                poly_coords = [(x * width, y * height) for x, y in polygon.exterior.coords]
                poly_x, poly_y = zip(*poly_coords)
    
                sig_id = tuple(sig_obj.id)
                area_id = self.world.signature_to_area.get(sig_id)
                area = self.world.areas.get(area_id)
                area_color = getattr(area, "color", "rgba(200,200,200,0.2)")
    
                activity_val = np.sum(sig_obj.activity) if sig_obj.activity is not None else 0
                norm_activity = activity_val / max(max_activity, 1)
                if norm_activity < activity_threshold:
                    continue
                
                
    
                opacity = 0.1 + (1 - 0.1) * min(norm_activity, 1.0)
    
                hover_text = (
                    f"<b>Signature:</b> {sig_id}<br>"
                    f"<b>Area:</b> {area_id if area_id is not None else 'Unassigned'}<br>"
                    f"<b>Activity:</b> {activity_val:.2f}"
                )
    
                fig.add_trace(
                    go.Scatter(
                        x=poly_x,
                        y=poly_y,
                        mode="lines" if show_outline else "none",
                        fill="toself",
                        line=dict(
                            color="black" if show_outline else area_color,
                            width=2
                        ),
                        fillcolor=area_color,
                        hovertext=hover_text,
                        hoverinfo="text",
                        name=f"Area {area_id}" if area_id else "Unassigned",
                        showlegend=(area_id not in used_area_ids),
                        legendgroup=f"area_{area_id}" if area_id else "area_unassigned",
                        opacity=opacity
                    ),
                    row=row,
                    col=col,
                )
    
                used_area_ids.add(area_id)
    
            # Background plan image
            xref = f"x{idx}" if idx > 1 else "x"
            yref = f"y{idx}" if idx > 1 else "y"
    
            layout_images.append(dict(
                source=plan.image,
                x=0, y=0,
                sizex=width, sizey=height,
                xref=xref, yref=yref,
                sizing="stretch",
                layer="below"
            ))
    
            fig.update_xaxes(visible=False, range=[0, width], scaleanchor=yref, row=row, col=col)
            fig.update_yaxes(visible=False, range=[height, 0], scaleanchor=xref, row=row, col=col)
    
        fig.update_layout(
            images=layout_images,
            title="📡 Signature Surfaces & Readers (Opacity ∝ Activity)",
            margin=dict(l=10, r=10, t=40, b=10),
            height=rows * max_height * scale,
            width=columns * max_width * scale,
            legend=dict(
                title="Legend",
                itemsizing='constant',
                bordercolor="black",
                borderwidth=1,
                tracegroupgap=5
            )
        )
    
        fig.show()




    def plot_animated_signature_activity(self, grid_size=(10, 10), sigma=10, scale=1.0,
                                         heatmap_opacity=0.5, time_bin="2h"):
    
    
        all_times = pd.concat([
            sig.activity for sig in self.world.signatures.values()
            if hasattr(sig, "activity") and isinstance(sig.activity, pd.Series)
        ], axis=1).index
    
        time_bins = pd.date_range(start=all_times.min(), end=all_times.max(), freq=time_bin)
    
        num_plans = len(self.world.plans)
        columns = math.ceil(math.sqrt(num_plans))
        rows = math.ceil(num_plans / columns)
        max_width = max(plan.size[0] for plan in self.world.plans.values())
        max_height = max(plan.size[1] for plan in self.world.plans.values())
    
        fig = make_subplots(
            rows=rows,
            cols=columns,
            subplot_titles=list(self.world.plans.keys()),
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
        )
    
        x_grids, y_grids = {}, {}
        for plan_name, plan in self.world.plans.items():
            width, height = plan.size
            x_grids[plan_name] = np.linspace(0, width, grid_size[0])
            y_grids[plan_name] = np.linspace(0, height, grid_size[1])
    
        # === First pass: compute global max ===
        global_max = 0
        smoothed_by_time = []
    
        for t_start, t_end in zip(time_bins[:-1], time_bins[1:]):
            frame_data = []
            for plan_name, plan in self.world.plans.items():
                width, height = plan.size
                activity_grid = np.zeros(grid_size)
                x_scale = width / grid_size[0]
                y_scale = height / grid_size[1]
    
                for sig in self.world.signatures.values():
                    if sig.dominant_plan != plan_name:
                        continue
                    if not getattr(sig, "dominant_plan_polygon", None) or not hasattr(sig.dominant_plan_polygon, "contains"):
                        continue
                    if not hasattr(sig, "activity") or sig.activity.empty:
                        continue
    
                    time_slice = sig.activity.loc[t_start:t_end]
                    if time_slice.empty:
                        continue
                    activity_weight = time_slice.sum()
    
                    for i in range(grid_size[0]):
                        for j in range(grid_size[1]):
                            x_coord = i * x_scale + x_scale / 2
                            y_coord = j * y_scale + y_scale / 2
                            point = Point(x_coord / width, y_coord / height)
                            if sig.dominant_plan_polygon.contains(point):
                                activity_grid[j, i] += activity_weight
    
  
                
                smoothed = gaussian_filter(activity_grid, sigma=sigma)
                log_smoothed = np.log1p(smoothed)
                global_max = max(global_max, log_smoothed.max())
                frame_data.append((plan_name, log_smoothed))

    
            smoothed_by_time.append((t_start, frame_data))
    
        # === Second pass: generate frames with fixed zmax ===
        frames = []
        for t_start, frame_data in smoothed_by_time:
            traces = []
            for plan_name, smoothed in frame_data:
                traces.append(go.Heatmap(
                        z=smoothed,
                        x=x_grids[plan_name],
                        y=y_grids[plan_name],
                        zmin=0,
                        zmax=global_max if global_max > 0 else 1,
                        colorscale="Viridis",
                        zsmooth="best",
                        showscale=False,
                        opacity=heatmap_opacity  # entire heatmap layer has this opacity
                    ))
            frames.append(go.Frame(data=traces, name=str(t_start)))
    
        for i, trace in enumerate(frames[0].data):
            fig.add_trace(trace, row=(i // columns) + 1, col=(i % columns) + 1)
    
        # === Background images and axis ===
        layout_images = []
        for idx, (plan_name, plan) in enumerate(self.world.plans.items(), start=1):
            width, height = plan.size
            xref = f"x{idx}" if idx > 1 else "x"
            yref = f"y{idx}" if idx > 1 else "y"
            layout_images.append(dict(
                source=plan.image,
                x=0, y=0,
                sizex=width, sizey=height,
                xref=xref, yref=yref,
                sizing="stretch",
                layer="below",
                opacity=1
            ))
            fig.update_xaxes(visible=False, range=[0, width], scaleanchor=yref, row=(idx - 1) // columns + 1, col=(idx - 1) % columns + 1)
            fig.update_yaxes(visible=False, range=[height, 0], scaleanchor=xref, row=(idx - 1) // columns + 1, col=(idx - 1) % columns + 1)
    
        # === Final layout ===
        fig.update_layout(
            images=layout_images,
            height=rows * max_height * scale,
            width=columns * max_width * scale,
            template="plotly_white",
            title="🌀 Animated Signature Activity Over Time",
            updatemenus=[{
                "type": "buttons",
                "buttons": [{
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                }]
            }],
            sliders=[{
                "active": 0,
                "steps": [{
                    "label": str(frame.name),
                    "method": "animate",
                    "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                } for frame in frames]
            }]
        )
    
        fig.update(frames=frames)
        fig.show()
