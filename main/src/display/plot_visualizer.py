from display_manager import go, px, cycle, display, pd, make_subplots, chain, Markdown, Counter, pio
from textwrap import wrap
from datetime import timedelta

class PlotVisualizer:
    def __init__(self, world, periods_df=None):
        self.periods_df = periods_df
        self.world = world

    def plot_normalized_activity(
        self,
        entities,
        experiment_id,
        mode_label="Entity",
        title_suffix="",
        show_total=True,
        sort_by_activity=False,
        top_n=5,
        df_period=None,
        with_labels=True 
    ):
        """
        Plots normalized activity time series for a selection of entities.
        """
        fig = go.Figure()
        color_palette = cycle(px.colors.qualitative.D3)

        if sort_by_activity:
            entity_sums = {
                key: ent.activity_rel.sum()
                for key, ent in entities.items()
                if ent.activity_rel is not None
            }
            sorted_entities = sorted(
                entities.items(), key=lambda x: entity_sums.get(x[0], 0), reverse=True
            )
        else:
            sorted_entities = list(entities.items())

        sorted_entities = sorted_entities[:top_n]
        total_activity = None

        for key, ent in sorted_entities:
            if ent.activity_rel is not None:
                name = f"{mode_label}: {key}"
                if hasattr(ent, "reader_ids"):
                    name = "[" + ",".join(ent.reader_ids) + "]"
                    

                if hasattr(ent, "name"):
                    name = ent.name
                    
                if show_total:
                    total_activity = (
                        ent.activity_rel if total_activity is None
                        else total_activity.add(ent.activity_rel, fill_value=0)
                    )
                color_ent = getattr(ent, "color", None) or next(color_palette)
                fig.add_trace(
                    go.Scatter(
                        x=ent.activity_rel.index,
                        y=ent.activity_rel.values,
                        mode="lines",
                        name=name,
                        line=dict(color=color_ent),
                        opacity=0.6 if show_total else 0.8,
                    )
                )

        if show_total and total_activity is not None:
            fig.add_trace(
                go.Scatter(
                    x=total_activity.index,
                    y=total_activity.values,
                    mode="lines",
                    name=f"Total {mode_label} Activity",
                    line=dict(color="black", width=2),
                    opacity=0.9,
                )
            )
            fig.data = (fig.data[-1],) + fig.data[:-1] # type: ignore

        if df_period is not None:
            self._add_period_shading(fig, entities, df_period, with_labels=with_labels)

        self._finalize_layout(fig, experiment_id, mode_label, title_suffix)
        return fig

    def _add_period_shading(self, fig, entities, df_periods, with_labels=True):
        if df_periods is None or df_periods.empty:
            return

        all_series = [e.activity_rel for e in entities.values() if e.activity_rel is not None]
        if not all_series:
            return

        max_y = max(max(series) for series in all_series) * 1.05

        for _, row in df_periods.iterrows():
            fillcolor = (
                "rgba(255, 0, 0, 0.1)" if row["type"] == "h" else
                "rgba(0, 0, 255, 0.1)" if row["type"] == "l" else
                "rgba(255, 165, 0, 0.1)"
            )

            fig.add_vrect(
                x0=row["start_unix_datetime"],
                x1=row["end_unix_datetime"],
                fillcolor=fillcolor,
                opacity=1.0,
                line_width=0,
                layer="above"
            )

            if with_labels:
                x_pos = row["start_unix_datetime"] + timedelta(seconds=10)
                fig.add_annotation(
                    x=x_pos,
                    y=max_y,
                    text=row["label"],
                    showarrow=False,
                    textangle=270,
                    yanchor="top",
                    xanchor="left",
                    font=dict(size=10, color="black"),
                    bgcolor="rgba(255,255,255,0.6)",
                )

        fig.update_yaxes(range=[0, max_y])
    
        
    def _finalize_layout(self, fig, experiment_id, mode_label="Entity", title_suffix=""):
        fig.update_layout(
            title=f"Normalized Activity by {mode_label} — Experiment: {experiment_id} {title_suffix}",
            xaxis=dict(
                title="Time",
                rangeslider=dict(visible=True),
                type="date",
                tickformat="%H:%M:%S",
            ),
            yaxis=dict(title="Normalized Activity"),
            hovermode="x unified",
            height=600,
            template="plotly_white",
            legend=dict(
                title=f"{mode_label}",
                font=dict(size=10),
            ),
        )


    def plot_transition_debug_activity(
        self,
        activity_series,
        smoothed,
        norm_derivative,
        transition_df,
        experiment_id,
        smooth_sigma,
        with_labels=True
    ):
        print("📈 Debug Plot: Raw, Smoothed & Derivative")

        class DummyEntity:
            def __init__(self, series):
                self.activity_rel = series

        raw_series = (activity_series - activity_series.min()) / (
            activity_series.max() - activity_series.min() + 1e-8
        )
        smooth_series = (smoothed - smoothed.min()) / (
            smoothed.max() - smoothed.min() + 1e-8
        )
        deriv_series = (norm_derivative + 1) / 2

        entities = {
            "Raw": DummyEntity(pd.Series(raw_series, index=activity_series.index)),
            "Smoothed": DummyEntity(pd.Series(smooth_series, index=activity_series.index)),
            "Derivative": DummyEntity(pd.Series(deriv_series, index=activity_series.index)),
        }

        return self.plot_normalized_activity(
            entities=entities,
            experiment_id=experiment_id,
            mode_label="",
            title_suffix=f"(σ={smooth_sigma})",
            show_total=False,
            sort_by_activity=False,
            df_period=transition_df,
            with_labels=with_labels
        )
            
    def plot_period_shading_stages(self, df_initial, df_transition, df_final):
        """
        Final display-optimized version with:
        - Vertical labels from 'label' column
        - Light gray rectangle borders
        - Top-left aligned background-colored subplot titles
        """
        from plotly.subplots import make_subplots
    
        row_titles = ["Initial Segmentation", "Detected Transitions", "Final Segmentation"]
    
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[None, None, None]
        )
    
        def add_period_rects(df, row_idx, colors, opacity=0.3):
            y0 = (3 - row_idx) / 3
            y1 = (3 - row_idx + 1) / 3
    
            for _, row in df.iterrows():
                x0 = pd.to_datetime(row["start_unix_datetime"])
                x1 = pd.to_datetime(row["end_unix_datetime"])
                duration = (x1 - x0).total_seconds()
                if duration < 120:
                    continue
    
                label = str(row.get("label", ""))
                period_type = row.get("type", "")
                color = colors.get(period_type, "rgba(200,200,200,1)")
    
                fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    fillcolor=color,
                    opacity=opacity,
                    layer="below",
                    line=dict(width=0.5, color="gray")
                )
    
                midpoint = x0 + (x1 - x0) / 2
                fig.add_annotation(
                    x=midpoint,
                    y=(y0 + y1) / 2,
                    xref="x",
                    yref="paper",
                    text=label,
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    textangle=90,
                    align="center"
                )
    
        color_map = {
            "h": "rgba(255, 0, 0, 1)",
            "l": "rgba(0, 0, 255, 1)",
            "t_up": "rgba(255, 165, 0, 1)",
            "t_down": "rgba(255, 165, 0, 1)"
        }
    
        add_period_rects(df_initial, row_idx=1, colors=color_map)
        add_period_rects(df_transition, row_idx=2, colors=color_map)
        add_period_rects(df_final, row_idx=3, colors=color_map)
    
        # Time range across all data
        min_time = pd.concat([
            df_initial["start_unix_datetime"],
            df_transition["start_unix_datetime"],
            df_final["start_unix_datetime"]
        ]).min()
        max_time = pd.concat([
            df_initial["end_unix_datetime"],
            df_transition["end_unix_datetime"],
            df_final["end_unix_datetime"]
        ]).max()
    
        # Invisible traces for axis alignment
        for i in range(1, 4):
            fig.add_trace(go.Scatter(
                x=[min_time, max_time],
                y=[0, 0],
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False
            ), row=i, col=1)
    
        # Axis styling
        for i in range(1, 4):
            fig.update_yaxes(
                showticklabels=False,
                title_text="",
                showgrid=False,
                zeroline=False,
                row=i, col=1
            )
            fig.update_xaxes(
                showticklabels=(i == 3),
                showgrid=False,
                zeroline=False,
                type="date",
                title_text="Time" if i == 3 else "",
                row=i, col=1
            )
    
        # Top-left aligned titles using real subplot domains
        for i, title in enumerate(row_titles, start=1):
            yaxis = f"yaxis{i if i > 1 else ''}"
            y_top = fig['layout'][yaxis]['domain'][1]  # top of row
            fig.add_annotation(
                text=title,
                x=0,
                y=y_top,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",  # precise alignment to top of domain
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="rgba(240,240,240,0.9)",
                borderpad=4,
                align="left",
                xshift=10
            )

    
        fig.update_layout(
            height=400,
            width=500,
            template="plotly_white",
            title=None,
            showlegend=False,
            margin=dict(t=30, b=40, l=40, r=30)
        )
    
        return fig
