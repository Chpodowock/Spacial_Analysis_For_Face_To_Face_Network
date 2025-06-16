from display_manager import go, px, cycle, display, pd, make_subplots, chain, Markdown, Counter,pio

class PlotVisualizer:
    def __init__(self,world, periods_df=None):
        self.periods_df = periods_df
        self.world = world
        
            
    
    def plot_agent_entropy_distribution(self, agents: dict):
        """
        Plot histogram of normalized entropy values for all agents using Plotly.
    
        Args:
            agents (dict): agent_id → Agent objects with .normalized_entropy computed.
        """
        entropies = [
            agent.normalized_entropy
            for agent in agents.values()
            if getattr(agent, "normalized_entropy", None) is not None
        ]
    
        if not entropies:
            print("⚠️ No agent entropies available to plot.")
            return
    
        fig = go.Figure()
    
        fig.add_trace(go.Histogram(
            x=entropies,
            nbinsx=20,
            marker_color="steelblue",
            opacity=0.75,
            name="Agent Entropies"
        ))
    
        fig.update_layout(
            title="🧠 Normalized Entropy of Area Visits",
            xaxis_title="Normalized Entropy (0 = focused, 1 = dispersed)",
            yaxis_title="Number of Agents",
            bargap=0.05,
            template="plotly_white",
            height=450
        )
    
        return fig

    # ==== Normalized Activity Line Plot ====
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
    ):
        """
        Plots normalized activity time series for given entities.

        Args:
            entities (dict): Dictionary of entities with 'activity' attribute.
            experiment_id (str): Experiment identifier for titles.
            mode_label (str): Label for entity type.
            title_suffix (str): Additional title suffix.
            show_total (bool): Whether to overlay total activity.
            sort_by_activity (bool): Sort entities by total activity sum.
            top_n (int): Number of entities to display (default 5).
        """
        fig = go.Figure()
        color_palette = cycle(px.colors.qualitative.D3)

        # Sort entities by total activity if requested
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
            sorted_entities = entities.items()

        # Limit to top N entities
        sorted_entities = list(sorted_entities)[:top_n]

        total_activity = None

        # Plot each entity
        for key, ent in sorted_entities:
            if ent.activity_rel is not None:
                name = f"{mode_label}: {key}"
                if hasattr(ent, "reader_ids"):
                    sig_label = "[" + ",".join(ent.reader_ids) + "]"
                    name = f"{mode_label}: {sig_label}"

                if show_total:
                    total_activity = (
                        ent.activity_rel
                        if total_activity is None
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

        # Plot total activity overlay
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
            fig.data = (fig.data[-1],) + fig.data[:-1]  # Total on top

        # Period shading
        self._add_period_shading(fig, entities,df_period)

        # Layout adjustments
        self._finalize_layout(fig, experiment_id, mode_label, title_suffix)
        
        return fig

    def plot_transition_debug_activity(
        self,
        activity_series,
        smoothed,
        norm_derivative,
        transition_df,
        experiment_id,
        smooth_sigma,
    ):
        """
        Debug plot for transition detection: raw activity, smoothed signal, derivative.

        Args:
            activity_series (pd.Series): Raw activity series (DatetimeIndex).
            smoothed (np.array): Smoothed signal values.
            norm_derivative (np.array): Normalized derivative values.
            transition_df (pd.DataFrame): Transition periods (for shading).
            experiment_id (str): Experiment ID.
            smooth_sigma (float): Sigma value used for smoothing.
        """
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
            "Smoothed": DummyEntity(
                pd.Series(smooth_series, index=activity_series.index)
            ),
            "Derivative": DummyEntity(
                pd.Series(deriv_series, index=activity_series.index)
            ),
        }

        fig = self.plot_normalized_activity(
            entities=entities,
            experiment_id=experiment_id,
            mode_label="Total Activity, Smoothed & Derivative",
            title_suffix=f"(σ={smooth_sigma})",
            show_total=False,
            sort_by_activity=False,
            df_period = transition_df,
        )
        return fig

    # ==== Generic Bar Plot ====
    def plot_bar_distribution(
        self,
        x_values,
        y_values,
        title,
        xaxis_title,
        yaxis_title,
        color="skyblue",
        x_tickangle=0,
        width=0.6,
        height=500,
    ):
        fig = go.Figure(
            go.Bar(
                x=x_values,
                y=y_values,
                text=y_values,
                textposition="outside",
                marker=dict(color=color, line=dict(color="black", width=1)),
                width=width,
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            xaxis_tickangle=x_tickangle,
            template="plotly_white",
            height=height,
        )
        

        return fig
    

    def plot_signature_distributions(self):
        """
        Plot side-by-side histograms of signature attributes:
        - Number of plans involved
        - Dominant plan ratio
        - Multi-plan flag
        - Dominant plan occurrences
        """
        if not self.world.signatures:
            print("⚠️ No signatures found in WorldModel. Compute signatures first.")
            return

        # === Extract Signature Attributes ===
        data = {
            "num_plans_involved": [],
            "dominant_plan_ratio": [],
            "multi_plan": [],
            "dominant_plan": [],
        }

        for sig_obj in self.world.signatures.values():
            data["num_plans_involved"].append(sig_obj.num_plans_involved)
            data["dominant_plan_ratio"].append(sig_obj.dominant_plan_ratio)
            data["multi_plan"].append(sig_obj.multi_plan)
            data["dominant_plan"].append(sig_obj.dominant_plan)

        sig_df = pd.DataFrame(data)

        # === Setup 2x2 Plotly Subplots ===
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Number of Plans Involved",
                "Dominant Plan Ratio",
                "Multi-Plan Flag (True/False)",
                "Dominant Plan Occurrences"
            ),
            horizontal_spacing=0.15,
            vertical_spacing=0.2
        )

        # === Plot 1: Number of Plans Involved ===
        counts = sig_df["num_plans_involved"].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=counts.index.astype(str), y=counts.values, name="Num Plans"),
            row=1, col=1
        )

        # === Plot 2: Dominant Plan Ratio Histogram ===
        fig.add_trace(
            go.Histogram(x=sig_df["dominant_plan_ratio"], nbinsx=20, name="Dominant Ratio"),
            row=1, col=2
        )

        # === Plot 3: Multi-Plan Flag ===
        multi_counts = sig_df["multi_plan"].value_counts()
        fig.add_trace(
            go.Bar(x=multi_counts.index.astype(str), y=multi_counts.values, name="Multi Plan"),
            row=2, col=1
        )

        # === Plot 4: Dominant Plan Occurrences ===
        dom_counts = sig_df["dominant_plan"].value_counts()
        fig.add_trace(
            go.Bar(x=dom_counts.index.astype(str), y=dom_counts.values, name="Dominant Plan"),
            row=2, col=2
        )

        # === Layout ===
        fig.update_layout(
            title_text=f"Signature Distributions — Experiment: {self.world.experiment_id}",
            height=700,
            template="plotly_white",
            showlegend=False
        )

        # Axis titles
        fig.update_xaxes(title_text="Number of Plans", row=1, col=1)
        fig.update_xaxes(title_text="Dominant Ratio", row=1, col=2)
        fig.update_xaxes(title_text="Multi-Plan", row=2, col=1)
        fig.update_xaxes(title_text="Dominant Plan", row=2, col=2)

        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Number of Signatures", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        return fig

    # ==== Period Shading Helper ====
    def _add_period_shading(self, fig, entities, df_periods):
        if df_periods is None or df_periods.empty:
            return

        all_series = [e.activity_rel for e in entities.values() if e.activity_rel is not None]
        if not all_series:
            return

        max_y = max(max(series) for series in all_series) * 1.05

        for _, row in df_periods.iterrows():
            fig.add_vrect(
                x0=row["start_unix_datetime"],
                x1=row["end_unix_datetime"],
                fillcolor=(
                    "red"
                    if row["type"] == "h"
                    else "blue" if row["type"] == "l" else "orange"
                ),
                opacity=0.2,
                line_width=0,
                layer="below",
                annotation_text=row["label"],
                annotation_position="top left",
            )

        fig.update_yaxes(range=[0, max_y])

    # ==== Layout Helper ====
    def _finalize_layout(
        self, fig, experiment_id, mode_label="Entity", title_suffix=""
    ):
        fig.update_layout(
            title=f"📊 Normalized Activity by {mode_label} — Experiment: {experiment_id} {title_suffix}",
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
            legend=dict(title=f"{mode_label}s", font=dict(size=10)),
        )

    def report_world_model(self, verbose=True, top_n_signatures=12):
        """
        Generate an enriched summary report for a given WorldModel instance.

        Args:
            world_model (WorldModel): The WorldModel instance.
            verbose (bool): Show sample data preview.
            top_n_signatures (int): Number of top signatures to display.
        """
        experiment_id = self.world.experiment_id
        df = self.world.df

        # === Summary Header ===
        print(f"\n🧭 WorldModel Report — Experiment: {experiment_id}")
        print("═══════════════════════════════════════════════")
        print(f"📊 Contacts: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"🗺️ Plans loaded: {len(self.world.plans)}")
        print(f"📡 Readers registered: {len(self.world.readers)}")
        print(f"🔗 Signatures computed: {len(self.world.signatures)}")
        print("═══════════════════════════════════════════════")

        # === Optional Sample Preview ===
        if verbose:
            print("\n🔎 Sample Data (df.head()):")
            display(df.head())

        # === Compute Summary Metrics ===
        total_contacts = len(df)
        unique_readers = set(chain.from_iterable(df["signature"]))
        num_unique_readers = len(unique_readers)
        avg_readers_per_signature = df["signature"].apply(len).mean()
        min_readers = df["signature"].apply(len).min()
        max_readers = df["signature"].apply(len).max()
        unique_signatures = df["signature"].apply(lambda x: tuple(sorted(x))).nunique()

        # === Display Summary Table ===
        summary_md = f"""
        ### 🧪 Experiment Summary — {experiment_id}
    
        | Metric                            | Value                       |
        |------------------------------------|-----------------------------|
        | 📊 Total contacts                  | `{total_contacts}`          |
        | 📡 Unique readers                  | `{num_unique_readers}`      |
        | 🔁 Avg readers per signature       | `{avg_readers_per_signature:.2f}` |
        | 🔼 Max readers in one signature    | `{max_readers}`             |
        | 🔽 Min readers in one signature    | `{min_readers}`             |
        | 📡 Unique signatures               | `{unique_signatures}`       |
    
        **Reader IDs:**  
        `{', '.join(sorted(unique_readers))}`
        """
        display(Markdown(summary_md))

        # === Side-by-Side Signature & Reader Distributions ===
        signature_counts = df["signature"].value_counts().head(top_n_signatures)
        signature_labels = ["[" + ",".join(sig) + "]" for sig in signature_counts.index]

        reader_counts = Counter(chain.from_iterable(df["signature"]))
        readers_activity = pd.Series(reader_counts).sort_values(ascending=False)

        # Subplots for Signature & Reader Distributions
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Top {top_n_signatures} Signatures",
                "Reader Contact Counts",
            ),
            horizontal_spacing=0.15,
        )

        # Signature Bar Plot
        fig.add_trace(
            go.Bar(
                x=signature_labels,
                y=signature_counts.values,
                text=signature_counts.values,
                textposition="outside",
                marker=dict(color="teal", line=dict(color="black", width=1)),
                width=0.6,
                name="Signatures",
            ),
            row=1,
            col=1,
        )

        # Reader Bar Plot
        fig.add_trace(
            go.Bar(
                x=readers_activity.index,
                y=readers_activity.values,
                text=readers_activity.values,
                textposition="outside",
                marker=dict(color="mediumseagreen", line=dict(color="black", width=1)),
                width=0.6,
                name="Readers",
            ),
            row=1,
            col=2,
        )

        # Layout adjustments
        fig.update_layout(
            title=f"📊 Signature & Reader Distributions — {experiment_id}",
            template="plotly_white",
            height=500,
        )

        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)

        fig.update_yaxes(title_text="Number of Contacts", row=1, col=1)
        fig.update_yaxes(title_text="Number of Contacts", row=1, col=2)

        return fig

        print(f"\n✅ Report completed for {experiment_id}")

    def report_periods(self):
        """
        Generate a summary report for periods (from PeriodManager):
        - Total timeline duration
        - Breakdown by period type
        - Transition count
        - Histogram visualization
        """
        period_manager = self.world.period_manager

        if (
            period_manager.final_period_df is None
            or period_manager.final_period_df.empty
        ):
            print(
                f"❗ No final periods computed for {self.world.experiment_id}. Run adjust_periods() first."
            )
            return

        experiment_id = self.world.experiment_id
        period_df = period_manager.final_period_df
        transition_df = period_manager.transition_df

        print(f"\n📝 Period Report — {experiment_id}")
        print("═══════════════════════════════════════════════")

        # === Total Duration ===
        total_duration = (
            period_df["end_unix_datetime"] - period_df["start_unix_datetime"]
        ).dt.total_seconds().sum() / 60
        print(f"⏱️ Total timeline duration: {total_duration:.2f} minutes")

        # === Compute Duration per Type ===
        period_df = period_df.copy()
        period_df["duration_min"] = (
            period_df["end_unix_datetime"] - period_df["start_unix_datetime"]
        ).dt.total_seconds() / 60

        # Compute High/Low/Transitions separately
        high_duration = period_df.loc[period_df["type"] == "h", "duration_min"].sum()
        low_duration = period_df.loc[period_df["type"] == "l", "duration_min"].sum()
        up_duration = period_df.loc[period_df["type"] == "t_up", "duration_min"].sum()
        down_duration = period_df.loc[
            period_df["type"] == "t_down", "duration_min"
        ].sum()

        # === Textual Breakdown ===
        print("\n📊 Duration breakdown by period type:")
        print(f"  - High Activity: {high_duration:.2f} min")
        print(f"  - Low Activity: {low_duration:.2f} min")
        print(f"  - Up Transitions: {up_duration:.2f} min")
        print(f"  - Down Transitions: {down_duration:.2f} min")

        # === Plot: High & Low as columns, Transitions stacked ===
        fig = go.Figure()

        # High Activity Bar
        fig.add_trace(
            go.Bar(
                x=["High Activity"],
                y=[high_duration],
                name="High Activity",
                marker=dict(color="red", line=dict(color="black", width=1)),
                text=f"{high_duration:.1f} min",
                textposition="outside",
                width=0.6,
            )
        )

        # Low Activity Bar
        fig.add_trace(
            go.Bar(
                x=["Low Activity"],
                y=[low_duration],
                name="Low Activity",
                marker=dict(color="blue", line=dict(color="black", width=1)),
                text=f"{low_duration:.1f} min",
                textposition="outside",
                width=0.6,
            )
        )

        # Up Transition (stacked in "Transition" column)
        fig.add_trace(
            go.Bar(
                x=["Transition"],
                y=[up_duration],
                name="Up Transition",
                marker=dict(color="yellow", line=dict(color="black", width=1)),
                text=f"{up_duration:.1f} min",
                textposition="outside",
                width=0.6,
            )
        )

        # Down Transition (stacked in "Transition" column)
        fig.add_trace(
            go.Bar(
                x=["Transition"],
                y=[down_duration],
                name="Down Transition",
                marker=dict(color="darkorange", line=dict(color="black", width=1)),
                text=f"{down_duration:.1f} min",
                textposition="outside",
                width=0.6,
            )
        )

        # === Layout ===
        fig.update_layout(
            title=f"🗓️ Period Type Duration Distribution — {experiment_id}",
            xaxis_title="Period Type",
            yaxis_title="Duration (minutes)",
            barmode="stack",
            template="plotly_white",
            height=500,
        )

        return fig