


from display_manager import go, pd


class MatrixVisualizer:

    def plot_activity_matrix(
        self,
        matrix,
        title,
        xlabel="Period",
        ylabel="Entity",
        cmap="Viridis",
        linewidth=0.5,
        linecolor="gray",
        annot=True,
        fmt=".0f",
    ):

        # Convert matrix to numeric (if needed)
        matrix_numeric = matrix.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Plotly Heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix_numeric.values,
                x=matrix_numeric.columns,
                y=matrix_numeric.index,
                colorscale=cmap,
                showscale=True,
                hovertemplate="%{y} | %{x} → %{z}<extra></extra>",
                zsmooth=False,
            )
        )

        # Add annotations manually if requested
        if annot:
            annotations = []
            for i, y_val in enumerate(matrix_numeric.index):
                for j, x_val in enumerate(matrix_numeric.columns):
                    value = matrix_numeric.iat[i, j]
                    annotations.append(
                        dict(
                            x=x_val,
                            y=y_val,
                            text=format(value, fmt),
                            showarrow=False,
                            font=dict(color="black", size=10),
                        )
                    )
            fig.update_layout(annotations=annotations)

        # Layout adjustments
        fig.update_layout(
            title=title,
            xaxis=dict(
                title=xlabel,
                tickangle=45,
                showgrid=True,
                gridwidth=linewidth,
                gridcolor=linecolor,
            ),
            yaxis=dict(
                title=ylabel,
                autorange="reversed",
                showgrid=True,
                gridwidth=linewidth,
                gridcolor=linecolor,
            ),
            margin=dict(l=60, r=10, t=50, b=50),
            template="plotly_white",
            height=600,
        )

        return fig

    def plot_cosine_similarity_matrix(
        self,
        cos_sim_df,
        title="Cosine Similarity",
        cmap="Viridis",
        linewidth=0.5,
        linecolor="gray",
        annot=False,
        fmt=".2f",
        grid_lines=True,
    ):
        """
        Plot a cosine similarity matrix as an interactive Plotly heatmap.

        Args:
            cos_sim_df (pd.DataFrame): Cosine similarity matrix (square DataFrame).
            title (str): Plot title.
            cmap (str): Color scale.
            linewidth (float): Gridline width.
            linecolor (str): Gridline color.
            annot (bool): Annotate each cell with its value.
            fmt (str): Annotation format (e.g., '.2f').
            grid_lines (bool): Add separation lines between cells.
        """
        x_labels = cos_sim_df.columns.astype(str).tolist()
        y_labels = cos_sim_df.index.astype(str).tolist()

        # Heatmap trace
        fig = go.Figure(
            data=go.Heatmap(
                z=cos_sim_df.values,
                x=x_labels,
                y=y_labels,
                colorscale=cmap,
                showscale=True,
                zmin=0,
                zmax=1,
                colorbar=dict(title="Cosine Similarity"),
                hovertemplate="%{y} | %{x} → %{z:.2f}<extra></extra>",
                zsmooth=False,
            )
        )

        if annot:
            values = cos_sim_df.values
            n_cols = len(x_labels)
            n_rows = len(y_labels)

            annotations = [
                dict(
                    x=j,
                    y=i,
                    text=f"{values[i, j]:{fmt}}",
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    xref="x",
                    yref="y",
                )
                for i in range(n_rows)
                for j in range(n_cols)
            ]

            # Update axis tick labels (numeric positions but correct text)
            fig.update_layout(
                annotations=annotations,
                xaxis=dict(
                    tickmode="array", tickvals=list(range(n_cols)), ticktext=x_labels
                ),
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(n_rows)),
                    ticktext=y_labels,
                    autorange="reversed",
                ),
            )

        # Optional separation grid lines (visual cell borders)
        if grid_lines:
            # Compute grid lines (corrected alignment)
            shapes = []
            n_cols = len(x_labels)
            n_rows = len(y_labels)

            # Map index to numeric positions
            x_pos = list(range(n_cols))
            y_pos = list(range(n_rows))

            # Vertical lines between columns
            for i in range(1, n_cols):
                shapes.append(
                    dict(
                        type="line",
                        x0=i - 0.5,
                        x1=i - 0.5,
                        y0=-0.5,
                        y1=n_rows - 0.5,
                        xref="x",
                        yref="y",
                        line=dict(color=linecolor, width=linewidth),
                    )
                )

            # Horizontal lines between rows
            for i in range(1, n_rows):
                shapes.append(
                    dict(
                        type="line",
                        x0=-0.5,
                        x1=n_cols - 0.5,
                        y0=i - 0.5,
                        y1=i - 0.5,
                        xref="x",
                        yref="y",
                        line=dict(color=linecolor, width=linewidth),
                    )
                )

            # Fix axis tick mode to match numeric positions but show original labels
            fig.update_layout(
                xaxis=dict(tickmode="array", tickvals=x_pos, ticktext=x_labels),
                yaxis=dict(
                    tickmode="array",
                    tickvals=y_pos,
                    ticktext=y_labels,
                    autorange="reversed",
                ),
                shapes=shapes,
            )

        # Layout adjustments
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis=dict(
                title="Entity",
                tickangle=45,
                showgrid=True,
                gridwidth=linewidth,
                gridcolor=linecolor,
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                title="Entity",
                autorange="reversed",
                showgrid=True,
                gridwidth=linewidth,
                gridcolor=linecolor,
                tickfont=dict(size=10),
            ),
            margin=dict(l=80, r=20, t=50, b=60),
            template="plotly_white",
            height=600,
        )

        return fig
