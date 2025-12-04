


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
        label_max_len=10,
        **kwargs
    ):
        """
        Plot an activity matrix as a heatmap with optional value annotations and label truncation.
    
        Args:
            matrix (pd.DataFrame): Activity matrix.
            title (str): Plot title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            cmap (str): Color scale.
            linewidth (float): Gridline width.
            linecolor (str): Gridline color.
            annot (bool): Show numeric values on the heatmap.
            fmt (str): Format for value annotation.
            label_max_len (int): Max length for x and y labels (truncated with "…" if needed).
        """
    
        def truncate(label):
            label = str(label)
            return label[:label_max_len] + "…)" if len(label) > label_max_len else label
    
        # Convert to numeric and fill missing
        matrix_numeric = matrix.apply(pd.to_numeric, errors="coerce").fillna(0)
    
        # Sort by total row activity
        row_totals = matrix_numeric.sum(axis=1)
        matrix_numeric = matrix_numeric.loc[row_totals.sort_values(ascending=False).index]
    
        # Prepare labels
        y_labels_full = list(matrix_numeric.index)
        x_labels_full = list(matrix_numeric.columns)
        y_labels = [truncate(label) for label in y_labels_full]
        x_labels = [truncate(label) for label in x_labels_full]
        y_positions = list(range(len(y_labels)))
    
        # Heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix_numeric.values,
                x=x_labels,
                y=y_positions,
                colorscale=cmap,
                showscale=True,
                hovertemplate="%{y} | %{x} → %{z}<extra></extra>",
                zsmooth=False,
            )
        )
    
        # Add annotations
        if annot:
            annotations = []
            for i, y_pos in enumerate(y_positions):
                for j, x_val in enumerate(x_labels):
                    value = matrix_numeric.iat[i, j]
                    annotations.append(
                        dict(
                            x=x_val,
                            y=y_pos,
                            text=format(value, fmt),
                            showarrow=False,
                            font=dict(color="black", size=10),
                        )
                    )
            fig.update_layout(annotations=annotations)
    
        # Layout
        fig.update_layout(
            title=title,
            xaxis=dict(
                title=xlabel,
                tickangle=45,
                showgrid=True,
                gridwidth=linewidth,
                gridcolor=linecolor,
                autorange=False,
                range=[-0.5, len(x_labels) - 0.5],
            ),
            yaxis=dict(
                title=ylabel,
                showgrid=True,
                gridwidth=linewidth,
                gridcolor=linecolor,
                autorange=False,
                range=[len(y_positions) - 0.5, -0.5],
                tickmode="array",
                tickvals=y_positions,
                ticktext=y_labels
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
        label_max_len=10, 
    ):
        """
        Plot a cosine similarity matrix as an interactive Plotly heatmap.
    
        Args:
            cos_sim_df (pd.DataFrame): Square cosine similarity matrix with aligned index/columns.
            title (str): Plot title.
            cmap (str): Color scale to use.
            linewidth (float): Width of grid lines.
            linecolor (str): Color of grid lines.
            annot (bool): Whether to annotate cells with similarity values.
            fmt (str): Format for annotations (e.g., '.2f').
            grid_lines (bool): Whether to add visible cell borders.
            label_max_len (int): Max number of characters per axis label.
        """
        def truncate(label):
            return label[:label_max_len] + "…)" if len(label) > label_max_len else label
    
        # Truncate x and y labels
        x_labels_full = cos_sim_df.columns.astype(str).tolist()
        y_labels_full = cos_sim_df.index.astype(str).tolist()
        x_labels = [truncate(lbl) for lbl in x_labels_full]
        y_labels = [truncate(lbl) for lbl in y_labels_full]
        values = cos_sim_df.values
    
        # Create heatmap trace
        fig = go.Figure(
            data=go.Heatmap(
                z=values,
                x=x_labels,
                y=y_labels,
                colorscale=cmap,
                zmin=0,
                zmax=1,
                colorbar=dict(title="Cosine Similarity"),
                hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
                zsmooth=False,
            )
        )
    
        # Add text annotations if requested
        if annot:
            annotations = [
                dict(
                    x=x_labels[j],
                    y=y_labels[i],
                    text=f"{values[i, j]:{fmt}}",
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    xref="x",
                    yref="y",
                )
                for i in range(len(y_labels))
                for j in range(len(x_labels))
            ]
            fig.update_layout(annotations=annotations)
    
        # Add gridlines using shape overlays
        shapes = []
        if grid_lines:
            for i in range(1, len(x_labels)):
                shapes.append(dict(
                    type="line",
                    x0=x_labels[i], x1=x_labels[i],
                    y0=y_labels[0], y1=y_labels[-1],
                    xref="x", yref="y",
                    line=dict(color=linecolor, width=linewidth)
                ))
            for i in range(1, len(y_labels)):
                shapes.append(dict(
                    type="line",
                    x0=x_labels[0], x1=x_labels[-1],
                    y0=y_labels[i], y1=y_labels[i],
                    xref="x", yref="y",
                    line=dict(color=linecolor, width=linewidth)
                ))
    
        # Final layout adjustments
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
    
            xaxis=dict(
                title="Signature",
                tickangle=45,
                showgrid=True,
                gridwidth=linewidth,
                gridcolor=linecolor,
                tickfont=dict(size=10),
                type="category"
            ),
    
            yaxis=dict(
                title="Signature",
                autorange="reversed",
                showgrid=True,
                gridwidth=linewidth,
                gridcolor=linecolor,
                tickfont=dict(size=10),
                type="category"
            ),
    
            shapes=shapes,
            margin=dict(l=80, r=20, t=50, b=60),
            template="plotly_white",
            height=600
        )
    
        return fig
