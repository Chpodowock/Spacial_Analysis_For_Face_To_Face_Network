import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering


class MatrixSimilarity:
    def compute_activity_matrix(self, periods_df, entities):
        """
        Compute an activity matrix for given entities over specified periods.

        Args:
            periods_df (pd.DataFrame): Periods with 'label', 'start_unix', 'end_unix'.
            entities (dict): {entity_id: object with .activity (pd.Series indexed by datetime)}

        Returns:
            pd.DataFrame: Activity matrix (entities as rows, periods as columns)
        """
        activity_matrix = defaultdict(dict)

        for _, period in periods_df.iterrows():
            label = period["label"]
            start_time = pd.to_datetime(period["start_unix"], unit="s")
            end_time = pd.to_datetime(period["end_unix"], unit="s")

            for entity_id, obj in entities.items():
                # Filter activity for this period
                period_activity = obj.activity_rel[
                    (obj.activity_rel.index >= start_time) & (obj.activity_rel.index < end_time)
                ]
                total_activity = (
                    period_activity.sum() if not period_activity.empty else 0
                )
                activity_matrix[str(entity_id)][label] = total_activity

        # Build DataFrame (entities as rows, periods as columns)
        activity_df = pd.DataFrame(activity_matrix).T.fillna(0)

        return activity_df

    def compute_cosine_similarity(
        self, activity_input, reordered=True, num_groups=4, return_groups=False
    ):
        """
        Compute and optionally reorder a cosine similarity matrix from an activity matrix,
        and return optional groupings based on Spectral Clustering.

        Parameters:
            activity_input (pd.DataFrame): Rows = entities (e.g. readers), columns = features (e.g. time periods)
            reordered (bool): Whether to reorder the matrix based on group clustering
            num_groups (int or None): Number of clusters for Spectral Clustering (required if return_groups=True)
            return_groups (bool): Whether to return a Series of group labels

        Returns:
            pd.DataFrame: Cosine similarity matrix (reordered if specified)
            pd.Series (optional): Cluster/community assignments for each row
        """
        activity_input = activity_input.astype(float)

        # Step 1: Cosine similarity matrix
        cosine_sim = cosine_similarity(activity_input)
        np.fill_diagonal(cosine_sim, 1.0)

        similarity_df = pd.DataFrame(
            cosine_sim, index=activity_input.index, columns=activity_input.index
        )

        # Step 2: Spectral Clustering (if requested)
        if num_groups is not None and return_groups:
            clustering = SpectralClustering(
                n_clusters=num_groups,
                affinity="precomputed",
                assign_labels="discretize",
                random_state=42,
            )
            group_labels = clustering.fit_predict(cosine_sim)
            group_series = pd.Series(
                group_labels + 1, index=activity_input.index, name="Group"
            ) 

            if reordered:
                # Reorder based on cluster labels
                sorted_index = group_series.sort_values().index
                similarity_df = similarity_df.loc[sorted_index, sorted_index]

            return similarity_df, group_series

        # Optional: return reordered similarity matrix (without groups)
        if reordered:
            # Use PCA-based approximation for ordering if no clustering is requested
            from sklearn.decomposition import PCA

            coords = PCA(n_components=1).fit_transform(cosine_sim)
            order = np.argsort(coords[:, 0])
            reordered_index = activity_input.index[order]
            similarity_df = similarity_df.loc[reordered_index, reordered_index]

        return similarity_df