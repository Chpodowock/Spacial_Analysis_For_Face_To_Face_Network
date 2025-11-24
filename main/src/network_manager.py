from collections import Counter
import networkx as nx
import pandas as pd
import numpy as np
import os


class NetworkManager:

    def build_area_transition_graph(self, world):
        """
        Builds a directed graph where:
        - Nodes are area IDs with color attributes.
        - Edges represent the number of agent transitions between areas.

        Result is stored in self.area_transition_graph (networkx.DiGraph).
        """
        G = nx.DiGraph()
        transition_counter = Counter()

        for agent in world.agents.values():
            if hasattr(agent, "trajectorie") and not agent.trajectorie.empty:
                traj = agent.trajectorie.dropna()
                traj = traj[traj.shift() != traj]
                area_sequence = traj.values

                for i in range(len(area_sequence) - 1):
                    src = area_sequence[i]
                    dst = area_sequence[i + 1]
                    if src != dst:
                        transition_counter[(src, dst)] += 1

        for area_id, area in world.areas.items():
            G.add_node(area_id, color=area.color,name=area.name)

        for (src, dst), weight in transition_counter.items():
            if src not in G:
                G.add_node(src, color=world.areas[src].color)
            if dst not in G:
                G.add_node(dst, color=world.areas[dst].color)
            G.add_edge(src, dst, weight=weight)

        self.area_transition_graph = G

    def build_temporal_area_transition_graphs(self, world):
        """
        Builds a list of directed graphs (one per period) representing agent transitions
        between areas. Edge weights count the number of transitions within each period.

        Result is stored in self.temporal_area_graphs: List[Tuple[str, nx.DiGraph]]
        """
        temporal_graphs = []

        for _, row in world.period_manager.final_period_df.iterrows():
            start = pd.to_datetime(row["start"])
            end = pd.to_datetime(row["end"])
            label = row["label"]

            G = nx.DiGraph()
            transition_counter = Counter()

            for agent in world.agents.values():
                if hasattr(agent, "trajectorie") and not agent.trajectorie.empty:
                    traj = agent.trajectorie.loc[start:end].dropna()
                    traj = traj[traj.shift() != traj]
                    area_sequence = traj.values

                    for i in range(len(area_sequence) - 1):
                        src = area_sequence[i]
                        dst = area_sequence[i + 1]
                        if src != dst:
                            transition_counter[(src, dst)] += 1

            for area_id, area in world.areas.items():
                G.add_node(area_id, color=area.color)

            for (src, dst), weight in transition_counter.items():
                if src not in G:
                    G.add_node(src, color=world.areas[src].color)
                if dst not in G:
                    G.add_node(dst, color=world.areas[dst].color)
                G.add_edge(src, dst, weight=weight)

            temporal_graphs.append((label, G))

        self.temporal_area_graphs = temporal_graphs
                
    def build_temporal_agent_graphs_by_area(self, world, window="20s"):
        """
        For each Area and each fixed time window, build a contact graph:
            - Nodes: agents active in the area
            - Edges: weighted by number of face-to-face contacts in that area/time
    
        Stores results in:
            area.temporal_agent_graphs : List[Tuple[pd.Timestamp, nx.Graph]]
        """
        contact_df = world.df.copy()
        contact_df["datetime"] = pd.to_datetime(contact_df["datetime"])
        contact_df["time_bin"] = contact_df["datetime"].dt.floor(window)
    
        for area in world.areas.values():
            area.temporal_agent_graphs = []
    
            if area.active_matrix_df is None or area.active_matrix_df.empty:
                continue
    
            matrix = area.active_matrix_df.copy()
            matrix.index = pd.to_datetime(matrix.index).floor(window)
            matrix = matrix.groupby(matrix.index).any()  # collapse duplicates
    
            for time_bin, df_bin in contact_df.groupby("time_bin"):
                if time_bin not in matrix.index:
                    continue
    
                active_agents = set(matrix.columns[matrix.loc[time_bin]])
                if not active_agents:
                    continue
    
                df_filtered = df_bin[
                    df_bin["i"].isin(active_agents) &
                    df_bin["j"].isin(active_agents)
                ]
    
                if df_filtered.empty:
                    continue
    
                G = nx.Graph()
    
                for _, row in df_filtered.iterrows():
                    a, b = sorted((row["i"], row["j"]))
                    if a == b:
                        continue  # skip self-loop
                    if G.has_edge(a, b):
                        G[a][b]["weight"] += 1
                    else:
                        G.add_edge(a, b, weight=1)
    
                n, m = G.number_of_nodes(), G.number_of_edges()
                if m > 0 and n >= 2:
                    # Optional sanity check
                    max_edges = n * (n - 1) / 2
                    min_edges = n / 2
                    if not (min_edges <= m <= max_edges):
                        print(f"⚠️ {area.area_name} @ {time_bin}: n={n}, m={m} out of bounds [{min_edges}, {max_edges}]")
                    area.temporal_agent_graphs.append((time_bin, G))
    
        print("✅ Built fixed-window temporal contact graphs in all Area objects.")
        
        
        
        



    
    def export_node_edge_timeseries_per_area(self, world, output_dir="densification_data", areas=None):
        """
        For each area, compute the (N, M) per time step and export to a CSV file
        as an L x 2 array with no headers. Compatible with the densificationscalingMLE code.
    
        Each row in output CSV:  N, M
        """
    
        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)
        import glob
        # === Clean existing CSV files in the output directory ===
        old_files = glob.glob(os.path.join(output_dir, "nm_*.csv"))
        for f in old_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"[!] Failed to remove {f}: {e}")
        print(f"[i] Cleared {len(old_files)} old .csv file(s) in '{output_dir}'.")
    
        # Process areas
        areas = areas or list(world.areas.keys())
    
        for area_id in areas:
            area = world.areas[area_id]
            graphs = area.temporal_agent_graphs
    
            nm_values = []
            for time, G in graphs:
                n = G.number_of_nodes()
                m = G.number_of_edges()
                if n > 0 and m > 0:
                    nm_values.append((n, m))
    
            if nm_values:
                nm_array = np.array(nm_values, dtype=int)
                safe_area_name = area.name.replace(" ", "_").lower()
                filename = os.path.join(output_dir, f"nm_{safe_area_name}.csv")
                np.savetxt(filename, nm_array, delimiter=",", fmt="%d")
                print(f"[✓] Saved: {filename}")
            else:
                print(f"[!] Skipped {area.name} (no valid (N,M) data)")
    
