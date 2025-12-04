from pathlib import Path
import pandas as pd
import ast
from itertools import chain
from PIL import Image

from period_manager import PeriodManager



import plotly.colors as pc
import plotly.express as px
import itertools
import numpy as np
from collections import defaultdict

from entities import Plan, Area, Agent, Reader, Signature


class WorldModel:
    
    DEFAULT_FREQ = "20s"
    
    def __init__(
        self,
        experiment_id,
        experiments_definition,
        base_dir,
    ):
        self.base_dir = Path(base_dir)
        self.experiment_id = experiment_id
        self.time_offset = experiments_definition[experiment_id]["offset"]
        self.plans_definitions = experiments_definition[experiment_id]['plans']

        self.df_tijs = None
        self.period_manager = None
        self.plans = {}
        self.readers = {}
        self.signatures = {}
        self.areas = {}
        self.agents = {}

        
        self.activity_max = 1
        self.area_transition_graph = None
        self.temporal_area_graphs = None

        self.plans_dir = self.base_dir / "data" / "plans" / self.experiment_id
        
        
    def initialize(self):
        self.load_data()
        self.compute_signatures()
        self.compute_reader_activities()


    def load_data(self):
        self._load_contact_data()
        periods_df = self._load_periods_data()
        self.period_manager = PeriodManager(
            self.df_tijs, periods_df, self.experiment_id, self.time_offset
        )
        self._load_plans_and_readers()
        
    def _load_contact_data(self):
        data_path = (
            self.base_dir / "data" / "tijs" / f"tij_with_readers_{self.experiment_id}.dat"
        )
    
        try:
            self.df_tijs = pd.read_csv(data_path, delimiter=",")
            print(f"✅ Loaded contact data file: {data_path.name}")
    
            if "readers" in self.df_tijs.columns:
                self.df_tijs["readers"] = self.df_tijs["readers"].apply(ast.literal_eval)
                self.df_tijs.rename(columns={"readers": "signature"}, inplace=True)
                print("✅ Converted and renamed 'readers' column to 'signature'.")
    
            self.df_tijs["datetime"] = pd.to_datetime(self.df_tijs["t"], unit="s")
            self.activity_max = self.df_tijs.set_index("datetime").resample(self.DEFAULT_FREQ).size().max() or 1
            
    
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load contact data from {data_path}: {e}")
    
    
    def _load_periods_data(self):
        period_path = (
            self.base_dir / "data" / "periodes" / f"periodes_{self.experiment_id}.dat"
        )
    
        try:
            periods_df = pd.read_csv(
                period_path,
                sep=r"\s+",
                header=None,
                names=["start", "end", "label", "type"],
            )
            print(f"✅ Loaded periods file: {period_path.name}")
            return periods_df
    
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load periods from {period_path}: {e}")
    
    
    def _load_plans_and_readers(self):
        for plan_name, info in self.plans_definitions.items():
            plan_file_path = self.plans_dir / info["file"]
    
            if plan_file_path.exists():
                try:
                    image = Image.open(plan_file_path)
                    plan = Plan(plan_name, image, info["readers"])
                    self.plans[plan_name] = plan
    
                    print(
                        f"✅ Loaded plan '{plan.name}' ({plan.size[0]}x{plan.size[1]}) "
                        f"with {len(info['readers'])} readers"
                    )
    
                except Exception as e:
                    print(f"❌ Error loading plan image: {plan_file_path}")
                    print(f"   Exception: {e}")
                    print(f"⚠️ Proceeding to create readers for '{plan_name}' without valid plan image.")
    
            else:
                print(f"❌ Plan file not found: {plan_file_path}")
                print(f"⚠️ Proceeding to create readers for '{plan_name}' without plan image.")
    
            # Create Reader objects at given location
            for reader_id, (x_rel, y_rel) in info["readers"].items():
                self.readers[reader_id] = Reader(reader_id, x_rel, y_rel, plan_name)
                
    def compute_signatures(self):
        signature_objects = {}
        unique_signatures = (
            self.df_tijs["signature"]
            .value_counts()
            .sort_values(ascending=False)
            .index.to_list()
        )
        for sig in unique_signatures:
            signature = Signature(sig, self.readers)
            signature_objects[tuple(sig)] = signature

        self.signatures = signature_objects
        print(f"✅ Computed {len(self.signatures)} signatures.")
        
        self._compute_signature_activities()

    def compute_reader_activities(self, freq=DEFAULT_FREQ):
        expanded = self.df_tijs.loc[
            self.df_tijs.index.repeat(self.df_tijs["signature"].str.len())
        ].copy()
        expanded["reader"] = list(chain.from_iterable(self.df_tijs["signature"]))
        expanded["datetime"] = pd.to_datetime(expanded["t"], unit="s")


        for reader_id, reader_obj in self.readers.items():
            series = (
                expanded[expanded["reader"] == reader_id]
                .set_index("datetime")
                .resample(freq)
                .size()
            )
            reader_obj.activity_rel = (series / self.activity_max).fillna(0)

    def _compute_signature_activities(self, freq=DEFAULT_FREQ):


        for sig_obj in self.signatures.values():
            matching_rows = self.df_tijs[
                self.df_tijs["signature"].apply(lambda s: set(s) == set(sig_obj.id))
            ]

            if matching_rows.empty:
                sig_obj.activity = None
                continue

            series = matching_rows.set_index("datetime").resample(freq).size()
            sig_obj.activity_rel = (series / self.activity_max).fillna(0)
            
            sig_obj.activity = series.fillna(0)
            


    def define_areas_by_group(self, signature_group):
    
        def hex_to_rgba(hex_color, alpha=1):
            rgb = pc.hex_to_rgb(hex_color)
            return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
    
        colors = px.colors.qualitative.D3
        color_cycle = itertools.cycle(colors)
    
        self.areas = {}
        area_numbers = signature_group.unique()
    
        for area_number in area_numbers:
            raw_names = signature_group[signature_group == area_number].index
    
            # Parse signature keys
            signature_names = []
            for name in raw_names:
                if isinstance(name, str) and name.startswith("(") and name.endswith(")"):
                    try:
                        name = ast.literal_eval(name)
                    except:
                        pass
                signature_names.append(name)
    
            # Get Signature objects
            signature_objs_in_area = {name: self.signatures[name] for name in signature_names}
    
            # === Compute dominant plan by activity-weighted vote ===
            plan_weights = defaultdict(float)
            for sig in signature_objs_in_area.values():
                if sig.dominant_plan is None:
                    continue
                activity = np.sum(sig.activity) if hasattr(sig, "activity") else 0
                plan_weights[sig.dominant_plan] += activity
    
            if plan_weights:
                dominant_plan = max(plan_weights, key=plan_weights.get)
            else:
                dominant_plan = "UnknownPlan"
    
            # Generate name and color
            area_name = f"{dominant_plan}_{area_number}"
            area_color = hex_to_rgba(next(color_cycle))
    
            area = Area(
            area_number,
            area_name,
            signature_objs_in_area,
            self.activity_max,
            area_color
        )
            self.areas[area_number] = area


    def compute_agents_entropies(self):
        count = 0
        for agent in self.agents.values():
            if hasattr(agent, "compute_entropy"):
                agent.compute_entropy()
                count += 1
        print(f"✅ Computed entropy for {count} agents.")

                
    def assign_agents_to_areas_over_time(self, freq=DEFAULT_FREQ):
        """
        Assign each agent a time series of area_ids based on their observed interactions.
        Combines both 'i' and 'j' roles, maps signatures to areas, and resamples over time.
          
        The result is stored in:
            self.agents[agent_id].trajectorie : pd.Series indexed by time
            self.signature_to_area : mapping from signature → area ID
        """
        signature_to_area = {
            tuple(sig): area.id for area in self.areas.values() for sig in area.signatures
        }
          

        df = self.df_tijs.copy()
        df["datetime"] = pd.to_datetime(df["t"], unit="s")
          
        agent_df = pd.melt(
            df,
            id_vars=["datetime", "signature"],
            value_vars=["i", "j"],
            var_name="role",
            value_name="agent_id"
        )

        agent_df["signature"] = agent_df["signature"].apply(
            lambda s: tuple(s) if not isinstance(s, tuple) else s
        )

        agent_df["area_id"] = agent_df["signature"].apply(signature_to_area.get)
          

        agent_df.dropna(subset=["area_id"], inplace=True)
        agent_df.set_index("datetime", inplace=True)
        self.agents = {}
        for agent_id, group in agent_df.groupby("agent_id"):
            area_series = (
                group["area_id"]
                .resample(freq)
                .agg(lambda x: x.value_counts().idxmax() if not x.empty else None)
                .ffill()
                .bfill()
            )
            agent = Agent(str(agent_id))
            agent.trajectorie = area_series
            self.agents[agent_id] = agent
        self.signature_to_area = signature_to_area
          
        print(f"✅ Assigned {len(self.agents)} agents with position over time.")
    
    
    def compute_active_agent_to_area(self):
        """
        For each Area object, compute:
          - area.active_agents_df: DataFrame[time → List[agent_ids]]
          - area.active_matrix_df: DataFrame[time × agent] with boolean presence
        """
        agent_ids = list(self.agents.keys())
    
        for area in self.areas.values():
            presence_dict = {}
    
            for agent_id, agent in self.agents.items():
                traj = getattr(agent, "trajectorie", None)
                if traj is not None and not traj.empty:
                    mask = traj == area.id
                    times = traj[mask].index
    
                    for t in times:
                        presence_dict.setdefault(t, []).append(agent_id)
    
            if presence_dict:
                sorted_times = sorted(presence_dict)
                active_agents_df = pd.DataFrame(
                    [(t, presence_dict[t]) for t in sorted_times],
                    columns=["time", "agents"]
                ).set_index("time")
                binary_data = {
                    t: {aid: (aid in presence_dict[t]) for aid in agent_ids}
                    for t in sorted_times
                }
    
                active_matrix_df = pd.DataFrame.from_dict(binary_data, orient="index").sort_index()
    
            else:
                active_agents_df = pd.DataFrame(columns=["agents"]).set_index(pd.DatetimeIndex([]))
                active_matrix_df = pd.DataFrame(index=pd.DatetimeIndex([]), columns=agent_ids)


            area.active_agents_df = active_agents_df
            area.active_matrix_df = active_matrix_df
    
    