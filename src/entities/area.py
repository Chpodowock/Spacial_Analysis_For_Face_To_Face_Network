from entity import Entity
class Area(Entity):
    def __init__(self, area_id, area_name, signatures, activity_max, area_color=None):
        super().__init__(area_id, name=area_name)
        self.signatures = signatures  # dict of signature_id: Signature
        self.color = area_color
        self.activity_max = activity_max
        self.active_agent = None
        self.active_agents_df = None
        self.active_matrix_df = None
        
        self.temporal_agent_graphs = []

        self.compute_activity()


    def compute_activity(self):
        total_activity = None
        for sig in self.signatures.values():
            sig_activity = getattr(sig, 'activity', None)
            if sig_activity is not None:
                if total_activity is None:
                    total_activity = sig_activity.copy()
                else:
                    total_activity = total_activity.add(sig_activity, fill_value=0)

        if total_activity is not None:
            self.activity = total_activity.fillna(0)
            self.activity_rel = self.activity / self.activity_max
        else:
            self.activity = None
            self.activity_rel = None
            
