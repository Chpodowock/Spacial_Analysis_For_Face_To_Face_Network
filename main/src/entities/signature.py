from shapely.geometry import Point, LineString, MultiPoint
from collections import Counter
from entity import Entity


class Signature(Entity):
    def __init__(self, signature_id, readers, threshold=0.4):
        """
        Initialize a Signature object.

        Args:
            signature_id (tuple): Tuple of reader IDs composing the signature.
            readers (dict): Dictionary mapping reader IDs to Reader objects.
            threshold (float): Threshold for dominant plan attribution.
        """
        super().__init__(entity_id=signature_id, name=f"Signature-{','.join(signature_id)}")
        self.readers = [readers[r_id] for r_id in signature_id if r_id in readers]

        # Plan affiliation
        self.plan_counts = self._count_readers_per_plan()
        self.plans_involved = set(self.plan_counts)
        self.num_plans_involved = len(self.plans_involved)

        # Dominant plan logic
        self.dominant_plan, self.dominant_plan_ratio = self._get_dominant_plan(threshold)
        self.multi_plan = self.num_plans_involved > 1 and self.dominant_plan is None

        # Readers in dominant plan
        self.readers_in_dominant_plan = self._get_readers_in_dominant_plan()

        # Geometry
        self.dominant_plan_polygon = self._compute_dominant_plan_polygon()

        # Activity metrics inherited from Entity
        self.activity = 0
        self.activity_rel = 0

    def __repr__(self):
        return (
            f"<Signature id={self.id} "
            f"dominant_plan={self.dominant_plan} "
            f"readers={len(self.readers)}>"
        )

    def _count_readers_per_plan(self):
        """Count how many readers in the signature belong to each plan."""
        return dict(Counter(r.plan_name for r in self.readers if hasattr(r, "plan_name")))

    def _get_dominant_plan(self, threshold):
        """
        Determines dominant plan based on reader distribution.

        Returns:
            (str or None, float): (Dominant plan name or None, ratio value)
        """
        total = len(self.readers)
        if total == 0 or not self.plan_counts:
            return None, 0.0

        dominant_plan = max(self.plan_counts, key=self.plan_counts.get)
        ratio = self.plan_counts[dominant_plan] / total
        return (dominant_plan, ratio) if ratio >= threshold else (None, ratio)

    def _get_readers_in_dominant_plan(self):
        """Return the subset of readers in the dominant plan."""
        if not self.dominant_plan:
            return []
        return [r for r in self.readers if getattr(r, "plan_name", None) == self.dominant_plan]

    def _compute_dominant_plan_polygon(self, min_size=0.05, smooth_buffer=0.05, verbose=True):
        """
        Compute a convex hull or buffered shape from readers in the dominant plan.

        Returns:
            shapely.geometry.Polygon or None
        """
        if not self.dominant_plan:
            if verbose:
                print(f"[Signature {self.id}] No dominant plan assigned.")
            return None

        positions = [
            (r.x_rel, r.y_rel)
            for r in self.readers_in_dominant_plan
            if hasattr(r, "x_rel") and hasattr(r, "y_rel")
        ]

        if not positions:
            return None
        elif len(positions) == 1:
            return Point(positions[0]).buffer(min_size)
        elif len(positions) == 2:
            return LineString(positions).buffer(min_size)
        else:
            hull = MultiPoint(positions).convex_hull
            return hull.buffer(smooth_buffer) if smooth_buffer > 0 else hull
