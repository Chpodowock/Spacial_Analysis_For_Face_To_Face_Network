from entity import Entity
class Reader(Entity):
    def __init__(self, reader_id, x_rel=None, y_rel=None, plan_name=None):
        super().__init__(reader_id)
        self.x_rel = x_rel
        self.y_rel = y_rel
        self.plan_name = plan_name

    def get_absolute_position(self, plan_size):
        width, height = plan_size
        return self.x_rel * width, self.y_rel * height