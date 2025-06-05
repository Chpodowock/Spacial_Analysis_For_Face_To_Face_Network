class Entity:
    def __init__(self, entity_id, name=None):
        self.id = entity_id
        self.name = name or str(entity_id)
        self.color = None
        self.activity = None
        self.activity_rel = None

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.id} name={self.name}>"
