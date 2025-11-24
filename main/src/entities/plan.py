

class Plan:
    def __init__(self, name, image, readers_dict):
        self.name = name
        self.image = image
        self.size = image.size
        self.readers = readers_dict