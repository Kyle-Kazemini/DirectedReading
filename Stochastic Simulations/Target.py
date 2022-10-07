class Target:
    def __init__(self, coordinates, resources, number):
        self.coordinates = coordinates
        self.resources = resources
        self.number = number

    def __str__(self):
        return f'Target number: {self.number}, coordinates: {self.coordinates}, resources: {self.resources}'
