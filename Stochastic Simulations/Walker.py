from Constants import *


class Walker:
    def __init__(self, x, y, resources=0):
        self.x = x
        self.y = y
        self.resources = resources
        self.coordinates = (x, y)

    def move(self, rand):
        if rand <= 0.25 and self.x > -X_GRID_SIZE:
            self.x -= 1
        elif 0.25 < rand <= 0.5 and self.y < Y_GRID_SIZE:
            self.y += 1
        elif 0.5 < rand <= 0.75 and self.x < X_GRID_SIZE:
            self.x += 1
        elif 0.75 < rand <= 1 and self.y > -Y_GRID_SIZE:
            self.y -= 1
        self.coordinates = (self.x, self.y)

    def __str__(self):
        return f'Walker coordinates: {self.coordinates}, resources: {self.resources}'
