import random
import numpy as np
from Target import *
from Walker import *


split = np.zeros(shape=NUM_TARGETS)
walker = targets = None

for i in range(SPLIT_RUNS):
    walker = Walker(X_START, Y_START)
    targets = {}
    for t in range(NUM_TARGETS):
        newTarget = Target((random.randint(-X_GRID_SIZE, Y_GRID_SIZE),
                            random.randint(-X_GRID_SIZE, Y_GRID_SIZE)),
                           RESOURCES, t + 1)
        targets[newTarget.coordinates] = newTarget

    for j in range(RUNS):
        rand = random.uniform(0, 1)
        walker.move(rand)

        # If the walker lands on a target and the target has a resource, the walker takes it.
        if walker.coordinates in targets.keys() and targets[walker.coordinates].resources > 0:
            walker.resources += 1
            targets[walker.coordinates].resources -= 1

            if targets[walker.coordinates].resources == 0:
                index = targets.get(walker.coordinates).number - 1  # Number is 1:N and an index needs to be 0:(N-1)
                # If one of the targets reaches 0 resources, track it and restart
                split[index] += 1
                break

print(walker)
print(split)
for i in targets.values():
    print(i)
