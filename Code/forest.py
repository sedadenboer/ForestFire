import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from vegetation import Plant
from visualize import visualize


class Forest:
    # cell states and neighbor indices (Moore neighborhood)
    EMPTY = 0
    TREE = 1
    FIRE = 2
    MOORE_NEIGHBORS = ((-1,-1), (-1,0), (-1,1), (0,-1), (0, 1), (1,-1), (1,0), (1,1))
        
    def __init__(self, dimension, density, burnup_chance, visualize):
        # parameters
        self.dimension = dimension
        self.density = density
        self.burnup_chance = burnup_chance
        self.moore_coords = Forest.MOORE_NEIGHBORS
        self.grid = self.make_grid()
        self.frames = [self.get_forest_state()]
        self.visualize = visualize

    def make_grid(self):
        grid = np.empty(shape=(self.dimension, self.dimension), dtype=object)

        for i in range(self.dimension):
            for j in range(self.dimension):
                grid[i][j] = Plant(Forest.EMPTY)

        trees_number = round((self.dimension ** 2) * self.density)
        tree_indices = np.random.choice(
            range(self.dimension * self.dimension),
            trees_number,
            replace=False
        )

        for index in tree_indices:
            row = index // self.dimension
            col = index % self.dimension
            grid[row][col] = Plant(Forest.TREE)

        return grid

    def start_fire(self):
        tree = self.get_random_tree()
        tree.change_state(Forest.FIRE)

    def get_random_tree(self):
        while True:
            row = np.random.randint(0, self.dimension)
            col = np.random.randint(0, self.dimension)
            plant = self.grid[row][col]
            if plant.is_tree():
                return plant

    def get_lit_neighbors(self, row, col):
        lit_neighbors = 0
        neighbors = 0

        for neighbor in self.moore_coords:
            vert, hor = neighbor
            vert_neighbor = row + vert
            hor_neighbor = col + hor

            if (
                vert_neighbor >= 0 and vert_neighbor < self.dimension and
                hor_neighbor >= 0 and hor_neighbor < self.dimension
            ):
                neighbors += 1

                if self.grid[vert_neighbor][hor_neighbor].is_burning():
                    lit_neighbors += 1

        return neighbors, lit_neighbors

    def fire_chance(self, row, col):
        total_neighbors = self.get_lit_neighbors(row, col)[0]
        lit_neighbors_num = self.get_lit_neighbors(row, col)[1]
        chance_fire = lit_neighbors_num / total_neighbors
        return chance_fire

    def check_fire_forest(self):
        for row in self.grid:
            for plant in row:
                if plant.is_burning():
                    return True
        return False

    def forest_state(self):
        for row_idx, row in enumerate(self.grid):
            for col_idx, plant in enumerate(row):
                if plant.is_empty():
                    continue

                if plant.is_burning() and np.random.uniform() <= self.burnup_chance:
                    plant.change_state(Forest.EMPTY)
                elif plant.is_tree() and np.random.uniform() <= self.fire_chance(row_idx, col_idx):
                    plant.change_state(Forest.FIRE)

    def get_forest_state(self):
        # extract states from Plant objects
        return np.array([[plant.state for plant in row] for row in self.grid])
        
    def simulate(self):
        self.start_fire()

        while self.check_fire_forest():
            self.forest_state()
            frame = self.get_forest_state()
            self.frames.append(frame.copy())

        if self.visualize:
            visualize(
                self.frames, showplot=True, saveplot=True,
                filename='simulation_animation', colors=['black', '#6E750E', 'crimson']
            )

        return self.frames