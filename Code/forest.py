# forest.py
#
# Course: Complex System Simulation, UvA
# Year: 2022/2023
# 
# Description: Contains the Forest class that represents the simulation
# area for forest fires to occur. It contains methods for initializing
# the forest, starting a fire, updating the state of the forest, and visualizing
# the simulation.
# 
# Dependencies: vegetation.py, visualize.py

import numpy as np
from vegetation import Plant
from visualize import visualize
from typing import List, Tuple


class Forest:
    # cell states and neighbor indices (Moore neighborhood)
    EMPTY = 0
    TREE = 1
    FIRE = 2
    MOORE_NEIGHBORS = ((-1,-1), (-1,0), (-1,1), (0,-1), (0, 1), (1,-1), (1,0), (1,1))
    
    def __init__(self, dimension: int, density: float, burnup_chance: float, visualize: bool) -> None:
        """Forest model of the region where forest fires occur. Represented by a 2D grid,
        containing "Plant" objects that represent generic trees in the basic version of the model.
        The cells can be empty, tree, or a tree on fire. The state of the forest changes over time
        according to the forest fire cellular automata model, where each cell interacts
        with its Moore neighborhood.

        Args:
            dimension (int): size of the grid
            density (float): forest density
            burnup_chance (float): chance for a tree to burn down in a timestep
            visualize (bool): if a visualization should be made
        """
        # parameters
        self.dimension = dimension
        self.density = density
        self.burnup_chance = burnup_chance
        self.moore_coords = Forest.MOORE_NEIGHBORS
        self.grid = self.make_grid()
        self.frames = [self.get_forest_state()]
        self.visualize = visualize

    def make_grid(self) -> List[List[Plant]]:
        """Initializes the forest with a given dimension and tree density.

        Returns:
            List[List[Plant]]: forest grid
        """
        # initialize empty grid
        grid = np.empty(shape=(self.dimension, self.dimension), dtype=object)

        # fill grid with "empty" Plant objects
        for i in range(self.dimension):
            for j in range(self.dimension):
                grid[i][j] = Plant(Forest.EMPTY)

        # choose random spots in the grid to place trees, according to predefined density
        trees_number = round((self.dimension ** 2) * self.density)
        tree_indices = np.random.choice(
            range(self.dimension * self.dimension),
            trees_number,
            replace=False
        )

        # place trees on randomly chosen cells
        for index in tree_indices:
            row = index // self.dimension
            col = index % self.dimension
            grid[row][col] = Plant(Forest.TREE)

        return grid

    def start_fire(self) -> None:
        """Initializes the fire in a random cell
        by setting the Tree's state to fire.
        """
        tree = self.get_random_tree()
        tree.change_state(Forest.FIRE)

    def get_random_tree(self) -> Plant:
        """Get a random Tree.

        Returns:
            Plant: random Tree cell
        """
        # keep searching until randomly chosen cell is a Tree
        while True:
            row = np.random.randint(0, self.dimension)
            col = np.random.randint(0, self.dimension)
            plant = self.grid[row][col]

            # if cell is a Tree, return the cell
            if plant.is_tree():
                return plant

    def get_lit_neighbors(self, row: int, col: int) -> Tuple[int, int]:
        """Calculates the number of neighboring cells and lit
        (burning) neighboring cells for a given cell.

        Args:
            row (int): row coordinate
            col (int): column coordinate

        Returns:
            Tuple[int, int]: count of total neighbors and lit neighbors
        """
        lit_neighbors = 0
        neighbors = 0

        # iterate over moore neighborhood coordinates
        for neighbor in self.moore_coords:
            # extract vertical and horizontal displacements
            vert, hor = neighbor
            vert_neighbor = row + vert
            hor_neighbor = col + hor

            # check if the neighboring cell is within the grid bounds
            if (
                vert_neighbor >= 0 and vert_neighbor < self.dimension and
                hor_neighbor >= 0 and hor_neighbor < self.dimension
            ):
                # increment the count of the total neighbors
                neighbors += 1

                # check if the neighboring cell is burning
                if self.grid[vert_neighbor][hor_neighbor].is_burning():
                    lit_neighbors += 1

        return neighbors, lit_neighbors

    def fire_chance(self, row: int, col: int) -> float:
        """Calculates the probability of a cell catching fire based on the number of lit (burning)
        neighboring cells.

        Args:
            row (int): row coordinate
            col (int): column coordinate

        Returns:
            float: probability of the cell catching fire
        """
        # get count of total neighbors and lit neighbors
        total_neighbors = self.get_lit_neighbors(row, col)[0]
        lit_neighbors_num = self.get_lit_neighbors(row, col)[1]

        # calculate probability of catching fire
        chance_fire = lit_neighbors_num / total_neighbors
        return chance_fire

    def check_fire_forest(self) -> bool:
        """Checks if the forest is on fire somewhere.

        Returns:
            bool: if forest is on fire
        """
        # check in all cells if a Tree is burning
        for row in self.grid:
            for plant in row:
                if plant.is_burning():
                    return True
        return False

    def forest_state(self) -> None:
        """Updates the state of the forest based on forest fire spread rules.
        """
        for row_idx, row in enumerate(self.grid):
            for col_idx, plant in enumerate(row):
                # skip empty cells
                if plant.is_empty():
                    continue
                
                # if burning cell and random number <= burnup chance, change to empty
                if plant.is_burning() and np.random.uniform() <= self.burnup_chance:
                    plant.change_state(Forest.EMPTY)
                # if tree cell and random number <= fire chance, change to burning
                elif plant.is_tree() and np.random.uniform() <= self.fire_chance(row_idx, col_idx):
                    plant.change_state(Forest.FIRE)

    def get_forest_state(self) -> List[List[int]]:
        """Extracts states from Plant objects and returns them in a 2D list.
        This is to have an integer representation of the grid.

        Returns:
            List[List[int]]: 2D list of all Plant states
        """
        # extract states from Plant objects
        return [[plant.state for plant in row] for row in self.grid]
        
    def simulate(self) -> List[List[int]]:
        """Simulate the forest fire spread and return the frames.

        Returns:
            List[List[int]]: list of frames capturing the forest state during the simulation
        """
        # message to user
        print("Running simulation...")

        # start the fire
        self.start_fire()

        while self.check_fire_forest():
            # update forest state 
            self.forest_state()

            # add current state to frames
            self.frames.append(self.get_forest_state())
        
        # finished message
        print("Simulation completed!")

        if self.visualize:
            # visualize the simulation
            visualize(
                self.frames, showplot=True, saveplot=True,
                filename='simulation_animation', colors=['black', '#6E750E', 'crimson']
            )

        return self.frames
