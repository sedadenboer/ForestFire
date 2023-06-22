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
    BURNED = 3
    MOORE_NEIGHBOURS = ((-1,-1), (-1,0), (-1,1), (0,-1), (0, 1), (1,-1), (1,0), (1,1))
    VON_NEUMANN_NEIGHBOURS = ((-1, 0), (0, -1), (0, 1), (1, 0))
    
    def __init__(self, default: bool, dimension: int, density: float, burnup_time: int, neighbourhood_type: str, visualize: bool) -> None:
        """Forest model of the region where forest fires occur. Represented by a 2D grid,
        containing "Plant" objects that represent generic trees in the basic version of the model.
        The cells can be empty, tree, fire, or burned. The state of the forest changes over time
        according to the forest fire cellular automata model, where each cell interacts
        with its Moore neighborhood.

        Args:
            default (bool): runs default version with fire_chance=1
            dimension (int): size of the grid
            density (float): forest density
            burnup_time (int): time for a tree to burn down
            neighbourhood_type (str): "moore" or "von_neumann"
            visualize (bool): if a visualization should be made
        """
        # parameters
        self.default = default
        self.dimension = dimension
        self.density = density
        self.burnup_time = burnup_time
        self.flux = 0
        self.grid = self.make_grid()
        self.frames = [self.get_forest_state()]
        self.visualize = visualize
        if neighbourhood_type == "moore":
            self.neighbourhood = Forest.MOORE_NEIGHBOURS
        else:
            self.neighbourhood = Forest.VON_NEUMANN_NEIGHBOURS

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
        """Initializes a line of fire on top of the grid."""
        # get cells on the top row of the grid
        top_row = self.grid[0]

        for cell in top_row:
            # check if the cell is a Tree and set it on fire
            if cell.is_tree():
                cell.change_state(Forest.FIRE)

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
        for neighbor in self.neighbourhood:
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
        # Convert the grid to a NumPy array
        grid_array = self.get_forest_state()

        # Check if the value 2 exists in the grid
        is_present = np.any(grid_array == 2)

        if is_present:
            return True
        return False

    def update_forest_state(self) -> None:
        """Updates the state of the forest based on forest fire spread rules.
        """
        cells_to_set_on_fire = []

        for row_idx, row in enumerate(self.grid):
            for col_idx, plant in enumerate(row):
                # skip empty cells
                if plant.is_empty():
                    continue

                # if the cell is burning and the burning counter reaches burnup time, change to burned state
                if plant.is_burning():
                    if plant.burning_time == self.burnup_time:
                        plant.change_state(Forest.BURNED)
                    else:
                        # increment the burning counter
                        plant.burning_time += 1
                # if the cell is a tree and has a burning neighbor, add it to the list
                elif plant.is_tree() and self.get_lit_neighbors(row_idx, col_idx)[1] > 0:
                    cells_to_set_on_fire.append((row_idx, col_idx))

        # set the tree cells on fire after iterating over all cells
        for cell in cells_to_set_on_fire:
            row_idx, col_idx = cell
            self.grid[row_idx][col_idx].change_state(Forest.FIRE)
    
    def get_forest_state(self) -> List[List[int]]:
        """Extracts states from Plant objects and returns them in a 2D list.
        This is to have an integer representation of the grid.

        Returns:
            List[List[int]]: 2D list of all Plant states
        """
        # extract states from Plant objects
        return np.array([[plant.state for plant in row] for row in self.grid])
    
    def check_percolation(self) -> bool:
        """Checks if the bottom row of self.grid contains any FIRE or BURNED cells.

        Returns:
            bool: True if the bottom row contains FIRE or BURNED cells, False otherwise.
        """
        # get cells on the bottom row of the grid
        bottom_row = self.grid[-1]

        for cell in bottom_row:
            # check if a cell is on fire or has been burned
            if cell.is_burning() or cell.is_burned():
                return True
        
        return False
    
    def simulate(self) -> List[List[int]]:
        """Simulate the forest fire spread and return the frames.

        Args:
            waiting_time (int): time to wait for a forest fire to develop

        Returns:
            List[List[int]]: list of frames capturing the forest state during the simulation
        """
        # # message to user
        # print("Running simulation...")

        time = 0

        # start fire
        self.start_fire()
        
        # keep running until waiting time for ignition or until fires are extinguished
        while self.check_fire_forest():
            # update forest state and add current state to frames
            self.update_forest_state()
            self.frames.append(self.get_forest_state())
            
            time += 1

        if self.visualize:
            # visualize the simulation
            visualize(
                self.frames, showplot=True, saveplot=True,
                filename='simulation_animation', colors=['tan', '#6E750E', 'crimson', 'black']
            )

        return self.frames

    def forest_decrease(self) -> float:
        """Calculates the percentage difference in trees between the first and last frame.

        Returns:
            float: percentage tree decrease
        """

        initial_trees = 0
        final_trees = 0
        for row_idx, row in enumerate(self.frames[0]):
            for col_idx, plant in enumerate(row):
                if plant.is_tree():
                    initial_trees += 1

        for row_idx, row in enumerate(self.frames[-1]):
            for col_idx, plant in enumerate(row):
                if plant.is_tree():
                    final_trees += 1
                    
        self.flux = (initial_trees - final_trees) / initial_trees * 100