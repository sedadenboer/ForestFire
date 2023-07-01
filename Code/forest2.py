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
import constants

class Forest:
    # cell states and neighbor indices (Moore neighborhood)
    # EMPTY = 0
    # TREE = 1
    # GRASS = 2
    # SHRUB = 3
    # FIRE = 10
    # BURNED = -1
    MOORE_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    VON_NEUMANN_NEIGHBOURS = ((-1, 0), (0, -1), (0, 1), (1, 0))

    def __init__(self, grid_type: str, dimension: int, density: float, burnup_time: int, veg_ratio: List[float],
                 neighbourhood_type: str, visualize: bool) -> None:
        """Forest model of the region where forest fires occur. Represented by a 2D grid,
        containing "Plant" objects that represent generic trees in the basic version of the model.
        The cells can be empty, tree, fire, or burned. The state of the forest changes over time
        according to the forest fire cellular automata model, where each cell interacts
        with its Moore neighborhood.

        Args:
            grid_type (str): the layout of vegetation ('default' or 'stripe' / 'vertical' / 'random')
            dimension (int): size of the grid
            density (float): forest density
            burnup_time (int): time for a tree to burn down
            veg_ratio (List[float]): the ratio between different vegetation type
            neighbourhood_type (str): "moore" or "von_neumann"
            visualize (bool): if a visualization should be made
        """
        # parameters
        self.grid_type = grid_type
        self.dimension = 31
        self.density = density
        self.burnup_time = burnup_time
        self.veg_ratio = veg_ratio
        self.grid = self.make_grid()
        self.frames = [self.get_forest_state()]
        self.visualize = visualize
        if neighbourhood_type == "moore":
            self.neighbourhood = Forest.MOORE_NEIGHBOURS
        else:
            self.neighbourhood = Forest.VON_NEUMANN_NEIGHBOURS

    def make_grid(self) -> np.ndarray:
        """Initializes the forest with a given dimension and tree density.

        Returns:
            np.ndarray: forest grid
        """
        # make grid with "empty" Plant objects
        coordinates = [ (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 10), (2, 11), (2, 16), 
                        (3,2), (3,6), (3,10), (3, 12), (3,15), (3,17), 
                        (4,2), (4,3), (4, 4), (4,6), (4, 7), (4, 8), (4,10), (4,12),(4,13), (4,15), (4, 16), (4,17),
                        (5,4), (5, 6), (5, 10), (5, 12), (5,14), (5,18),
                        (6,2),(6,3), (6,4), (6,6), (6,7), (6,8), (6,9), (6,10), (6,11), (6,14), (6,18),
                        (7,1),
                        (8,2),
                        (9,2), (9,4), (9,7), (9,8), (9,9), (9,11), (9,13), (9,15), (9,17), (9,21), (9,25), (9, 28),
                        (10, 2), (10,3), (10,7), (10,11), (10,13), (10,15), (10,17), (10,20), (10,22), (10,25), (10,26), (10,28),
                        (11,2), (11,3), (11,7), (11,8), (11,9), (11,11), (11,12), (11,13), (11, 15), (11,17), (11,20), (11,21), (11,22), (11, 25), (11, 26), (11,28),
                        (12,2), (12,4), (12,7), (12,13), (12,15), (12,17), (12,19), (12, 23), (12, 25), (12, 27), (12, 28),
                        (13,2), (13,5), (13,7), (13,8), (13,9), (13,10), (13,12), (13,16), (13,17),(13,19), (13,23), (13,25), (13,27),(13,28),
                        (14,6),(14,11),(14,18),(14,24),(14,28),
                        (15, 10),(15,12),(15,29),
                        (16,2), (16, 3), (16, 4), (16, 6), (16, 7), (16,8), (16,10), (16,12), (16,13), (16,15), (16,18), (16,19), (16,21),(16,22), (16,24),(16,25), (16,26), (16,27), (16,28),
                        (17,2), (17,4), (17,6), (17,10), (17,12), (17,13), (17,15), (17,17), (17,20), (17,22), (17,24), (17,27),
                        (18, 2), (18,4), (18,6), (18,7), (18,8), (18,10), (18,12), (18,14), (18,15), (18,17), (18,20), (18,22), (18,24), (18,27),
                        (19,2), (19,3),(19,6), (19,10), (19,12), (19,14), (19,15), (19,17), (19,20),(19,22), (19,24), (19,27),
                        (20,2), (20,4), (20,6), (20,7), (20,8), (20,10), (20,12),(20,15), (20,16),(20,18), (20,19), (20,23), (20,24), (20,27),
                        (21,2),(21,5),(21,9),
                        (22,8),
                        (23,4), (23,8), (23,10), (23,15), (23,16), (23,17), (23,18), (23,19), (23,20),
                        (24,3), (24,5), (24,8), (24,9), (24,16), (24,18),
                        (25, 3), (25,4), (25,5), (25,8), (25,9), (25,16), (25,18), (25,19), (25,20),
                        (26,2), (26,6), (26,8), (26,10),(26,12), (26,13), (26,16), (26,18),
                        (27, 2), (27,6), (27,8), (27,11), (27,14), (27,15), (27,18), (27,19), (27,20),
                        (28,7)]
        
        grid = np.full((31, 31), Plant(constants.EMPTY), dtype=object)
        plant_type = [constants.EMPTY, constants.TREE, constants.GRASS, constants.SHRUB]

        if self.grid_type == 'default':
            # choose random spots in the grid to place trees, according to predefined density
            trees_number = len(coordinates)
            tree_indices = coordinates

        # place trees on randomly chosen cells
            for index in tree_indices:
                grid[index[0]][index[1]] = Plant(constants.TREE)


        return grid

    def get_random_plant(self) -> Plant:
        """Get a random Tree.

        Returns:
            Plant: random Tree cell
        """
        # keep searching until randomly chosen cell is a Tree
        while True:
            row = np.random.randint(0, self.dimension)
            col = np.random.randint(0, self.dimension)
            plant = self.grid[row][col]

            if plant.is_tree() or plant.is_shrub() or plant.is_grass():
                # if cell is a Tree, return the cell
                return plant

    def start_fire_randomly(self) -> None:
        """Initializes a cell of fire randomly somewhwere."""
        plant = self.get_random_plant()
        plant.change_state(constants.FIRE)

    def start_fire(self) -> None:
        """Initializes a line of fire on top of the grid."""
        # get cells on the top row of the grid
        top_row = self.grid[0]

        for cell in top_row:
            # check if the cell is a Tree and set it on fire
            if cell.is_tree():
                cell.change_state(constants.FIRE)

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
            if (vert_neighbor >= 0 and vert_neighbor < self.dimension and
                hor_neighbor >= 0 and hor_neighbor < self.dimension):
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
        total_neighbors, lit_neighbors_num = self.get_lit_neighbors(row, col)

        if self.grid_type == 'default':
            site_igni_p, site_humidity_p = 1, 1
        else:
            site_igni_p = self.grid[row, col].igni_probability()
            site_humidity_p = self.grid[row, col].humidity_effect()

        # calculate probability of catching fire (with ignition p and humidity effect)
        chance_fire = lit_neighbors_num / total_neighbors * site_igni_p * site_humidity_p
        return chance_fire

    def check_fire_forest(self) -> bool:
        """Checks if the forest is on fire somewhere.

        Returns:
            bool: if forest is on fire
        """
        # Convert the grid to a NumPy array
        grid_array = self.get_forest_state()

        # Check if the Fire value exists in the grid
        is_present = np.any(grid_array == constants.FIRE)

        if is_present:
            return True
        return False

    def get_forest_state(self) -> np.ndarray:
        """Extracts states from Plant objects and returns them in a 2D list.
        This is to have an integer representation of the grid.

        Returns:
            List[List[int]]: 2D list of all Plant states
        """
        # extract states from Plant objects
        return np.array([[plant.state for plant in row] for row in self.grid])

    def get_edge_cells(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """Retrieve edge cells of the grid.

        Returns:
            Tuple[np.array, np.array, np.array, np.array]: arrays of edge cells
        """
        top_edge = self.grid[0, :]
        bottom_edge = self.grid[-1, :]
        left_edge = self.grid[1:-1, 0]
        right_edge = self.grid[1:-1, -1]

        return [top_edge, bottom_edge, left_edge, right_edge]

    def check_percolation_bottom(self) -> bool:
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

    def check_percolation(self) -> bool:
        """Checks if fire has reached all edges of the grid.

        Returns:
            bool: True if edges contain FIRE or BURNED cells, False otherwise
        """
        top, bot, left, right = self.get_edge_cells()

        # check for fire/burned plants in edges
        fire_top = [tree for tree in top if tree.is_burned() or tree.is_burning()]
        fire_bot = [tree for tree in bot if tree.is_burned() or tree.is_burning()]
        fire_left = [tree for tree in left if tree.is_burned() or tree.is_burning()]
        fire_right = [tree for tree in right if tree.is_burned() or tree.is_burning()]

        # count the number of edges that contain(ed) fire
        n_edges_fire = sum([1 for lst in [fire_top, fire_bot, fire_left, fire_right] if len(lst) > 0])
        return n_edges_fire == 4

    def forest_decrease(self) -> float:
        """Calculates the percentage difference in trees between the first and last frame.

        Returns:
            float: ratio of final trees/intial trees
        """
        initial_trees = np.count_nonzero(self.frames[0] == constants.TREE)
        final_trees = np.count_nonzero(self.frames[-1] == constants.TREE)

        flux = (final_trees / initial_trees)

        return flux

    def update_forest_state(self) -> None:
        """Updates the state of the forest based on forest fire spread rules.
        """
        cells_to_set_on_fire = []

        for row_idx, row in enumerate(self.grid):
            for col_idx, plant in enumerate(row):
                # skip empty cells
                if plant.is_empty() or plant.is_burned():
                    continue

                # if the cell is burning and the burning counter reaches burnup time, change to burned state
                if plant.is_burning():
                    if plant.burning_time == self.burnup_time:
                        plant.change_state(constants.BURNED)
                    else:
                        # increment the burning counter
                        plant.burning_time += 1
                # if the cell is a tree and has a burning neighbor, compute the fire chance
                # to decide if adding it to the list
                elif plant.is_tree() or plant.is_grass() or plant.is_shrub():
                    ignition_p = self.fire_chance(row_idx, col_idx)
                    if np.random.random() <= ignition_p:
                        cells_to_set_on_fire.append((row_idx, col_idx))

        # set the tree cells on fire after iterating over all cells
        for cell in cells_to_set_on_fire:
            row_idx, col_idx = cell
            self.grid[row_idx][col_idx].change_state(constants.FIRE)

    def simulate(self) -> List[np.array]:
        """Simulate the forest fire spread and return the frames.

        Returns:
            List[np.array]: list of frames capturing the forest state during the simulation
        """
        # # message to user
        # print("Running simulation...")

        time = 0

        # start fire
        self.start_fire_randomly()

        # keep running until waiting time for ignition or until fires are extinguished
        while self.check_fire_forest():
            # update forest state and add current state to frames
            self.update_forest_state()
            self.frames.append(self.get_forest_state())

            time += 1

        # # finished message
        # print("Simulation completed!")

        if self.visualize:
            # visualize the simulation
            visualize(
                self.frames, showplot=True, saveplot=True,
                filename='simulation_animation'
            )

        return self.frames