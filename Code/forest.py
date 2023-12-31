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
    MOORE_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    VON_NEUMANN_NEIGHBOURS = ((-1, 0), (0, -1), (0, 1), (1, 0))

    def __init__(self, grid_type: str = 'default', dimension: int = 100, density: float = 0.7, burnup_time: int = 10, veg_ratio: List[float] = [0.5, 0.3, 0.2],
                 neighbourhood_type: str = 'moore', visualize: bool = False,
                 igni_ratio_exp: bool = False, adjust_igni_tree: float = constants.P_TREE, adjust_igni_grass: float = constants.P_GRASS, adjust_igni_shrub: float = constants.P_SHRUB) -> None:
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
            igni_ratio_exp (bool): if True, experiment for varying veg ratios and ignition measures enabled,
                                   otherwise normal ratio's and ignition probabilities used
            adjust_igni_tree (float): adjusted experimental value for the tree ignition probability
            adjust_igni_grass (float): adjusted experimental value for the grass ignition probability
            adjust_igni_shrub (float): adjusted experimental value for the shrub ignition probability
        """
        # parameters
        self.grid_type = grid_type
        self.dimension = dimension
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
        self.igni_ratio_exp = igni_ratio_exp
        self.adjust_igni_tree = adjust_igni_tree
        self.adjust_igni_grass = adjust_igni_grass
        self.adjust_igni_shrub = adjust_igni_shrub

    def make_grid(self) -> np.ndarray:
        """Initializes the forest with a given dimension and tree density.

        Returns:
            np.ndarray: forest grid
        """
        # make grid with "empty" Plant objects
        grid = np.full((self.dimension, self.dimension), Plant(constants.EMPTY), dtype=object)
        plant_type = [constants.EMPTY, constants.TREE, constants.GRASS, constants.SHRUB]

        # default settings
        if self.grid_type == 'default':
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
                grid[row][col] = Plant(constants.TREE)
        else:
            veg_grid = np.zeros((self.dimension, self.dimension))

            # place vegetation in stripe layout
            if self.grid_type == 'stripe':
                lengths = np.round(np.array(self.veg_ratio) * len(grid[:, 0])).astype(int)
                splits = np.split(np.arange(self.dimension), np.cumsum(lengths)[:-1])

                # fill in the grid
                for i, split in enumerate(splits):
                    veg_grid[split] = np.random.choice([constants.EMPTY,plant_type[i+1]],
                                                    size=(len(split), self.dimension), p=[1-self.density, self.density])
            # place vegetation in vertical layout
            elif self.grid_type == 'vertical':
                lengths = np.round(np.array(self.veg_ratio) * len(grid[:,0])).astype(int)
                splits = np.split(np.arange(self.dimension), np.cumsum(lengths)[:-1])

                # fill in the grid
                for i, split in enumerate(splits):
                    veg_grid[split] = np.random.choice([constants.EMPTY, plant_type[i + 1]],
                                                    size=(len(split), self.dimension), p=[1 - self.density, self.density])

                veg_grid = np.rot90(veg_grid)
            # place vegetation randomly
            elif self.grid_type == 'random':
                p_list = np.concatenate([np.array([1 - self.density]), np.array(self.veg_ratio) * self.density])
                veg_grid = np.random.choice(plant_type, size=(self.dimension, self.dimension), p=p_list)

            # plant plants
            for i in range(veg_grid.shape[0]):
                for j in range(veg_grid.shape[1]):
                    grid[i][j] = Plant(veg_grid[i][j])

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
                # if cell is a plant, return the cell
                return plant

    def start_fire_randomly(self) -> None:
        """Initializes a cell of fire randomly somewhwere."""
        plant = self.get_random_plant()
        plant.change_state(constants.FIRE)

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
            if 0 <= vert_neighbor < self.dimension and 0 <= hor_neighbor < self.dimension:
                # increment the count of the total neighbors
                neighbors += 1

                # check if the neighboring cell is burning
                if self.grid[vert_neighbor][hor_neighbor].is_burning():
                    lit_neighbors += 1

        return neighbors, lit_neighbors

    def fire_chance(self, row: int, col: int, ) -> float:
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
            # ignore effects of ignition or humidity probabilities
            site_igni_p, site_humidity_p = 1, 1
        else:
            # experimental ignition values
            if self.igni_ratio_exp:
                if self.grid[row, col].state == constants.TREE:
                    self.grid[row, col].set_ignition(self.adjust_igni_tree)
                elif self.grid[row, col].state == constants.GRASS:
                    self.grid[row, col].set_ignition(self.adjust_igni_grass)
                elif self.grid[row, col].state == constants.SHRUB:
                    self.grid[row, col].set_ignition(self.adjust_igni_shrub)
                 
            # standard ignition and humidity values
            site_igni_p = self.grid[row, col].ignition
            site_humidity_p = self.grid[row, col].humidity

        # calculate probability of catching fire (with ignition p and humidity effect)
        chance_fire = lit_neighbors_num / total_neighbors * site_igni_p * site_humidity_p
        return chance_fire

    def check_fire_forest(self) -> bool:
        """Checks if the forest is on fire somewhere.

        Returns:
            bool: if forest is on fire
        """
        # convert the grid to a NumPy array
        grid_array = self.get_forest_state()

        # check if the Fire value exists in the grid
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

    def burned_area(self) -> float:
        """Calculates the relative amount of burned trees at the
        end of the simulation.

        Returns:
            float: burned area ratio
        """
        return np.count_nonzero(self.frames[-1] == constants.BURNED) / (self.dimension ** 2)

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