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
    MOORE_NEIGHBORS = ((-1,-1), (-1,0), (-1,1), (0,-1), (0, 1), (1,-1), (1,0), (1,1))
    wind_factor = 0.1  # an arbitrary factor to control the influence of wind

    def __init__(self, dimension: int, density: float, burnup_time: int, ignition_chance: float, visualize: bool, double_cel: bool, wind_direction: str, wind_factor: float) -> None:
        # parameters
        self.dimension = dimension
        self.density = density
        self.burnup_time = burnup_time
        self.ignition_chance = ignition_chance
        self.wind_direction = wind_direction
        self.wind_factor = wind_factor
        self.grid = self.make_grid()
        self.frames = [self.get_forest_state()]
        self.visualize = visualize
        self.double_cel = double_cel

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
        """Initializes a fire in a random cell
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
        if self.double_cel == False:

            for neighbor in Forest.MOORE_NEIGHBORS:
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
        else :

            for neighbor in Forest.MOORE_NEIGHBORS_DOUBLE:

                for neighbor in Forest.MOORE_NEIGHBORS:
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

    def get_fire_prob(self, row: int, col: int, wind_direction: str, wind_speed: float) -> float:
        fire_prob = 0
        neighbors, lit_neighbors = self.get_lit_neighbors(row, col)

        fire_prob += lit_neighbors / neighbors

        # Increase or decrease probability based on wind direction and speed
        wind_prob = 0
        if wind_direction == 'N' and row < self.dimension - 1:
            wind_prob += self.wind_factor  # North cell
        if wind_direction == 'S' and row > 0:
            wind_prob -= self.wind_factor  # South cell
        if wind_direction == 'E' and col < self.dimension - 1:
            wind_prob += self.wind_factor  # East cell
        if wind_direction == 'W' and col > 0:
            wind_prob -= self.wind_factor  # West cell

        # Scale wind influence by wind speed
        wind_prob *= wind_speed

        # Ensure total probability is within [0, 1]
        fire_prob = max(0, min(1, fire_prob + wind_prob))

        return fire_prob

    def update_probabilities(self, burning_cells):
        for cell in burning_cells:
            y, x = cell
            if self.wind_direction == 'N':
                if y > 0: self.grid[y-1, x] += self.wind_factor  # North cell
                if y > 1: self.grid[y-2, x] += self.wind_factor / 2  # North cell 2nd row
                if y < self.grid.shape[0] - 1: self.grid[y+1, x] -= self.wind_factor  # South cell
            elif self.wind_direction == 'E':
                if x < self.grid.shape[1] - 1: self.grid[y, x+1] += self.wind_factor  # East cell
                if x > 0: self.grid[y, x-1] -= self.wind_factor  # West cell
            elif self.wind_direction == 'S':
                if y < self.grid.shape[0] - 1: self.grid[y+1, x] += self.wind_factor  # South cell
                if y > 0: self.grid[y-1, x] -= self.wind_factor  # North cell
            elif self.wind_direction == 'W':
                if x > 0: self.grid[y, x-1] += self.wind_factor  # West cell
                if x < self.grid.shape[1] - 1: self.grid[y, x+1] -= self.wind_factor  # East cell
        self.grid = np.clip(self.grid, 0, 1)  # Probabilities must stay between 0 and 1.

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

    def update_forest_state(self) -> None:
        """Updates the state of the forest based on forest fire spread rules.
        """
        for row_idx, row in enumerate(self.grid):
            for col_idx, plant in enumerate(row):
                # skip empty cells
                if plant.is_empty():
                    continue
                
                # if burning cell and random number <= burnup chance, change to empty
                if plant.is_burning():
                    # increment the burning counter
                    plant.burning_time += 1

                    # if the burning counter reaches burnup time, change to burned state
                    if plant.burning_time == self.burnup_time:
                        plant.change_state(Forest.BURNED)
                # if tree cell and random number <= fire chance, change to burning
                elif plant.is_tree() and np.random.uniform() <= self.get_fire_prob(row_idx, col_idx, self.wind_direction, self.wind_factor):
                    plant.change_state(Forest.FIRE)
                # probablistically start random fires 
                elif plant.is_tree() and np.random.uniform() <= self.ignition_chance:
                    plant.change_state(Forest.FIRE)

    def get_forest_state(self) -> List[List[int]]:
        """Extracts states from Plant objects and returns them in a 2D list.
        This is to have an integer representation of the grid.

        Returns:
            List[List[int]]: 2D list of all Plant states
        """
        # extract states from Plant objects
        return [[plant.state for plant in row] for row in self.grid]
    
    def simulate(self, waiting_time: int) -> List[List[int]]:
        """Simulate the forest fire spread and return the frames.

        Args:
            waiting_time (int): time to wait for a forest fire to develop

        Returns:
            List[List[int]]: list of frames capturing the forest state during the simulation
        """
        # message to user
        print("Running simulation...")

        time = 0
        
        # keep running until waiting time for ignition or until fires are extinguished
        while time <= waiting_time or self.check_fire_forest():
            # update forest state and add current state to frames
            self.update_forest_state()
            self.frames.append(self.get_forest_state())
            
            time += 1
        
        # finished message
        print("Simulation completed!")

        if self.visualize:
            # visualize the simulation
            visualize(
                self.frames, showplot=True, saveplot=True,
                filename='simulation_animation', colors=['tan', '#6E750E', 'crimson', 'black']
            )

        return self.frames