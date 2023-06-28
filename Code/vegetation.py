# vegetation.py
#
# Course: Complex System Simulation, UvA
# Year: 2022/2023
# 
# Description: This module defines the Plant class which
# represents the vegetation in the forest. Contains functions
# to change the Plant state and to check for its possible states.

import constants


class Plant:
    def __init__(self, state):
        """Initialize a Plant object with the given state.

        Args:
            state (int): state of the plant
        """
        self.state = state
        self.burning_time = 0

        # initialize ignition probability
        if self.state == constants.TREE:
            self.ignition = constants.P_TREE
        elif self.state == constants.GRASS:
           self.ignition = constants.P_GRASS
        elif self.state == constants.SHRUB:
            self.ignition = constants.P_SHRUB

        # initialize humidity
        if self.state == constants.TREE: 
            self.humidity = constants.H_TREE
        elif self.state == constants.GRASS:
            self.humidity = constants.H_GRASS
        elif self.state == constants.SHRUB:
            self.humidity = constants.H_SHRUB

    def change_state(self, new_state: int) -> None:
        """Change the state of the plant.

        Args:
            new_state (int): the new state to assign to the plant
        """
        self.state = new_state

    def is_burning(self) -> int:
        """Check if the plant is burning.

        Returns:
            int: True if the plant is burning, False otherwise
        """
        return self.state == constants.FIRE

    def is_tree(self) -> int:
        """Check if the plant is a tree.

        Returns:
            int: True if the plant is a tree, False otherwise
        """
        return self.state == constants.TREE

    def is_grass(self) -> int:
        """Check if the plant is a grass.

        Returns:
            int: True if the plant is a grass, False otherwise
        """
        return self.state == constants.GRASS

    def is_shrub(self) -> int:
        """Check if the plant is a shrub.

        Returns:
            int: True if the plant is a shrub, False otherwise
        """
        return self.state == constants.SHRUB

    def is_burned(self) -> int:
        """Check if the plant is burned.

        Returns:
            int: True if the plant is burned, False otherwise
        """
        return self.state == constants.BURNED

    def is_empty(self) -> bool:
        """Check if the plant is empty.

        Returns:
            bool: True if the plant is empty, False otherwise
        """
        return self.state == constants.EMPTY

    def set_ignition(self, value: float) -> float:
        """Change the probability associated with vegetation.
        
        Args:
            value (float): new ignition probability
        """
        self.ignition = value

    def set_humidity(self, value: float) -> float:
        """Change the humidity effect associated with vegetation.
        
        Args:
            value (float): new humidity value
        """
        self.humidity = value

    def __repr__(self) -> str:
        """Return a string representation of the plant.

        Returns:
            str: string representation of the plant
        """
        return str(self.state)