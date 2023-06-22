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
        return self.state == 10

    def is_tree(self) -> int:
        """Check if the plant is a tree.

        Returns:
            int: True if the plant is a tree, False otherwise
        """
        return self.state == 1
    
    def is_grass(self) -> int:
        """Check if the plant is a grass.

        Returns:
            int: True if the plant is a grass, False otherwise
        """
        return self.state == 2
    
    def is_shrub(self) -> int:
        """Check if the plant is a shrub.

        Returns:
            int: True if the plant is a shrub, False otherwise
        """
        return self.state == 3

    def is_burned(self) -> int:
        """Check if the plant is burned.

        Returns:
            int: True if the plant is burned, False otherwise
        """
        return self.state == -1
    
    def is_empty(self) -> bool:
        """Check if the plant is empty.

        Returns:
            bool: True if the plant is empty, False otherwise
        """
        return self.state == 0
    
    def igni_probability(self) -> float:
        """Return the probability associated with vegetation

        Returns:
            float: vegetation ignition probability
        """
        if self.state == 1: # tree
            return constants.TREE
        elif self.state == 2: # grass
            return constants.GRASS
        elif self.state == 3: # shrub
            return constants.SHRUB

    def __repr__(self) -> str:
        """Return a string representation of the plant.

        Returns:
            str: string representation of the plant
        """
        return str(self.state)
