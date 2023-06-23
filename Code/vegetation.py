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

    def igni_probability(self) -> float:
        """Return the probability associated with vegetation

        Returns:
            float: vegetation ignition probability
        """
        if self.state == constants.TREE:  # tree
            return constants.P_TREE
        elif self.state == constants.GRASS:  # grass
            return constants.P_GRASS
        elif self.state == constants.SHRUB:  # shrub
            return constants.P_SHRUB

    def humidity_effect(self) -> float:
        """Return the humidity effect associated with vegetation

        Returns:
            float: vegetation humidity effect probability
        """
        if self.state == constants.TREE:  # tree
            return constants.H_TREE
        elif self.state == constants.GRASS:  # grass
            return constants.H_GRASS
        elif self.state == constants.SHRUB:  # shrub
            return constants.H_SHRUB

    def __repr__(self) -> str:
        """Return a string representation of the plant.

        Returns:
            str: string representation of the plant
        """
        return str(self.state)