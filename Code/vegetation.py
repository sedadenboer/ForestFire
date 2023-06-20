# vegetation.py
#
# Course: Complex System Simulation, UvA
# Year: 2022/2023
# 
# Description: This module defines the Plant class which
# represents the vegetation in the forest. Contains functions
# to change the Plant state and to check for its possible states.


class Plant:
    def __init__(self, state):
        """Initialize a Plant object with the given state.

        Args:
            state (int): state of the plant
        """
        self.state = state

    def change_state(self, new_state: int) -> None:
        """Change the state of the plant.

        Args:
            new_state (int): the new state to assign to the plant
        """
        self.state = new_state

    def is_burning(self) -> bool:
        """Check if the plant is burning.

        Returns:
            bool: True if the plant is burning, False otherwise
        """
        return self.state == 2

    def is_tree(self) -> bool:
        """Check if the plant is a tree.

        Returns:
            bool: True if the plant is a tree, False otherwise
        """
        return self.state == 1

    def is_empty(self) -> bool:
        """Check if the plant is empty.

        Returns:
            bool: True if the plant is empty, False otherwise
        """
        return self.state == 0

    def __repr__(self) -> str:
        """Return a string representation of the plant.

        Returns:
            str: string representation of the plant
        """
        return str(self.state)
