# visualize.py
#
# Course: Complex System Simulation, UvA
# Year: 2022/2023
# 
# Description: Contains a visualization function
# for the spread of a forest fire on a grid.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as c
import matplotlib.animation as animation
from typing import List


def visualize(grid_values: List[List[int]], showplot: bool = True, saveplot: bool = True, 
              filename: str = 'simulation_animation', colors: List[str] = ['black', 'green', 'red']):
    """Animates the Cellular automata simulation result.

    Args:
        grid_values (List[np.ndarray]): a list of the grids with Plant states generated during the simulation
        showplot (bool, optional): show the visualization (defaults to True)
        saveplot (bool, optional): saves the visualization as a gif (defaults to False)
        filename (str, optional): filename used to save animation (defaults to 'simulation_animation')
        colors (List[str], optional): colors used in animation, where ength of list must correspond with number of 
                                      unique states in grid (defaults to ['black', 'green', 'red'])
    """

    # set up figure and colors
    fig = plt.figure(figsize=(8, 8))
    cmap = c.ListedColormap(colors)

    # plot frames
    ims = [[plt.imshow(grid, vmin=0, vmax=len(colors), cmap=cmap, animated=True)] for grid in grid_values]

    plt.axis('off')
    plt.tight_layout()

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    if saveplot:
        ani.save('Output/' + filename + '.gif', writer=animation.PillowWriter(fps=60))

    if showplot:
        plt.show()
