# numerical_optimization.py
#
# Course: Complex System Simulation, UvA
# Year: 2022/2023
#
# Description: The original codes are slow for large scale experiments,
# this script optimizes the process with numpy vectorization features
#
# Dependencies: vegetation.py, constants.py plot.py

import numpy as np
import constants as const
from plot import ignition_vs_ratio_heatmap
from typing import List, Union
import multiprocessing as mp
import os
import json

# burn_up grid, used to record the burn up time of each cell
burnup_t_grid = np.zeros((100,100))

# make random vegetation grid and start fire
def make_grid(density: float, dimension: int, veg_ratio: List[float]) -> List[List[int]]:
    """Initializes the forest with a given dimension and tree density.

    Args:
        dimension (int): size of the grid
        density (float): forest density
        veg_ratio (List[float]): the ratio between different vegetation type

    Returns:
        List[List[int]]: the state grid
    """
    plant_type = [const.EMPTY,const.TREE,const.GRASS,const.SHRUB]
    p_list = np.concatenate([np.array([1-density]), np.array(veg_ratio)*density])
    veg_grid = np.random.choice(plant_type, size=(dimension, dimension), p=p_list)
    
    # start fire randomly
    while True:
        random_row = np.random.randint(0, veg_grid.shape[0])
        random_column = np.random.randint(0, veg_grid.shape[1])
        
        if (veg_grid[random_row, random_column] != const.EMPTY):
            break
    veg_grid[random_row, random_column] = const.FIRE
    
    return veg_grid
    
    
def simulate(state_grid: List[List[int]], burnup_t: int, burnup_t_grid: List[List[int]]) -> Union[List[List[int]], List[List[int]]]:
    """Simulate one step of the forest fire model

    Args:
        state_grid (List[List[int]]): 2d state grid for the last time step
        burnup_t (int): the burn up time
        burnup_t_grid (List[List[int]]): 2d grid with burning time of each cell since the start of simulation to the last time step

    Returns:
        List[List[int]]: 2d state grid for the current time step
        List[List[int]]: 2d grid with burning time of each cell since the start of simulation to the current time step
    """
    
    # firstly, take the on_fire grid, which has 0 for non-fire cell and 1 for on-fire cell
    # NOTE: extra layers for boundary cells, and the boundary cells always have less ignition probability
    on_fire_grid = np.zeros((state_grid.shape[0]+2, state_grid.shape[1]+2))
    # filll in the on-fire status
    on_fire_grid[1:-1, 1:-1][state_grid == const.FIRE] = 1
    
    # Now calculate the ignition probability of each cell 
    # for cell with no on-fire neighbours the probability is zero
    # (EAST + WEST + SOUTH + NORTH + SE + SW + NW + NE) / 8 (MOORE_NEIGHBOURS)
    igni_prob_grid = (on_fire_grid[1:-1, 2:] + on_fire_grid[1:-1, :-2] + on_fire_grid[2:, 1:-1] + on_fire_grid[:-2, 1:-1] + \
                     on_fire_grid[2:, 2:] + on_fire_grid[2:, :-2] + on_fire_grid[:-2, :-2] + on_fire_grid[:-2, 2:]) / 8
    
    # add in the vegetation and humidity effect
    igni_prob_grid[state_grid == const.TREE] *= const.P_TREE * const.H_TREE
    igni_prob_grid[state_grid == const.GRASS] *= const.P_GRASS * const.H_GRASS
    igni_prob_grid[state_grid == const.SHRUB] *= const.P_SHRUB * const.H_SHRUB
    
    # Now sample the new state grid
    # Generate a random array with the same shape as the ignition probability matrix
    random_array = np.random.random(igni_prob_grid.shape)
    # Create the new on-fire grid where values are 1 where random values are less than or equal to probabilities
    new_on_fire_grid = np.where(random_array <= igni_prob_grid, 1, 0)
    # replace the burned and empty cells
    new_on_fire_grid[state_grid == const.BURNED] = 0
    new_on_fire_grid[state_grid == const.EMPTY] = 0
    
    # Now create the new state grid
    state_grid[new_on_fire_grid == 1] = const.FIRE
    
    # Consider the burnup time
    burnup_t_grid[state_grid == const.FIRE] += 1
    state_grid[burnup_t_grid >= burnup_t] = const.BURNED
    
    return state_grid, burnup_t_grid


def experiment(density: float, dimension: int, veg_ratio: List[float], burnup_t: int) -> Union[float, List[List[float]]]:
    """Run experiments, each experiment contains independent simulation multiple times.
       Return the average burned area and percolation probability for each experiment

    Args:
        density (float): the plant density
        dimension (int): foestry grid dimension
        veg_ratio (List[float]): vegetation ratio, in the order: [tree], [grass], [shrub]
        burnup_t (int): burn up time

    Returns:
        float: average burned area for each experiment
        float: average percolation probability for each experiment
    """
    
    # set number of experiments and simulations
    n_experiments = 5
    n_simulations = 10
    
    # Run experiments
    
    print(f"\n------ DENSITY={density} ------\n")
    
    avg_perco_prob = []
    
    burned_areas = []
    
    print(const.P_GRASS)
    
    for n in range(n_experiments):
        percolation_count = 0
        
        for _ in range(n_simulations):
            
            # Initialize the grid firstly
            state_grid = make_grid(density, dimension, veg_ratio)
            burnup_t_grid = np.zeros((dimension,dimension))
            burnup_t_grid[state_grid == const.FIRE] += 1
            frames = []
            
            # Stops the simulation when there is no fire any more
            while const.FIRE in state_grid:
                frames.append(state_grid.copy())
                state_grid, burnup_t_grid = simulate(state_grid, burnup_t, burnup_t_grid)
            
            # for calculating burned area
            burned_area = np.count_nonzero(state_grid == const.BURNED) / (dimension**2)
            burned_areas.append(burned_area)
                
            # check percolation
            if const.BURNED in state_grid[-1][:] and \
               const.BURNED in state_grid[0][:] and  \
               const.BURNED in state_grid[:][0] and \
               const.BURNED in state_grid[:][-1]:
                
                percolation_count += 1
                
        # retrieve percolation probability over the n simulations
        percolation_chance = percolation_count / n_simulations
        
        avg_perco_prob.append(percolation_chance)
    
    # We are interested in the average percolation probability    
    avg_burned_area = np.array(burned_areas).mean()
    avg_perco_prob = np.array(avg_perco_prob).mean()
        
    print(f"\n------ DENSITY: {density} finished ------\n")
                
    return avg_burned_area, avg_perco_prob


def experiments(rep: int, density: float):
    """Run experiments with different ignition probability and vegetation ratio settings

    Args:
        rep (int): the sequence number of this experiment
        density (float): plant density for this experiment
    """
    
    # Heatmap experiments
    dimension = 50
    density = density
    setp_size = 0.1
    veg_ratios = np.arange(0.00, 1.00 + setp_size, setp_size)
    igni_probs = np.arange(0.00, 1.00 + setp_size, setp_size)
    burnup_t = 10
    
    
    perco_prob = []
    burned_area = []
    for veg_ratio in veg_ratios:
        avg_perco_probs = []
        avg_burned_areas = []
        for igni_prob in igni_probs:
            
            const.P_GRASS = igni_prob
    
            # serial version
            avg_burned_area, avg_perco_prob = experiment(density, dimension, [veg_ratio, 1-veg_ratio, 0], burnup_t)
            
            avg_perco_probs.append(avg_perco_prob)
            avg_burned_areas.append(avg_burned_area)
            
        perco_prob.append(avg_perco_probs)
        burned_area.append(avg_burned_areas)
            
    perco_prob_dict = dict(zip(np.round(veg_ratios, 2),perco_prob))
    avg_burn_dict = dict(zip(np.round(veg_ratios, 2),burned_area))
    
    print(perco_prob_dict, avg_burn_dict)
    
    filename_perco = f'perco_rep{rep}'
    filename_area = f'area_rep{rep}'
    
    filepath_perco = f"Output/Random_Heatmaps/perco_density_{density}"
    filepath_area = f"Output/Random_Heatmaps/area_density_{density}"
    
    if True:

        if not os.path.exists(filepath_perco):
            os.makedirs(filepath_perco)

        with open(filepath_perco + f'/{filename_perco}.json', 'w') as fp:
            json.dump(perco_prob_dict, fp)

        if not os.path.exists(filepath_area):
            os.makedirs(filepath_area)
            
        with open(filepath_area + f'/{filename_area}.json', 'w') as fp:
            json.dump(avg_burn_dict, fp)
            
    # For visualization
    filename = 'perco_prob'
    ignition_vs_ratio_heatmap(perco_prob_dict, veg_ratios, filename, savefig=False)


if __name__ == "__main__":
    
    rep = [1,2,3,4,5]
    
    pool = mp.Pool(5)
    
    import functools
    
    for density in [0.6]:
        
        pool.map(functools.partial(experiments, density=density), rep)
    
    pool.close()
    