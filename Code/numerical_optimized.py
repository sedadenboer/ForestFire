# numerical_optimized.py
#
# Course: Complex System Simulation, UvA
# Year: 2022/2023
#
# Description: The original codes are slow for large scale experiments,
# this script optimizes the process with numpy vectorization features
#
# Dependencies: vegetation.py, constants.py

import numpy as np
import constants as const
from typing import List, Tuple
from visualize import visualize
import multiprocessing as mp
import json

# burn_up grid, used to record the burn up time of each cell
burnup_t_grid = np.zeros((100,100))

# make random vegetation grid
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
    
    # start fire
    top_row = veg_grid[0][:]
    top_row[top_row != const.EMPTY] = const.FIRE
    veg_grid[0][:] = top_row
    
    # # or start fire randomly
    # while True:
    #     random_row = np.random.randint(0, veg_grid.shape[0])
    #     random_column = np.random.randint(0, veg_grid.shape[1])
        
    #     if (veg_grid[random_row, random_column] != const.EMPTY):
    #         break
    # veg_grid[random_row, random_column] = const.FIRE
    
    return veg_grid
    
    
def simulate(state_grid, burnup_t, burnup_t_grid):
    
    # firstly, take the on_fire grid, which has 0 for non-fire cell and 1 for on-fire cell
    # NOTE: extra layers for boundary cells, and the boundary cells always have less ignition probability
    on_fire_grid = np.zeros((state_grid.shape[0]+2, state_grid.shape[1]+2))
    # filll in the on-fire status
    on_fire_grid[1:-1, 1:-1][state_grid == const.FIRE] = 1
    
    # Now calculate the ignition probability of each cell 
    # for cell with no on-fire neighbours the probability is zero
    # (EAST + WEST + SOUTH + NORTH + SE + SW + NW + NE) / 8 (VON_NEUMANN_NEIGHBOURS)
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


def experiment(density: float, dimension: int, veg_ratio: List[float], burnup_t: int) -> List[List[int]]:
    
    # set number of experiments and simulations
    n_experiments = 5
    n_simulations = 10
    
    # Run experiments
    
    print(f"\n------ DENSITY={density} ------\n")
    
    results = []
    
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
            if const.BURNED in state_grid[-1][:]:
                percolation_count += 1
                
            # retrieve percolation probability over the n simulations
        percolation_chance = percolation_count / n_simulations
        # print(f"percolation count: {percolation_count}, n simulations: {n_simulations}")
        # print("chance:", percolation_chance)
        
        
        results.append(percolation_chance)
        
    avg_burned_area = np.array(burned_areas).mean()
        
    print(f"\n------ DENSITY: {density} finished ------\n")
                
    return avg_burned_area, results


def experiments(rep, density):
    
    # # Veg line plot experiment
    # step = 0.01
    # densities = np.arange(0.01, 1 + step, step)
    # dimension = 100
    # veg_ratios = np.arange(0.00, 1.00+0.01, 0.01)
    # burnup_ts = [1,2,3,4,5,6,7,8,9,10]
    
    # pool = mp.Pool(8)
    # import functools
    
    # for veg_ratio in veg_ratios:
    #     for burnup_t in burnup_ts:
    
    #         # run experiments
            
    #         # # serial version
    #         # results = []
    #         # for density in densities:
    #         #     frames, result = experiment(density, dimension, [veg_ratio, 1-veg_ratio, 0], burnup_t)
    #         #     results.append(result)
            
    #         # parallel version
    #         results = pool.map(functools.partial(experiment,
    #                                              dimension=dimension, 
    #                                              veg_ratio=[veg_ratio, 1-veg_ratio, 0], 
    #                                              burnup_t=burnup_t), densities)
                                
    #         # frames = frames[-1]
            
    #         results_dict = dict(zip(densities,results))
                
    #         print(results_dict)
            
    #         if True:
    #             with open(f'Output/size100_1/cript_p_b{burnup_t}_r_{veg_ratio}_{round(1- veg_ratio, 1)}.json', 'w') as fp:
    #                 json.dump(results_dict, fp)
                    
    
    # Heatmap experiments
    dimension = 100
    density = density
    setp_size = 0.05
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
            avg_burned_area, results = experiment(density, dimension, [veg_ratio, 1-veg_ratio, 0], burnup_t)
            
            # We are interested in the average percolation probability
            avg_percolation_prob = np.mean(np.array(results))
            
            avg_perco_probs.append(avg_percolation_prob)
            avg_burned_areas.append(avg_burned_area)
                                
            # frames = frames[-1]
            
        perco_prob.append(avg_perco_probs)
        burned_area.append(avg_burned_areas)
            
    perco_prob_dict = dict(zip(np.round(veg_ratios, 2),perco_prob))
    avg_burn_dict = dict(zip(np.round(veg_ratios, 2),burned_area))
    
    print(perco_prob_dict, avg_burn_dict)
    
    filename_perco = f'perco_rep{rep}'
    filename_area = f'area_rep{rep}'
    
    if True:
        with open(f'Output/Heatmaps/perco_density_{density}/{filename_perco}.json', 'w') as fp:
            json.dump(perco_prob_dict, fp)
            
        with open(f'Output/Heatmaps/area_density_{density}/{filename_area}.json', 'w') as fp:
            json.dump(avg_burn_dict, fp)
            
    
    # visualize(
    #             frames, showplot=True, saveplot=True,
    #             filename='simulation_animation'
    #         )


if __name__ == "__main__":
    
    rep = [1,2,3,4,5]
    
    pool = mp.Pool(5)
    
    import functools
    
    for density in [0.6, 0.7, 0.8, 0.9]:
        
        pool.map(functools.partial(experiments, density=density), rep)
    
    pool.close()
    