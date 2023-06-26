# main.py
#
# Course: Complex System Simulation, UvA
# Year: 2022/2023
# 
# Description: main code to run the forest fire model.
# 
# Dependencies: forest.py

import argparse
import numpy as np
import time
from forest import Forest
from typing import List, Dict
import json
from plot import density_lineplot
import constants


def experiment(densities: List[float], n_experiments: int, n_simulations: int,
               grid_type: str, dimension: int, burnup_time: int, neighbourhood_type: str,
               visualize: bool, save_data: bool, make_plot: bool) -> Dict:
    """Runs n experiments for different values of p. For every experiment a simulation
    with the same settings is repeated m times, from which the percolation proability
    for that value of p is calculated. These probabilities are saved into a dictionary.

    Args:
        densities (List[float]): density values (p) to experiment with
        n_experiments (int): number of experiments for each p value
        n_simulations (int): number of simulations to repeat for each experiment
        grid_type (str): the layout of vegetation ('default' or 'mixed')
        dimension (int): dimension of the Forest grid
        burnup_time (int): burnup time for Trees
        neighbourhood_type (str): neighbourhood model ("moore" or "von_neumann")
        visualize (bool): whether to visualize the Forest model
        save_data (bool): whether to save the percolation data
        make_plot (bool): whether to make a lineplot

    Returns:
        Dict: Dictionary with percolation probabilities for different p values
    """
    percolation_info = {}

    # for each density value
    for p in densities:
        print(f"\n------ DENSITY={p} ------\n")

        # perform n experiments
        for n in range(n_experiments):
            print("\nEXPERIMENT:", n, "\n")

            percolation_count = 0
            model = Forest(
                    grid_type=grid_type,
                    vegetation_grid=grid,
                    dimension=dimension,
                    density=p,
                    burnup_time=burnup_time,
                    neighbourhood_type=neighbourhood_type,
                    visualize=visualize
                )
            
            # of a predefined range of simulations
            for _ in range(n_simulations):
            
                # run simulation
                model.simulate()

                # check if percolation occured
                if model.check_percolation():
                    percolation_count += 1

            # retrieve percolation probability over the n simulations
            percolation_chance = percolation_count / n_simulations
            print(f"percolation count: {percolation_count}, n simulations: {n_simulations}")
            print("chance:", percolation_chance)

            # save percolation probabilities per experiment in a dictionary
            if p in percolation_info:
                percolation_info[p].append(percolation_chance)
            else:
                percolation_info[p] = [percolation_chance]

    crit_density = get_critical_density(percolation_info)
    print()
    print(percolation_info)
    print('critical density:', crit_density)

    filename = f'density_nexp={n_experiments}_nsim={n_simulations}_grtype={grid_type}_d={dimension}_btime={burnup_t}_nbh={neighbourhood_type}_critd={crit_density}'
    if save_data:
        with open(f'Output/{filename}.json', 'w') as fp:
            json.dump(percolation_info, fp)
    if make_plot:
        density_lineplot(percolation_info, filename, savefig=True)

    return percolation_info

def get_critical_density(data: Dict) -> float:
    """Calculates the average of lists in a dictionary and returns the key for which the average first surpasses 0.5.

    Args:
        dictionary (Dict): Dictionary with lists of values

    Returns:
        float: density for which the average first surpasses 0.5, or None if no such key exists.
    """
    for key, values in data.items():
        average = np.mean(values)
        if average > 0.5:
            return key
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Forest Fire Model')

    parser.add_argument('mode', nargs='?', choices=['test', 'crit_p'],
                        help='Specify the mode to run (test, crit_p)')
    # grid dimension input
    parser.add_argument('--dimension', type=int, required=False, help='dimension of the grid')
    # grid dimension input
    parser.add_argument('--density', type=float, required=False, help='density of the plant')
    # fire burnup time input
    parser.add_argument('--burnup', type=int, required=False, help='fire burnup time')
    # vegetation ratio input
    parser.add_argument('--veg_ratio', type=float, required=False, nargs=3, action='store',
                        help='The ratio of vegetation in the grid, format: [tree] [grass] [shrub]')
    # vegetation grid type input
    parser.add_argument('grid_type', nargs='?', choices=['stripe', 'block', 'random'],
                        help='Specify the mode to run (test, crit_p)')

    args = parser.parse_args()
    
    ################ Take inputs ################
    
    if args.dimension is not None:
        dimension = args.dimension
    else:
        dimension = 50
    if args.density is not None:
        density = args.density
    else:
        density = 0.65
    if args.burnup is not None:
        burnup_t = args.burnup
    else:
        burnup_t = 1
    # take vegetation ratio from input
    if args.veg_ratio is not None:
        veg_ratio = args.veg_ratio
    else:
        veg_ratio = []
        
    # compute the vegetaion layout grid as a 2d matrix
    plant_type = [constants.EMPTY,constants.TREE,constants.GRASS,constants.SHRUB]
    # initialize the grid
    grid = np.zeros((dimension,dimension))
    try:
        if args.grid_type == 'stripe':
            lengths = np.round(np.array(veg_ratio) * len(grid[:,0])).astype(int)
            splits = np.split(np.arange(dimension), np.cumsum(lengths)[:-1])
            
            # fill in the grid
            for i, split in enumerate(splits):
                grid[split] = np.random.choice([constants.EMPTY,plant_type[i+1]], 
                                                    size=(len(split), dimension), p=[1-density,density])
                
        elif args.grid_type == 'block':
            ...
        elif args.grid_type == 'random':
            p_list = np.concatenate([np.array([1-density]), np.array(veg_ratio)*density])
            grid = np.random.choice(plant_type, size=(dimension, dimension), p=p_list)
    except:
        print("Must specify vegetation ratio if you want to enable vegetation grid!!!")
        

    # if no argument provided
    if args.mode is None:
        parser.print_help()
    # default test run
    elif args.mode == 'test':
        model = Forest(
            grid_type='default',
            vegetation_grid=grid,
            dimension=dimension,
            density=density,
            burnup_time=burnup_t,
            neighbourhood_type="moore",
            visualize=False
        )
        
        start_time = time.time()
        model.simulate()
        print(f"Duration:{time.time() - start_time} s")

    # lineplot run for determining critical density
    elif args.mode == 'crit_p':
        step = 0.05
        results = experiment(
            densities=np.arange(0 + step, 1 + step, step),
            n_experiments=10,
            n_simulations=30,
            grid_type='default',
            dimension=dimension,
            burnup_time=burnup_t,
            neighbourhood_type="moore",
            visualize=False,
            save_data=True,
            make_plot=True
        )