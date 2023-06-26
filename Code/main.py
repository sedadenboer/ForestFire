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
from forest import Forest
from typing import List, Dict
import json
from plot import density_lineplot

import multiprocessing as mp

def experiment(density: float, n_experiments: int, n_simulations: int, veg_ratio: List[float],
               grid_type: str, dimension: int, burnup_time: int, neighbourhood_type: str,
               visualize: bool) -> None:
    """Runs n experiments for different values of p. For every experiment a simulation
    with the same settings is repeated m times, from which the percolation proability
    for that value of p is calculated. These probabilities are saved into a dictionary.

    Args:
        density (float): density value (p) to experiment with
        n_experiments (int): number of experiments for each p value
        n_simulations (int): number of simulations to repeat for each experiment
        veg_ratio (List[float]): the ratio between different vegetation type
        grid_type (str): vegetation layout type ('default' or 'stripe' / 'vertical' / 'random')
        dimension (int): dimension of the Forest grid
        burnup_time (int): burnup time for Trees
        neighbourhood_type (str): neighbourhood model ("moore" or "von_neumann")
        visualize (bool): whether to visualize the Forest model.

    Returns:
        Dict: Dictionary with percolation probabilities for different p values
    """

    # for each density value

    print(f"\n------ DENSITY={density} ------\n")
    results = []

    # perform n experiments
    for n in range(n_experiments):
        print("\nEXPERIMENT:", n, "\n")
        
        percolation_count = 0

        # of a predefined range of simualtions
        for _ in range(n_simulations):
            
            model = Forest(
                grid_type=grid_type,
                dimension=dimension,
                density=density,
                burnup_time=burnup_time,
                veg_ratio=veg_ratio,
                neighbourhood_type=neighbourhood_type,
                visualize=visualize
            )

            # run simulation
            model.simulate()

            # check if percolation occured
            if model.check_percolation():
                percolation_count += 1
                
        # retrieve percolation probability over the n simulations
        percolation_chance = percolation_count / n_simulations
        print(f"percolation count: {percolation_count}, n simulations: {n_simulations}")
        print("chance:", percolation_chance)
        
        results.append(percolation_chance)
    
    return results


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
    parser.add_argument('grid_type', nargs='?', choices=['default', 'stripe', 'vertical', 'random'],
                        help='Specify the mode to run (test, crit_p)')
    # number of processors for simulation
    parser.add_argument('--np', type=int, required=False, help='fire burnup time')

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
    # parallel setting
    if args.np is not None:
        n_proc = args.np
    else:
        n_proc = 1
        
    

    # if no argument provided
    if args.mode is None:
        parser.print_help()
    # default test run
    elif args.mode == 'test':
        model = Forest(
            grid_type=args.grid_type,
            dimension=dimension,
            density=density,
            burnup_time=burnup_t,
            veg_ratio=veg_ratio,
            neighbourhood_type="moore",
            visualize=True
        )
        
        model.simulate()
    # lineplot run for determining critical density
    elif args.mode == 'crit_p':
        step = 0.05
        densities = np.arange(0, 1 + step, step)
        
        # # serial
        # results = []
        # for density in densities:
        #     result = experiment(
        #             density=density,
        #             n_experiments=5,
        #             n_simulations=10,
        #             veg_ratio=veg_ratio,
        #             grid_type=args.grid_type,
        #             dimension=dimension,
        #             burnup_time=burnup_t,
        #             neighbourhood_type="moore",
        #             visualize=False
        #         )
            
        #     results.append(result)
            
        # parallel
        pool = mp.Pool(n_proc)
        import functools
        results = pool.map(functools.partial(experiment,
                                n_experiments=5,
                                n_simulations=10,
                                veg_ratio=veg_ratio,
                                grid_type=args.grid_type,
                                dimension=dimension,
                                burnup_time=burnup_t,
                                neighbourhood_type="moore",
                                visualize=False), densities)
        
        pool.close()
        
        keys = densities
        results_dict = dict(zip(densities,results))
        
        print(results_dict)
        
        if True:
            with open('Output/percolation_data_veg_bt.json', 'w') as fp:
                json.dump(results_dict, fp)
            
        # Make a density lineplot
        density_lineplot(results_dict, savefig=True)