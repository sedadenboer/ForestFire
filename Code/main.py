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
from experiments import density_experiment, forest_decrease_experiment
import constants


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Forest Fire Model')

    parser.add_argument('mode', nargs='?', choices=['test', 'crit_p', 'burn_area'],
                        help='Specify the mode to run (test, crit_p, burn_area)')
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
            visualize=True
        )
        
        start_time = time.time()
        model.simulate()
        print(f"Duration:{time.time() - start_time}s")
    # lineplot run for determining critical density
    elif args.mode == 'crit_p':
        step = 0.05
        results = density_experiment(
            densities=np.arange(0 + step, 1 + step, step),
            n_experiments=10,
            n_simulations=20,
            grid_type='default',
            vegetation_grid=grid,
            dimension=dimension,
            burnup_time=burnup_t,
            neighbourhood_type="moore",
            visualize=False,
            save_data=True,
            make_plot=True
        )
    # lineplot run for determining final forest area / initial forest area
    elif args.mode == 'burn_area':
        step = 0.05
        results = forest_decrease_experiment(
            densities=np.arange(0 + step, 1 + step, step),
            n_simulations=20,
            grid_type='default',
            vegetation_grid=grid,
            dimension=dimension,
            burnup_time=burnup_t,
            neighbourhood_type="moore",
            visualize=False,
            save_data=True,
            make_plot=True
        )