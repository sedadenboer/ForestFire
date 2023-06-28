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
from experiments import density_experiment, forest_decrease_experiment


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
    parser.add_argument('grid_type', nargs='?', choices=['default', 'stripe', 'block', 'random'],
                        help='Specify the mode to run (test, crit_p, burn_area)')
    # number of processors for simulation
    parser.add_argument('--np', type=int, required=False, help='fire burnup time')

    args = parser.parse_args()
    
    ################ Take inputs ################
    
    if args.dimension:
        dimension = args.dimension
    else:
        dimension = 50
    if args.density:
        density = args.density
    else:
        density = 0.65
    if args.burnup:
        burnup_t = args.burnup
    else:
        burnup_t = 1
    # take vegetation ratio from input
    if args.veg_ratio:
        veg_ratio = args.veg_ratio
    else:
        veg_ratio = []
    # parallel setting
    if args.np:
        n_proc = args.np
    else:
        n_proc = 1

    # if no argument provided
    if not args.mode:
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
        results = density_experiment(
            densities=np.arange(0 + step, 1 + step, step),
            n_experiments=10,
            n_simulations=20,
            veg_ratio=veg_ratio,
            grid_type=args.grid_type,
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
            veg_ratio=veg_ratio,
            grid_type=args.grid_type,
            dimension=dimension,
            burnup_time=burnup_t,
            neighbourhood_type="moore",
            visualize=False,
            save_data=True,
            make_plot=True
        )