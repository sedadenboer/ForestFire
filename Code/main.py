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
from experiments import density_experiment, forest_decrease_experiment, beta_experiment, dimension_experiment
from plot import dimension_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Forest Fire Model')

    parser.add_argument('mode', nargs='?', choices=['test', 'crit_p', 'burn_area', 'beta', 'dimension'],
                        help='Specify the mode to run (test, crit_p, burn_area, beta, dimension)')
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
        step = 0.1
        results = density_experiment(
            densities=np.arange(0 + step, 1 + step, step),
            n_experiments=5,
            n_simulations=5,
            veg_ratio=veg_ratio,
            grid_type=args.grid_type,
            dimension=dimension,
            burnup_time=burnup_t,
            neighbourhood_type="moore",
            visualize=False,
            save_data=False,
            make_plot=True
        )
    # lineplot run for determining final forest area / initial forest area
    elif args.mode == 'burn_area':
        step = 0.05
        results = forest_decrease_experiment(
            densities=np.arange(0.8 + step, 1 + step, step),
            n_experiments=5,
            n_simulations=5,
            veg_ratio=veg_ratio,
            grid_type=args.grid_type,
            dimension=dimension,
            burnup_time=burnup_t,
            neighbourhood_type="moore",
            visualize=False,
            save_data=False,
            make_plot=True
        )
    # scatter plot for determining the critical exponent beta
    elif args.mode == 'beta':
        step = 0.05 # 0.005 maybe
        results = beta_experiment(
            densities=np.arange(0.6, 0.9 + step, step),
            n_experiments=2,
            n_simulations=2,
            veg_ratio=veg_ratio,
            grid_type=args.grid_type,
            dimension=dimension,
            burnup_time=burnup_t,
            neighbourhood_type="moore",
            visualize=False,
            save_data=False,
            make_plot=True
        )
    # line plots for different grid sizes
    elif args.mode == 'dimension':
        step = 0.05 # 0.005 maybe
        dimension=[10, 25, 100]
        multiple_results = {}
        for d in dimension:
            results = dimension_experiment(
                densities=np.arange(0.5, 0.9 + step, step),
                n_experiments=2,
                n_simulations=2,
                veg_ratio=veg_ratio,
                grid_type=args.grid_type,
                dimension=d,
                burnup_time=burnup_t,
                neighbourhood_type="moore",
                visualize=False,
                save_data=False,
                make_plot=False
            )
            # save percolation probabilities per experiment in a dictionary
            if d in multiple_results:
                multiple_results[d].append(results)
            else:
                multiple_results[d] = [results]

        dimension_plot(multiple_results, savefig=False)

    # elif args.mode == 'pattern':
    #     step = 0.01 # 0.005 maybe
    #     results = pattern_experiment(
    #         densities=np.arange(0 + step, 1 + step, step),
    #         n_experiments=5,
    #         n_simulations=5,
    #         veg_ratio=veg_ratio,
    #         grid_type=args.grid_type,
    #         dimension=dimension,
    #         burnup_time=burnup_t,
    #         neighbourhood_type="moore",
    #         visualize=False,
    #         save_data=False,
    #         make_plot=True
    #     )