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
from experiments import density_experiment, forest_decrease_experiment, wind_factor_experiment


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
    
    # don't know if I added it correctly.
    # parser.add_argument('--wind_direction',type =str,required=False, nargs=4, action='store',
    #                     help='The direction of the wind in the model, format : [N],[E],[S],[W]')
    # vegetation grid type input
    parser.add_argument('grid_type', nargs='?', choices=['default', 'stripe', 'block', 'random'],
                        help='Specify the mode to run (test, crit_p, burn_area)')
    
    # number of processors for simulation
    parser.add_argument('--np', type=int, required=False, help='fire burnup time')
    parser.add_argument('--use_wind', action='store_true', 
                    help='Use wind in the model')

    parser.add_argument('--wind_direction', type=str, choices=['N', 'E', 'S', 'W'], 
                        help='The direction of the wind in the model')

    parser.add_argument('--wind_factor', type=float,
                        help='The factor of the wind in the model')

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

     # default test run
    if args.use_wind:

        if args.wind_direction and args.wind_factor:
            # Use wind in the model with provided direction and factor
            wind_direction = args.wind_direction
            wind_factor = args.wind_factor
        else:
            print("If '--use_wind' is specified, both '--wind_direction' and '--wind_factor' must be provided.")
            exit(1)
    else:
        wind_direction = None
        wind_factor = None

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
            visualize=True,
            wind_direction=wind_direction,
            wind_factor=wind_factor,
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
            make_plot=True,
            wind_direction=wind_direction,
            wind_factor=wind_factor,
        )

    # lineplot run for determining final forest area / initial forest area
    elif args.mode == 'burn_area':
        step = 0.05
        results = forest_decrease_experiment(
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
            make_plot=True,
            wind_direction=wind_direction,
            wind_factor=wind_factor,
        )

        # lineplot run for determining percolation probability with different wind factors
    elif args.mode == 'wind_factor':
        wind_factors = np.arange(0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)  # example: wind factors from 0.1 to 1.0 with step 0.1
        results = wind_factor_experiment(
            wind_factors=wind_factors,
            n_experiments=10,
            n_simulations=20,
            veg_ratio=veg_ratio,
            grid_type=args.grid_type,
            dimension=dimension,
            burnup_time=burnup_t,
            neighbourhood_type="moore",
            visualize=False,
            save_data=True,
            make_plot=True,
            wind_direction=wind_direction,
            # note that wind_factor is not included here, it's taken care of by wind_factors
        )