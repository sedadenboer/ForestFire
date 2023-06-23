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
from plot import density_lineplot, multiple_plots, scatter_plot


def experiment(densities: List[float], n_experiments: int, n_simulations: int,
               default: bool, dimension: int, burnup_time: int, neighbourhood_type: str,
               visualize: bool, save_data: bool) -> Dict:
    """Runs n experiments for different values of p. For every experiment a simulation
    with the same settings is repeated m times, from which the percolation proability
    for that value of p is calculated. These probabilities are saved into a dictionary.

    Args:
        densities (List[float]): density values (p) to experiment with
        n_experiments (int): number of experiments for each p value
        n_simulations (int): number of simulations to repeat for each experiment
        default (bool): default spreading setting
        dimension (int): dimension of the Forest grid
        burnup_time (int): burnup time for Trees
        neighbourhood_type (str): neighbourhood model ("moore" or "von_neumann")
        visualize (bool): whether to visualize the Forest model.
        save_data (bool): whether to save the percolation data.

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

            # of a predefined range of simualtions
            for _ in range(n_simulations):
                model = Forest(
                    default=default,
                    dimension=dimension,
                    density=p,
                    burnup_time=burnup_time,
                    neighbourhood_type=neighbourhood_type,
                    visualize=visualize
                )

                # run simulation
                model.simulate()

                # check if percolation occured and for how many fore pixels
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

    print(percolation_info)
    print()

    # if save_data:
    #     with open('Output/percolation_data.json', 'w') as fp:
    #         json.dump(percolation_info, fp)

    return percolation_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Forest Fire Model')

    parser.add_argument('mode', nargs='?', choices=['test', 'crit_p', 'dimensions', 'beta'],
                        help='Specify the mode to run (test, crit_p, dimensions, beta)')

    args = parser.parse_args()

    # if no argument provided
    if args.mode is None:
        parser.print_help()
    # default test run
    elif args.mode == 'test':
        model = Forest(
            default=True,
            dimension=100,
            density=0.65,
            burnup_time=1,
            neighbourhood_type="von_neumann",
            visualize=True
        )
        
        model.simulate()
        print(f'The forest area decreased by {model.forest_decrease()} %')
    # lineplot run for determining critical density
    elif args.mode == 'crit_p':
        step = 0.01
        results = experiment(
            densities=np.arange(0, 1 + step, step),
            n_experiments=5,
            n_simulations=10,
            default=True,
            dimension=50,
            burnup_time=1,
            neighbourhood_type="von_neumann",
            visualize=False,
            save_data=True
        )

        # Make a density lineplot
        density_lineplot(results, savefig=False)
        

    elif args.mode == 'dimensions':
        multiple_results = {}
        step = 0.05
        dimensions = [5, 25, 50, 100]
        for d in dimensions:
            results = experiment(
                densities=np.arange(0.25, 0.75 + step, step),
                n_experiments=5,
                n_simulations=10,
                default=True,
                dimension=d,
                burnup_time=1,
                neighbourhood_type="von_neumann",
                visualize=False,
                save_data=True
        )      
            # save percolation probabilities per experiment in a dictionary
            if d in multiple_results:
                multiple_results[d].append(results)
            else:
                multiple_results[d] = [results]

        multiple_plots(multiple_results, savefig=False)

    # lineplot run for determining critical density
    elif args.mode == 'beta':
        step = 0.005 # 0.005 maybe
        results = experiment(
            densities=np.arange(0.58, 0.62 + step, step),
            n_experiments=5,
            n_simulations=10,
            default=True,
            dimension=25, # 50 at least
            burnup_time=1,
            neighbourhood_type="von_neumann",
            visualize=False,
            save_data=True
        )

        scatter_plot(results, savefig=False)