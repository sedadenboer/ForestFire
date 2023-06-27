# experiments.py
#
# Course: Complex System Simulation, UvA
# Year: 2022/2023
# 
# Description: functions for running experiments
# 
# Dependencies: plot.py

import numpy as np
from forest import Forest
from typing import List, Dict
import json
from plot import density_lineplot, forest_decrease_lineplot, wind_factor_lineplot


def density_experiment(densities: np.ndarray, n_experiments: int, n_simulations: int,
                       grid_type: str, veg_ratio: List[float], dimension: int, burnup_time: int, neighbourhood_type: str,
                       visualize: bool, save_data: bool, make_plot: bool) -> Dict:
    """Runs n experiments for different values of p. For every experiment a simulation
    with the same settings is repeated m times, from which the percolation probability
    for that value of p is calculated. These probabilities are saved into a dictionary.

    Args:
        densities (List[float]): density values (p) to experiment with
        n_experiments (int): number of experiments for each p value
        n_simulations (int): number of simulations to repeat for each experiment
        grid_type (str): the layout of vegetation ('default' or 'mixed')
        veg_ratio (List[float]): the ratio between different vegetation type
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

            # of a predefined range of simulations
            for _ in range(n_simulations):
                model = Forest(
                    grid_type=grid_type,
                    veg_ratio=veg_ratio,
                    dimension=dimension,
                    density=p,
                    burnup_time=burnup_time,
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

            # save percolation probabilities per experiment in a dictionary
            if p in percolation_info:
                percolation_info[p].append(percolation_chance)
            else:
                percolation_info[p] = [percolation_chance]

    crit_density = get_critical_density(percolation_info)
    print()
    print(percolation_info)
    print('critical density:', crit_density)

    filename = f'density_nexp={n_experiments}_nsim={n_simulations}_grtype={grid_type}_d={dimension}_btime={burnup_time}_nbh={neighbourhood_type}_critd={crit_density}'
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

def forest_decrease_experiment(densities: np.ndarray, n_simulations: int,
                               grid_type: str, veg_ratio: List[float], dimension: int, burnup_time: int, neighbourhood_type: str,
                               visualize: bool, save_data: bool, make_plot: bool) -> Dict:
    """Runs n experiments for different values of p. For every experiment a simulation
    with the same settings is repeated m times, from which the burned area/initial plant ratio
    for that value of p is calculated. These probabilities are saved into a dictionary.

    Args:
        densities (List[float]): density values (p) to experiment with
        n_simulations (int): number of simulations to repeat for each experiment
        grid_type (str): the layout of vegetation ('default' or 'mixed')
        veg_ratio (List[float]): the ratio between different vegetation type
        dimension (int): dimension of the Forest grid
        burnup_time (int): burnup time for Trees
        neighbourhood_type (str): neighbourhood model ("moore" or "von_neumann")
        visualize (bool): whether to visualize the Forest model
        save_data (bool): whether to save the percolation data
        make_plot (bool): whether to make a lineplot

    Returns:
        Dict: Dictionary with forest decrease for different p values
    """
    decrease_info = {}

    # for each density value
    for p in densities:
        print(f"\n------ DENSITY={p} ------\n")

        # of a predefined range of simulations
        for _ in range(n_simulations):
            model = Forest(
                grid_type=grid_type,
                veg_ratio=veg_ratio,
                dimension=dimension,
                density=p,
                burnup_time=burnup_time,
                neighbourhood_type=neighbourhood_type,
                visualize=visualize
            )

            # run simulation
            model.simulate()

            # check if percolation occured
            decrease = model.forest_decrease()

            print(f"forest decrease: {decrease}")

            # save percolation probabilities per experiment in a dictionary
            if p in decrease_info:
                decrease_info[p].append(decrease)
            else:
                decrease_info[p] = [decrease]

    print(f"\nN simulations:{n_simulations}")
    print(decrease_info)

    filename = f'forestdecrease_nsim={n_simulations}_grtype={grid_type}_d={dimension}_btime={burnup_time}_nbh={neighbourhood_type}'
    if save_data:
        with open(f'Output/{filename}.json', 'w') as fp:
            json.dump(decrease_info, fp)
    if make_plot:
        forest_decrease_lineplot(decrease_info, filename, savefig=True)

    return decrease_info

def wind_factor_experiment(wind_factors: np.ndarray, n_experiments: int, n_simulations: int,
                           grid_type: str, veg_ratio: List[float], dimension: int, burnup_time: int, 
                           neighbourhood_type: str, wind_direction: str, visualize: bool, save_data: bool, 
                           make_plot: bool) -> Dict:
    """Runs n experiments for different values of wind factors. For every experiment a simulation
    with the same settings is repeated m times, from which the percolation probability
    for that value of wind factor is calculated. These probabilities are saved into a dictionary.

    Args:
        wind_factors (List[float]): wind factor values to experiment with
        n_experiments (int): number of experiments for each wind factor
        n_simulations (int): number of simulations to repeat for each experiment
        grid_type (str): the layout of vegetation ('default' or 'mixed')
        veg_ratio (List[float]): the ratio between different vegetation type
        dimension (int): dimension of the Forest grid
        burnup_time (int): burnup time for Trees
        neighbourhood_type (str): neighbourhood model ("moore" or "von_neumann")
        wind_direction (str): the direction of the wind
        visualize (bool): whether to visualize the Forest model
        save_data (bool): whether to save the percolation data
        make_plot (bool): whether to make a lineplot

    Returns:
        Dict: Dictionary with percolation probabilities for different wind factor values
    """
    percolation_info_wind = {}

    # for each wind factor value
    for w in wind_factors:
        print(f"\n------ WIND_FACTOR={w} ------\n")

        # perform n experiments
        for n in range(n_experiments):
            print("\nEXPERIMENT:", n, "\n")

            percolation_count = 0

            # of a predefined range of simulations
            for _ in range(n_simulations):
                model = Forest(
                    grid_type=grid_type,
                    veg_ratio=veg_ratio,
                    dimension=dimension,
                    density=0.65,  # Keep density constant for this experiment, adjust as necessary
                    burnup_time=burnup_time,
                    neighbourhood_type=neighbourhood_type,
                    wind_direction=wind_direction,
                    wind_factor=w,
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

            # save percolation probabilities per experiment in a dictionary
            if w in percolation_info_wind:
                percolation_info_wind[w].append(percolation_chance)
            else:
                percolation_info_wind[w] = [percolation_chance]

    filename = f'wind_factor_nexp={n_experiments}_nsim={n_simulations}_grtype={grid_type}_d={dimension}_btime={burnup_time}_nbh={neighbourhood_type}_wd={wind_direction}'
    if save_data:
        with open(f'Output/{filename}.json', 'w') as fp:
            json.dump(percolation_info_wind, fp)
    if make_plot:
        wind_factor_lineplot(percolation_info_wind, filename, savefig=True)

    return percolation_info_wind