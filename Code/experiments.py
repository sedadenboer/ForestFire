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
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from plot import density_lineplot, burned_area_lineplot, ignition_vs_ratio_heatmap, beta_plot


def density_experiment(densities: np.ndarray, n_experiments: int, n_simulations: int,
                       grid_type: str, veg_ratio: List[float], dimension: int,
                       burnup_time: int, neighbourhood_type: str,
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
        Dict: dictionary with percolation probabilities for different p values
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

    # save data and make plot
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
        Dict: dictionary with forest decrease for different p values
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
                visualize=visualize,
            )

            # run simulation
            model.simulate()

            # check if percolation occured
            burned = model.burned_area()

            print(f"burned area: {burned}")

            # save percolation probabilities per experiment in a dictionary
            if p in decrease_info:
                decrease_info[p].append(burned)
            else:
                decrease_info[p] = [burned]

    # get critical density
    crit_density = get_critical_density(decrease_info)
    print(f"\nN simulations:{n_simulations}")
    print(decrease_info)
    print('critical density:', crit_density)

     # save data and make plot
    filename = f'burnedarea_nsim={n_simulations}_grtype={grid_type}_d={dimension}_btime={burnup_time}_nbh={neighbourhood_type}_critd={crit_density}'
    if save_data:
        with open(f'Output/{filename}.json', 'w') as fp:
            json.dump(decrease_info, fp)
    if make_plot:
        burned_area_lineplot(decrease_info, filename, savefig=True)

    return decrease_info

def ignition_vs_ratio_2(density: int, n_simulations: int,
                      grid_type: str, dimension: int, burnup_time: int,
                      fixed_ignition: float, varying_ignition: List[float], plant_ratios: List[List[float]],
                      visualize: bool, save_data: bool, make_plot: bool) -> Dict:
    """Percolation probability measured over n simulations, with varying plant ratios (tree/shrub) and varying
    ignition probabilities for shrubs, while keeping the ignition probability for trees fixed.

    Args:
        density (int): forest density
        n_simulations (int): number of simulations to repeat for each experiment
        grid_type (str): the layout of vegetation ('default' or 'mixed')
        dimension (int): dimension of the Forest grid
        burnup_time (int): burnup time for Trees
        fixed_ignition (float): _description_
        varying_ignition (List[float]): list of ratios between different vegetation type
        plant_ratios (List[List[float]]): _description_
        visualize (bool): whether to visualize the Forest model
        save_data (bool): whether to save the percolation data
        make_plot (bool): whether to make a heatmap

    Returns:
        Dict: dictionary with percolation probabilities for different plant ratios
              and ignition probabilities for shrubs
    """
    percolation_info = {}

    # for all plant ratios
    for ratio in plant_ratios:
        print('START, ratio:', ratio)
        # and varying ignition probability for shrubs
        for i in varying_ignition:
            print('ignition', i)
            percolation_count = 0
            # do n simulations and compute an average percolation 
            for _ in range(n_simulations):
                model = Forest(
                    grid_type=grid_type,
                    dimension=dimension,
                    density=density,
                    burnup_time=burnup_time,
                    veg_ratio=ratio,
                    visualize=visualize,
                    igni_ratio_exp=True,
                    adjust_igni_tree=fixed_ignition,
                    adjust_igni_shrub=i
                )

                model.simulate()

                # check if percolation occured
                if model.check_percolation():
                    percolation_count += 1

            # retrieve percolation probability over the n simulations
            percolation_chance = percolation_count / n_simulations

            print(f"percolation count: {percolation_count}")
            print("chance:", percolation_chance, '\n')

            # save percolation probabilities per experiment in a dictionary
            key = f'{round(ratio[0], 1)}/{round(ratio[2], 1)}'
            if key in percolation_info:
                percolation_info[key].append(percolation_chance)
            else:
                percolation_info[key] = [percolation_chance]

    # save data and make plot
    filename = f'heatmap_nsim={n_simulations}_grtype={grid_type}_dens={density}_dim={dimension}_btime={burnup_time}_fixigni={fixed_ignition}'
    if make_plot:
        ignition_vs_ratio_heatmap(percolation_info, varying_ignition, filename, savefig=True)
    if save_data:
        with open(f'Output/{filename}.json', 'w') as fp:
            json.dump(percolation_info, fp)

    return percolation_info

def beta_experiment(densities: np.ndarray, n_experiments: int, n_simulations: int,
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
    beta_info = {}
    step = 0.005
    # for each density value
    for p in densities:
        p = round(p, 6)

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
            if p in beta_info:
                beta_info[p].append(percolation_chance)
            else:
                beta_info[p] = [percolation_chance]

    crit_density = get_critical_density(beta_info)
    print()
    print(beta_info)
    print('critical density:', crit_density)

    for p in np.arange(0.5 + step, crit_density, step):
        p = round(p, 6)
        del beta_info[p]

    for p in np.arange(0.6, crit_density + 10 * step, -step):
        p = round(p, 6)
        del beta_info[p]    

    filename = f'beta_nexp={n_experiments}_nsim={n_simulations}_grtype={grid_type}_d={dimension}_btime={burnup_time}_nbh={neighbourhood_type}_critd={crit_density}'
    if save_data:
        with open(f'Output/{filename}.json', 'w') as fp:
            json.dump(beta_info, fp)
    if make_plot:
        beta_plot(beta_info, filename, crit_density, savefig=False)

    return beta_info

def get_fitness(data: Dict, crit_density: float) -> float:
    # create a DataFrame from the dictionary
    df = pd.DataFrame.from_dict(data)
    # reset the index and melt the DataFrame
    df = df.reset_index().melt(id_vars='index', var_name='p', value_name='Probability')
    # rename the columns
    df = df.rename(columns={'index': 'Experiment Number'})

    # linear fit of (p-p_c) as variable
    def power_law(x, a, b):
        return a * x ** b

    test_p = np.array(df['p'])
    pars, cov = curve_fit(f=power_law, xdata=np.array(df['p']) - crit_density, ydata=df['Probability'], bounds=(-np.inf, np.inf))
    perr = np.sqrt(np.diag(cov))

    print('a constant =', pars[0])
    print('beta =', pars[1])

    fit_data = []
    for element in test_p:
        fit_data.append(power_law(element - crit_density, *pars))

    r2 = r2_score(df['Probability'], fit_data)
    print('r2 score is', r2)

    return r2

def burn_experiment(densities: np.ndarray, n_experiments: int, n_simulations: int,
                    grid_type: str, veg_ratio: List[float], dimension: int, burnup_time: int, neighbourhood_type: str,
                    visualize: bool, save_data: bool) -> Dict:
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
    beta_info = {}
    step = 0.005
    variance = 0

    # for each density value
    for p in densities:
        p = round(p, 6)

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
            if p in beta_info:
                beta_info[p].append(percolation_chance)
            else:
                beta_info[p] = [percolation_chance]

    crit_density = get_critical_density(beta_info)
    print()
    print(beta_info)
    print('critical density:', crit_density)

    for p in np.arange(0.5 + step, crit_density, step):
        p = round(p, 6)
        del beta_info[p]

    for p in np.arange(0.6, crit_density + 10 * step, -step):
        p = round(p, 6)
        del beta_info[p]    

    for p in np.arange(crit_density, (len(beta_info) + 1) * step, step):
        a = np.std(beta_info[p])
        beta_info[p] = np.mean(beta_info[p])

    filename = f'burn_nexp={n_experiments}_nsim={n_simulations}_grtype={grid_type}_d={dimension}_btime={burnup_time}_nbh={neighbourhood_type}_critd={crit_density}'
    if save_data:
        with open(f'Output/{filename}.json', 'w') as fp:
            json.dump(beta_info, fp)

    goodness_of_fit = get_fitness(beta_info, crit_density)

    return goodness_of_fit, variance

def dimension_experiment(densities: np.ndarray, n_experiments: int, n_simulations: int, grid_type: str,
                         veg_ratio: List[float], dimension: int, burnup_time: int, neighbourhood_type: str,
                         visualize: bool) -> Dict:
    """Runs n experiments for different values of p. For every experiment a simulation
    with the same settings is repeated m times, from which the percolation probability
    for that value of p is calculated. for different dimensions These probabilities are saved into a dictionary.

    Args:
        densities (List[float]): density values (p) to experiment with
        n_experiments (int): number of experiments for each p value
        n_simulations (int): number of simulations to repeat for each experiment
        grid_type (str): the layout of vegetation ('default' or 'mixed')
        veg_ratio (List[float]): the ratio between different vegetation type
        dimension (list): dimension of the Forest grid
        burnup_time (int): burnup time for Trees
        neighbourhood_type (str): neighbourhood model ("moore" or "von_neumann")
        visualize (bool): whether to visualize the Forest model

    Returns:
        Dict: Dictionary with percolation probabilities for different p values
    """
    dimension_info = {}

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
            if p in dimension_info:
                dimension_info[p].append(percolation_chance)
            else:
                dimension_info[p] = [percolation_chance]

    crit_density = get_critical_density(dimension_info)
    print()
    print(dimension_info)
    print('critical density:', crit_density)

    return dimension_info