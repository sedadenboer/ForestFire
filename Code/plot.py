# plot.py
#
# Course: Complex System Simulation, UvA
# Year: 2022/2023
# 
# Description: Contains plotting functions that go  
# along the forest fire model.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from IPython.display import display


def density_lineplot(data: Dict, filename: str, savefig: bool) -> None:
    """Makes a lineplot of the percolation probability for different
    values of the forest density, with 95% confidence interval for
    n experiments and m simulations of the forest fire model.

    Args:
        data (Dict): percolation data of experiment
        savefig (bool): True if figure should be saved, otherwise False
    """
    # create a DataFrame from the dictionary
    df = pd.DataFrame.from_dict(data)
    # reset the index and melt the DataFrame
    df = df.reset_index().melt(id_vars='index', var_name='p', value_name='Probability')
    # rename the columns
    df = df.rename(columns={'index': 'Experiment Number'})

    # plot the line plot
    plt.figure()
    sns.set_style("ticks")
    sns.lineplot(data=df, x='p', y='Probability',
                 color='seagreen', markers=True, dashes=False)
    plt.xlabel('p')
    plt.ylabel('Probability of percolation')
    if savefig:
        plt.savefig(f'Plots/{filename}.png', dpi=400)
    plt.show()

def burned_area_lineplot(data: Dict, filename: str, savefig: bool) -> None:
    """Makes a lineplot of the forest decrease ratio for different
    values of the forest density, with 95% confidence interval for
    n experiments and m simulations of the forest fire model.

    Args:
        data (Dict): forest decrease data of experiment
        filename (str): experiment specific filename
        savefig (bool): True if figure should be saved, otherwise False
    """
    # create a DataFrame from the dictionary
    df = pd.DataFrame.from_dict(data)
    # reset the index and melt the DataFrame
    df = df.reset_index().melt(id_vars='index', var_name='p', value_name='Forest')
    # rename the columns
    df = df.rename(columns={'index': 'Simulation Number'})

    display(df)

    # plot the line plot
    plt.figure()
    sns.set_style('ticks')
    sns.lineplot(data=df, x='p', y='Forest',
                 color='tomato', markers=True, dashes=False)
    plt.xlabel('p')
    plt.ylabel('Final burned area percentage')
    if savefig:
        plt.savefig(f'Plots/burnedarea_{filename}.png', dpi=400)
    plt.show()

def ignition_vs_ratio_heatmap(data: Dict, ignition_list: List[float], filename: str, savefig: bool) -> None:
    """Makes a heatmap of the results of experimenting with the percolation probability
    over n simulations with varying plant ratios (tree/shrub) and varying ignition
    probabilities for shrubs, while keeping the ignition probability for trees fixed.

    Args:
        data (Dict): percolation data of experiment
        ignition_list (List[float]): range of ignition
        filename (str): experiment specific filename
        savefig (bool): True if figure should be saved, otherwise False
    """
    # create DataFrame from dictionary
    df = pd.DataFrame.from_dict(data)
    # use ignition probability of shrubs as index
    df.index = [round(num, 2) for num in ignition_list]

    display(df)

    # plot the heatmap
    plt.figure()
    sns.set_style('ticks')
    ax = sns.heatmap(df, annot=True, cbar_kws={'label': 'Percolation probability'}, cmap=sns.color_palette("rocket_r", as_cmap=True))
    ax.invert_yaxis()
    ax.set_xlabel('Vegetation ratio (tree/shrub)')
    ax.set_ylabel('Ignition probability shrub')
    if savefig:
        plt.savefig(f'Plots/heatmap_{filename}.png', dpi=400, bbox_inches='tight')
    plt.show()

def beta_plot(data: Dict, filename: str, crit_density: float, savefig: bool) -> None:
    """Makes a scatter plot for determining the critical exponent beta.
    Values are of the forest density, with 95% confidence interval for
    n experiments and m simulations of the forest fire model.

    Args:
        data (Dict): percolation data of experiment
        savefig (bool): True if figure should be saved, otherwise False
    """
    # create a DataFrame from the dictionary
    df = pd.DataFrame.from_dict(data)
    # reset the index and melt the DataFrame
    df = df.reset_index().melt(id_vars='index', var_name='p', value_name='Probability')
    # rename the columns
    df = df.rename(columns={'index': 'Experiment Number'})
    # linear fit of (p-p_c) as variable
    # function to calculate the power-law with constants a and b
    def power_law(x, a, b):
        return a * x ** b

    # print(df)
    test_p = np.array(df['p'])
    print(test_p)
    print(df['Probability'])

    pars, cov = curve_fit(f=power_law, xdata=np.array(df['p']) - crit_density, ydata=df['Probability'], bounds=(-np.inf, np.inf))
    perr = np.sqrt(np.diag(cov))
    print(perr)

    print('a constant =', pars[0])
    beta = "{:.2f}".format(pars[1])
    print('beta =', beta)

    fit_data = []
    for element in test_p:
        fit_data.append(power_law(element - crit_density, *pars))

    r2 = r2_score(df['Probability'], fit_data)
    print('r2 score is', r2)

    # plot the scatter plot
    plt.figure()
    plt.xlabel('p')
    plt.ylabel('Probability of percolation')
    plt.text(0.01, 0.2, f'\u03B2 ={beta}', fontsize=20)
    plt.plot(test_p - crit_density, fit_data, 'r', label='fit')
    plt.errorbar(test_p - crit_density, fit_data, yerr=perr[1], color='brown')
    plt.scatter(test_p - crit_density, df['Probability'], color='k', label='data')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

    if savefig:
        plt.savefig(f'Plots/beta_{filename}.png', dpi=400)

    plt.show()

    return pars[1], r2

def burn_plot(burn_times: list, gof: list, savefig: bool) -> None:
    """Makes a scatter plot for determining the critical exponent beta for different burn times.
    Values are of the forest density, with 95% confidence interval for
    n experiments and m simulations of the forest fire model.

    Args:
        data (Dict): percolation data of experiment
        savefig (bool): True if figure should be saved, otherwise False
    """
    plt.scatter(burn_times, gof)
    plt.xlabel('burnup time')
    plt.ylabel('goodnes of fit value')
    if savefig:
        plt.savefig('Plots/burn_plot.png', dpi=400)

    plt.show()

def dimension_plot(data, savefig: bool) -> None:
    """Makes a lineplot of the percolation probability for different
    values of the forest density for different grid size, with 95% confidence interval for
    n experiments and m simulations of the forest fire model.

    Args:
        data (Dict): percolation data of experiment
        savefig (bool): True if figure should be saved, otherwise False
    """
    dimensions = list(data.keys())
    plt.figure()
    for d in dimensions:
        # create a DataFrame from the dictionary
        df = pd.DataFrame.from_dict(data[d][0])
        # reset the index and melt the DataFrame
        df = df.reset_index().melt(id_vars='index', var_name='p', value_name='Probability')
        # rename the columns
        df = df.rename(columns={'index': 'Experiment Number'})

        # plot the line plot
        sns.set_style("ticks")
        sns.lineplot(data=df, x='p', y='Probability', markers=True, dashes=False, label=f'{d}')
        plt.xlabel('p')
        plt.ylabel('Probability of percolation')

    if savefig:
        plt.savefig(f'Plots/dimensions.png', dpi=400)

    plt.legend()
    plt.show()