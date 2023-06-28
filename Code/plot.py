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
from typing import Dict
from scipy.optimize import curve_fit
import scipy.stats as stats


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

def forest_decrease_lineplot(data: Dict, filename: str, savefig: bool) -> None:
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

    # plot the line plot
    plt.figure()
    sns.set_style("ticks")
    sns.lineplot(data=df, x='p', y='Forest',
                 color='tomato', markers=True, dashes=False)
    plt.xlabel('p')
    plt.ylabel('Final forest density / initial forest density')
    if savefig:
        plt.savefig(f'Plots/forestdecrease_{filename}.png', dpi=400)
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
    # Function to calculate the power-law with constants a and b
    def power_law(x, a, b):
        return a * np.power(x, b)

    pars, cov = curve_fit(f=power_law, xdata=df['p'] - crit_density + 0.001, ydata=df['Probability'], bounds=(-np.inf, np.inf))

    # plot the scatter plot
    plt.figure()
    #plt.scatter(data=df, x='p', y='Probability')
    #plt.plot(df['p'] - 0.54, power_law(df['p'] - 0.54, *pars))
    plt.xlabel('p')
    plt.ylabel('Probability of percolation')
    if savefig:
        plt.savefig(f'Plots/beta_{filename}.png', dpi=400)
    plt.text(0.5, 0.5, f'\u03B2 = {pars[1]}')
    plt.plot(df['p'] - crit_density + 0.001, power_law(df['p'] - crit_density + 0.001, *pars), 'r', label='data')
    plt.scatter(df['p'] - crit_density + 0.001, df['Probability'], color='k', label='fit')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
    plt.savefig(f'dimension.png', dpi=400)

    print('beta =', pars[1])

    # Chi-Square Goodness of Fit Test
    chi_square_test_statistic, p_value = stats.chisquare(
        df['Probability'], power_law(df['p'] - crit_density + 0.001))
    
    # chi square test statistic and p value
    print('chi_square_test_statistic is : ' +
        str(chi_square_test_statistic))
    print('p_value : ' + str(p_value))


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
        # if savefig:
        #     plt.savefig(f'Plots/dimensions_{filename}.png', dpi=400)
    plt.legend()
    plt.show()