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