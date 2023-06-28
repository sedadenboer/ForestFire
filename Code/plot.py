# plot.py
#
# Course: Complex System Simulation, UvA
# Year: 2022/2023
# 
# Description: Contains plotting functions that go  
# along the forest fire model.

import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def density_lineplot(data: Dict, savefig: bool) -> None:
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
        plt.savefig('Plots/density_lineplot_veg_b5.png', dpi=400)
    plt.show()
