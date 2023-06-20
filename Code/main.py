# main.py
#
# Course: Complex System Simulation, UvA
# Year: 2022/2023
# 
# Description: main code to run the forest fire model.
# 
# Dependencies: forest.py

from forest import Forest

if __name__ == "__main__":
    # initialize forest fire model
    model = Forest(
        dimension=100,
        density=0.8,
        burnup_chance=0.1,
        visualize=True
        )

    # run model
    model.simulate()