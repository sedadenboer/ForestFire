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
        density=0.75,
        burnup_time=6,
        ignition_chance=0.0001,
        wind_factor=0.1,  # this is just an example value; adjust according to your needs
        wind_direction='N',  # adjust this according to the wind direction in your simulation
        visualize=True,
        double_cel=False
        )

    # run model
    model.simulate(waiting_time=100)