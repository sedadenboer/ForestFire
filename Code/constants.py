# constants.py
#
# Course: Complex System Simulation, UvA
# Year: 2022/2023
# 
# Description: This module defines the ignition probability &&
# humidity reaction for each vegetation, and probability more 
# constants if we need

###################### Site states ######################
EMPTY = 0
TREE = 1
GRASS = 2
SHRUB = 3
FIRE = 4
BURNED = 5

###################### Ignition probability ######################
P_TREE = 0.8
P_GRASS = 0.3
P_SHRUB = 0.25

###################### Humidity reaction ######################
H_TREE = 1
H_SHRUB = 1
H_GRASS = 1

###################### State colors ######################
EMPTY_C = 'tan'
TREE_C = 'forestgreen'
GRASS_C = 'yellowgreen'
SHRUB_C = 'olivedrab'
FIRE_C = 'crimson'
BURNED_C = 'black'