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
FIRE = 10
BURNED = -1

###################### Ignition probability ######################
P_TREE = 1
P_SHRUB = 0.25
P_GRASS = 0.75

###################### Humidity reaction ######################
H_TREE = 0.3
H_SHRUB = 0.6
H_GRASS = 0.7