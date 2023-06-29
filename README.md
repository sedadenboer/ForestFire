# ForestFire
Forest Fire model for Complex System Simulation course of the master Computational Science at UvA (2023).

## Problem formulation
Forest fires in Europe have significant impacts on vegetation and ecosystems. Different types of vegetation in Europe have varying degrees of proneness to wildfires. Europe's diverse ecosystems, ranging from Mediterranean shrublands to boreal forests, have distinct characteristics that influence their vulnerability to fire. Wildfires arise from natural and human causes, affecting biodiversity, altering ecosystems, and contributing to soil degradation. With climate change and the rise of more extreme weather (especially drought), forest fires start to occur more frequently. Therefore, it is important to understand the dynamics of fire spread for effective forest management, conservation, and the development of strategies to reduce the devastating effects of forest fires. 

In this case, simple simulation models, like cellular automata, are relevant in the context of studying forest fires and percolation for several reasons. For instance, they allow us to study the fire behaviour in a controlled environment. We can, for example, model how the fire spreads and interacts with different vegetation types and other external factors like wind and humidity. This provides us with a cost-effective and efficient way to investigate numerous possibilities and gather data that might be challenging or impractical to obtain through real-world experiments. Eventually, once a model is validated against real-world data, its findings can be used to apply in practice and make informed decisions in fire management policies. Moreover, the simulations can serve as an educational tool for studying characteristics of complex systems, like emergence, self-organized criticallity and percolation. 

Studying vegetation in particular, is important for understanding how climate, fire, and ecosystems work together. We can identify regional differences in fire patterns, prevent forest homogenization, help forests adapt, make better plans to manage fires, and predict what might happen in the future for smarter forest management. Therefore we decided to implement a simple cellular automata forest fire model, with a focus on critical percolation density of the forest and two different vegetation types. \
We define a 2D grid model and adhere to the percolation definition and critical percolation density defined by Christensen and Moloney (2005): 

<em>"By definition, a cluster is percolating if and only if it is infinite. Clearly, clusters that span a finite lattice from left to right or top to bottom are candidates for percolating clusters in infinite. ... In an infinite system, there exists a_ critical occupation probability, P<sub>c</sub> , such that for p < P<sub>c</sub> there is no percolating infinite cluster, while for p > P<sub>c</sub> there is a percolating infinite cluster."</em> 

Our research question is formulated as follows: \
**How does the density and ratio of different vegetation types affect percolation in a forest fire model?**

### Hypothesis
Considering one single lightning strike as our only fire source and without tree regrowth, the percolation probability depends on the connectivity between the different cell types, which depends on the vegetation type and ignition chances. Having a well mixed grid with a plant type that is more frequent in terms of biomass and which has a lower ignition chance compared to the other plant type, will help reduce the wildfire spread.

## Model implementation
A grid is initially set up with plant cells based on a predetermined plant density. The proportion between trees, grass and shrubs (3 plant types), as well as the likelihood of ignition and the humidity for one plant type, can be adjusted. Once a cell is randomly ignited, the ignition probability of neighboring cells is calculated by taking into account the number of neighboring cells that are on fire, and the external factors (ignition probability and humidity). If a randomly generated value is lower than this ratio, the cell ignites.

### Rules and assumptions
- The forest is represented as a 2D grid.
- Each cell in the grid can have one of the following states: empty, plant, fire, or burned.
- The forest is initialized with a certain density of trees, which can be randomly distributed or follow a specific pattern.
- A single initial ignition event - 'lightning strike' - starts the wild fire.
- The fire spreads from burning cells to neighboring cells based on a neighborhood type (Moore or Von Neumann).
- The probability of a cell catching fire depends on the number of burning neighbors, the ignition probability of the vegetation type in the cell, and the humidity of the cell.
- The fire burns for a specified burnout time (simulation steps) before a tree cell transitions to the burned state.
- The simulation continues until all fires are extinguished or the waiting time for ignition is reached.
- No regrowth of trees because the growth timescales are much greater than the wild fire time scales.

#### Fire chance
chance_fire = lit_neighbors_num / total_neighbors * site_igni_p * site_humidity_p
`P = (Σf_m / Σn_m) * i * h`

### Baseline
To check how the basic model implementation behaves in terms of percolation, we created investigated how the percolation probability changes with the forest density for only one type of plant, in this case trees. The model settings are noted below:
- Humidity and ignition fixed at 1
- Dimension 100×100
- Burnout time 10
- Moore neighbourhood
  
The lineplot of density versus percolation chance shows a phase transition and the first density for which the percolation probability exceeds 0.55 is taken to be the critical density. (add parameter values)

### Vegetation experiments
Experiments with 2 or 3 vegetation types which represent trees and grass and shrubs in the case of 3 plant types
2 types:
the ratio of plant bio mass is varied from 0/100 to 100/0 on the x-axis while the ignition chance for one of the plant types is varied, keeping the other ignition chance fixed.
3 types:
Generating 3 different plots by keeping the bio mass and ignition chance for one plant type fixed per plot but varying them in different plots as to create different scenarios. The remaining 2 plant types are varied as described earlier.

## Results

## Conclusion

## Structure of the repository
* The program can be run with main.py.
* Code filemap: The code filemap contains all of the scripts of the model, experiments and plots.
* Output filemap: json files generated from the experiments will be saved here, as well as the animation GIFs.
* Plots filemap: plots generated from experimenets will be saved here.

```
.
├── Code    
│     ├── Output
│     └── Plots
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```
## Getting started
### Prerequisites
This program is fully written in [Python (3.11.0)](https://www.python.org/downloads/) and to run the code you will need some dependencies that can be installed with the following line of code:

`pip install -r requirements.txt`

### Testing

## Presentation link

## Authors
- [@sedadenboer](https://github.com/sedadenboer)
- [@Alex-notmyname](https://github.com/Alex-notmyname)
- [@AkjePurr](https://github.com/AkjePurr)
- [@reinoutmensing](https://github.com/reinoutmensing)

## References
Christensen, K., & Moloney, N. R. (2005). <em>Complexity and Criticality<em>. Imperial College Press.
