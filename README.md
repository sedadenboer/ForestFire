# ForestFire
Forest Fire model for Complex System Simulation course of the master Computational Science at UvA (2023).

## Problem formulation
Forest fires in Europe have significant impacts on vegetation and ecosystems. Different types of vegetation in Europe have varying degrees of proneness to wildfires. Europe's diverse ecosystems, ranging from Mediterranean shrublands to boreal forests, have distinct characteristics that influence their vulnerability to fire. Wildfires arise from natural and human causes, affecting biodiversity, altering ecosystems, and contributing to soil degradation. With climate change and the rise of more extreme weather (especially drought), forest fires start to occur more frequently. Therefore, it is important to understand the dynamics of fire spread for effective forest management, conservation, and the development of strategies to reduce the devastating effects of forest fires. 

In this case, simple simulation models, like cellular automata, are relevant in the context of studying forest fires and percolation for several reasons. For instance, they allow us to study the fire behaviour in a controlled environment. We can, for example, model how the fire spreads and interacts with different vegetation types and other external factors like wind and humidity. This provides us with a cost-effective and efficient way to investigate numerous possibilities and gather data that might be challenging or impractical to obtain through real-world experiments. Eventually, once a model is validated against real-world data, it's findings can be used to apply in practice and make informed decisions in fire management policies. Moreover, the simulations can serve as an educational tool for studying characteristics of complex systems, like emergence, self-organized criticallity and percolation. 

Studying vegetation in particular, is important for understanding how climate, fire, and ecosystems work together. We can identify regional differences in fire patterns, prevent forest homogenization, help forests adapt, make better plans to manage fires, and predict what might happen in the future for smarter forest management. Therefore we decided to implement a simple cellular automata forest fire model, with a focus on critical percolation density of the forest and two different vegetation types. \
We define a 2D grid model and adhere to the percolation definition and critical percolation density defined by Christensen and Moloney (2005): 

<em>"By definition, a cluster is percolating if and only if it is infinite. Clearly, clusters that span a finite lattice from left to right or top to bottom are candidates for percolating clusters in infinite. ... In an infinite system, there exists a_ critical occupation probability, P<sub>c</sub> , such that for p < P<sub>c</sub> there is no percolating infinite cluster, while for p > P<sub>c</sub> there is a percolating infinite cluster."</em> 

Our research question is formulated as follows: \
**How does the density and arrangement of different vegetation types affect percolation in a forest fire model?**

### Hypothesis

## Model implementation

### Asumptions
### Default 
### Vegetation
### Wind

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
