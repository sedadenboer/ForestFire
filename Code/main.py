# main.py
#
# Course: Complex System Simulation, UvA
# Year: 2022/2023
# 
# Description: main code to run the forest fire model.
# 
# Dependencies: forest.py

from forest import Forest

def experiment(densities, n_experiments, n_simulations,
               default, dimension, burnup_time, ignition_chance, visualize):

    percolation_info = {}

    for p in densities:
        print(f"\n------DENSITY={p}-------\n")
        for n in range(n_experiments):
            print("\nEXPERIMENT:", n, "\n")

            percolation_count = 0
            # run simulation n times
            for _ in range(n_simulations):
                model = Forest(
                    default=default,
                    dimension=dimension,
                    density=p,
                    burnup_time=burnup_time,
                    ignition_chance=ignition_chance,
                    visualize=visualize
                    )
                
                model.simulate()

                # check if percolation occured
                if model.check_percolated():
                    print("PERCOLATED!")
                    percolation_count += 1
            
            # retrieve percolation probability over the n simulations
            percolation_chance = percolation_count / n_simulations
            print("percolation count:", percolation_count, ",n simulations:", n_simulations)
            print("chance:", percolation_chance)
            # save percolation probabilities per experiment in a dictionary
            if p in percolation_info:
                percolation_info[p].append(percolation_chance)
            else:
                percolation_info[p] = [percolation_chance]
            
    print(percolation_info)
    
    return percolation_info
        

# if __name__ == "__main__":
    # initialize forest fire model
       
    # results = experiment(
    #     densities=[0.2, 0.5, 0.7],
    #     n_experiments=2,
    #     n_simulations=3,
    #     default=True,
    #     dimension=400,
    #     burnup_time=1,
    #     ignition_chance=0.0001,
    #     visualize=True
    #     )
    
    # print("result:\n", results)



if __name__ == "__main__":
    # initialize forest fire model
    model = Forest(
        default=False,
        dimension=100,
        density=0.75,
        burnup_time=1,
        ignition_chance=0.0001,
        visualize=True
        )
    
    model.simulate()
