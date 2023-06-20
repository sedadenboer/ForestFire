from forest import Forest
from vegetation import Plant
from visualize import visualize


if __name__ == "__main__":

    model = Forest(
        dimension=100,
        density=0.8,
        burnup_chance=0.1,
        visualize=True
        )

    model.simulate()