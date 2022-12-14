import sys
import numpy as np
from fluid import Fluid, fluid_compute, clear_env
from network import Model, Model_Weights, Genetic
from object_draw import compute_new_objects

import multiprocessing

# Main settings-------------------------

N = 10
EPSILON = 0.01
MAX_ITER = 40
EPOCHS = 1000000000000
Cn = 2
D_T = 0.1
VELOCITY_X = 2
VELOCITY_Y = 0
VELOCITY_Z = 0

# initial parallelipiped

P_X = 3
P_Y = 3
P_Z = 3
WIDTH = 5
DEPTH = 5
HEIGHT = 5

# Genetic algorithm settings------------

POPULATION_SIZE = 30 # only even numbers
POOLING_SIZE = 6

MUTATION_C = 0.2
MUTATION_PROBABILITY = 0.9

CROSS_DISTRIBUTION_INDEX = 0.5

# Main settings-------------------------

def metric(Cx, Cy, Cz): # Aim -> to minimize it
    if Cz <= 0: return float("inf")
    return Cx / Cz

if __name__ == "__main__":
    manager_dict = multiprocessing.Manager().dict()

    # init best_score
    best_score = float("inf")

    #init initial enviroment and initial fluid object 
    if len(sys.argv) < 2:
        raise ValueError('Enviromental map must be provided')

    init_env = list(np.load(sys.argv[1]))

    init_fluid = Fluid(N, 1, 0.0017, D_T, 4)
    init_fluid.set_obj(init_env)

    # init initial population
    population = [[Model(WIDTH * HEIGHT * DEPTH, Cn=Cn, d = DEPTH), init_fluid.copy(), float("inf")]  for i in range(POPULATION_SIZE)]

    # setting weights if provided
    if len(sys.argv) > 2:
        for i in range(2, len(sys.argv)):
            model_weights = Model_Weights.load(sys.argv[i])
            population[i - 2][0].set_weights(model_weights)

    # starting learning
    # Computing enviroment
    fluid_compute(EPSILON, MAX_ITER, init_fluid, -1, VELOCITY_X, VELOCITY_Y, VELOCITY_Z, manager_dict, ignore_epsilon=True)
    #starting epoch
    for epoch in range(EPOCHS):
        # Computing objects
        print("Epoch -", epoch + 1, '- Computing objects:')
        compute_new_objects(POPULATION_SIZE, population, N, P_X, P_Y, P_Z, WIDTH, DEPTH, HEIGHT, 0.1, init_fluid, 4)
        
        # Putting objects into enviroment
        print("Epoch -", epoch + 1, '- Computing velocities and metrics:')
        computes = []
        try:
            for i in range(POPULATION_SIZE):
                p = multiprocessing.Process(target=fluid_compute, args=(EPSILON, MAX_ITER, population[i][1], i, VELOCITY_X, VELOCITY_Y, VELOCITY_Z, manager_dict))
                computes.append(p)
                p.start()
            
            for pr in computes:
                pr.join()
            
        except KeyboardInterrupt:
            for pr in computes: 
                pr.terminate()

        for i in range(POPULATION_SIZE):
            print(manager_dict[i])
            population[i][2] = metric(*manager_dict[i])

        # Sorting population and removing duplicates
        for i in range(len(population)):
            if population[i][2] != float("inf") and population[i][2] in [j[2] for j in population[:i]]:
                population[i] = [Model(WIDTH * HEIGHT * DEPTH, Cn=Cn, d = DEPTH), init_fluid.copy(), float("inf")]

        population = sorted(population, key=lambda x: x[2]) 
        print("Epoch -", epoch + 1, "- Population metrics:", [i[2] for i in population])
        print("Epoch -", epoch + 1, "- Best metric:", population[0][2])

        # Saving best weights and object
        if best_score > population[0][2]:
            best_score = population[0][2]
            population[0][0].get_weights().save("weights.npy")
        
        # Updating population
        print("Epoch -", epoch + 1, '- Updating population')
        Genetic.update_population(POOLING_SIZE, POPULATION_SIZE, population, CROSS_DISTRIBUTION_INDEX, MUTATION_PROBABILITY, MUTATION_C)

        # Clearing objects' enviroments
        print("Epoch -", epoch + 1, '- Clearing enviroment:')
        clear_env(N, POPULATION_SIZE, population, init_env)

        print('\n\n\n')
