import random
import numpy as np
import matplotlib.pyplot as plt
from fluid import Fluid, Arrow3D
from network import Model, Genetic, Model_Weights

def __IND(x, y, z, N):
    return x + y * N + z * N * N

def gen_obj(N, x, y, z, a, b, c): # generating initial parallelepiped
    obj = [0 for i in range(N**3)]

    for i in range(x, x + a):
        for j in range(y, y + b):
            for k in range(z, z + c):
                obj[__IND(i, j, k, N)] = 1

    return obj

def obj_meshgrid(obj, N): # generating meshgrid of object to draw in plt
    x, y, z = [], [], []

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if obj[__IND(i, j, k, N)]:
                    x.append(i)
                    y.append(j)
                    z.append(k)
    return [x, y, z]

def add_velocities(fluid: Fluid, x_amount, y_amount, z_amount): # adding velocities to field
    new_field_x = [0 for i in range(fluid.N ** 3)]
    new_field_y = [0 for i in range(fluid.N ** 3)]
    new_field_z = [0 for i in range(fluid.N ** 3)] 

    for k in range(1, fluid.N - 1):
        for i in range(1, fluid.N - 1):
            for j in range(1, fluid.N - 1):
                if new_field_y[__IND(i, j - 1, k, fluid.N)] != 0:
                    break
                new_field_y[__IND(i, j - 1, k, fluid.N)] = y_amount
                if fluid.data["obj"][__IND(i, j, k, fluid.N)]:
                    break
            for j in range(fluid.N - 2, 0, -1):
                if new_field_y[__IND(i, j + 1, k, fluid.N)] != 0:
                    break
                new_field_y[__IND(i, j + 1, k, fluid.N)] = y_amount
                if fluid.data["obj"][__IND(i, j, k, fluid.N)]:
                    break
    
    for k in range(1, fluid.N - 1):
        for j in range(1, fluid.N - 1):
            for i in range(1, fluid.N - 1):
                if new_field_x[__IND(i - 1, j, k, fluid.N)] != 0:
                    break
                new_field_x[__IND(i - 1, j, k, fluid.N)] = x_amount
                if fluid.data["obj"][__IND(i, j, k, fluid.N)]:
                    break

            for i in range(fluid.N - 2, 0, -1):
                if new_field_x[__IND(i + 1, j, k, fluid.N)] != 0:
                    break
                new_field_x[__IND(i + 1, j, k, fluid.N)] = x_amount
                if fluid.data["obj"][__IND(i, j, k, fluid.N)]:
                    break
    
    for i in range(1, fluid.N - 1):
        for j in range(1, fluid.N - 1):
            for k in range(1, fluid.N - 1):
                if new_field_z[__IND(i, j, k - 1, fluid.N)] != 0:
                    break
                new_field_z[__IND(i, j, k - 1, fluid.N)] = z_amount
                if fluid.data["obj"][__IND(i, j, k, fluid.N)]:
                    break
            for k in range(fluid.N - 2, 0, -1):
                if new_field_z[__IND(i, j, k + 1, fluid.N)] != 0:
                    break
                new_field_z[__IND(i, j, k + 1, fluid.N)] = z_amount
                if fluid.data["obj"][__IND(i, j, k, fluid.N)]:
                    break

    for i in range(1, fluid.N - 1):
        for j in range(1, fluid.N - 1):
            for k in range(1, fluid.N - 1):
                fluid.data['Vx'][__IND(i, j, k, fluid.N)] += new_field_x[__IND(i, j, k, fluid.N)]
                fluid.data['Vy'][__IND(i, j, k, fluid.N)] += new_field_y[__IND(i, j, k, fluid.N)]
                fluid.data['Vz'][__IND(i, j, k, fluid.N)] += new_field_z[__IND(i, j, k, fluid.N)]

def clear(fluid: Fluid):
    fluid.data["Vx0"] = [0 for i in range(N*N*N)]
    fluid.data["Vy0"] = [0 for i in range(N*N*N)]
    fluid.data["Vz0"] = [0 for i in range(N*N*N)]
    fluid.data["Vx"] = [0 for i in range(N*N*N)]
    fluid.data["Vy"] = [0 for i in range(N*N*N)]
    fluid.data["Vz"] = [0 for i in range(N*N*N)]
    fluid.data["density"] = [0 for i in range(N*N*N)]
    fluid.data["s"] = [0 for i in range(N*N*N)]
    fluid.data["div"] = [0 for i in range(N*N*N)]
    fluid.data["p"] = [0 for i in range(N*N*N)]

# Main settings-------------------------

MAX_METRIC = 10**20

def metric(Cx, Cy, Cz): # Aim - to minimize it
    if Cz < 1 or Cx == 0: return MAX_METRIC
    return Cx / Cz

DRAW = False
DRAW_ARROWS = False
DRAW_OBJ = False
N = 6
ITERATIONS = 20
EPOCHS = 1000000
D_T = 0.1
VELOCITY_X = 2
VELOCITY_Y = 0
VELOCITY_Z = 0

# initial parallelipiped

P_X = 2
P_Y = 2
P_Z = 2
WIDTH = 2
DEPTH = 2
HEIGHT = 2

# Genetic algorithm settings------------

POPULATION_SIZE = 100
POOLING_SIZE = 15

MUTATION_C = 0.5
MUTATION_PROBABILITY = 0.6

CROSS_PROBABILITY = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # GRU_Vx, GRU_Vy, GRU_Vz, Relu_Layer, Sigmoid_Layer, Out_Layer | greater -> more probability to take gens from first parent

# Main settings-------------------------

if __name__ == "__main__":

    best_score = MAX_METRIC

    if DRAW:
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')

    init_obj = gen_obj(N, P_X, P_Y, P_Z, WIDTH, DEPTH, HEIGHT)
    init_fluid = Fluid(N, 1, 0.0017, D_T, 4)
    init_fluid.set_obj(init_obj)

    population = [[Model(), init_fluid.copy(), 0]  for i in range(POPULATION_SIZE)]
    for epoch in range(EPOCHS):
        for t in range(ITERATIONS):
            print("Epoch -", epoch, "Iteration -", t)
            if DRAW:
                ax.cla()
                ax.set_xlim(1, N - 1)
                ax.set_ylim(1, N - 1)
                ax.set_zlim(1, N - 1)

            for i in range(POPULATION_SIZE):
                add_velocities(population[i][1], VELOCITY_X, VELOCITY_Y, VELOCITY_Z)
                population[i][1].step()
            
            if DRAW_ARROWS:
                u = population[0][1].data["Vx"]
                v = population[0][1].data["Vy"]
                w = population[0][1].data["Vz"]

                for i in range(N):
                    for j in range(N):
                        for k in range(N):
                            if not population[0][1].data["obj"][__IND(i, j, k, N)]:
                                arrow = Arrow3D(i, j, k, u[__IND(i, j, k, N)], v[__IND(i, j, k, N)], w[__IND(i, j, k, N)], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
                                ax.add_artist(arrow)

            if DRAW_OBJ:
                ax.scatter(*obj_meshgrid(population[0][1].data["obj"], N), c = "#ff0000")
            if DRAW:
                plt.pause(0.5)
        
        for i in range(POPULATION_SIZE):
            f = population[i][1].forces_Newton()
            population[i][2] = metric(*f)
            clear(population[i][1])
        
        population = sorted(population, key=lambda x: x[2])
        print([i[2] for i in population])
        print("Best: ", population[0][2], " - metric")

        population[0][0].get_weights().save("weights.npy")

        if best_score > population[0][2]:
            best_score = population[0][2]
            np.save("obj.npy", np.array(population[0][1].data["obj"]))

        for i in range(POOLING_SIZE, POPULATION_SIZE):
            parent_1 = random.choice([j for j in range(POOLING_SIZE)])
            parent_2 = random.choice([j for j in range(POOLING_SIZE) if j != parent_1])

            population[i][0].gru_unit_u = Genetic.cross_GRU(population[parent_1][0].gru_unit_u, population[parent_2][0].gru_unit_u, CROSS_PROBABILITY[0])
            population[i][0].gru_unit_v = Genetic.cross_GRU(population[parent_1][0].gru_unit_v, population[parent_2][0].gru_unit_v, CROSS_PROBABILITY[1])
            population[i][0].gru_unit_w = Genetic.cross_GRU(population[parent_1][0].gru_unit_w, population[parent_2][0].gru_unit_w, CROSS_PROBABILITY[2])
            population[i][0].simple_relu = Genetic.cross_Simple_Layer(population[parent_1][0].simple_relu, population[parent_2][0].simple_relu, CROSS_PROBABILITY[3])
            population[i][0].simple_sigmoid = Genetic.cross_Simple_Layer(population[parent_1][0].simple_sigmoid, population[parent_2][0].simple_sigmoid, CROSS_PROBABILITY[4])
            population[i][0].simple_out = Genetic.cross_Simple_Layer(population[parent_1][0].simple_out, population[parent_2][0].simple_out, CROSS_PROBABILITY[5])

            if random.random() < MUTATION_PROBABILITY:
                if random.random() > 0.5:
                    ch = random.choice([population[i][0].gru_unit_u, population[i][0].gru_unit_v, population[i][0].gru_unit_w])
                    Genetic.mutate_GRU(ch, a = MUTATION_C)
                else:
                    ch = random.choice([population[i][0].simple_relu, population[i][0].simple_sigmoid, population[i][0].simple_out])
                    Genetic.mutate_Simple_Layer(ch, a = MUTATION_C)
        
        print("Computing new objects")
        for i in range(POPULATION_SIZE):
            new_obj = population[i][0].compute(population[i][1].data["Vx"], population[i][1].data["Vy"], population[i][1].data["Vz"],
                                                N, P_X, P_Y, P_Z, WIDTH, DEPTH, HEIGHT)
            population[i][1].set_obj(new_obj)
    if DRAW:
        plt.show()