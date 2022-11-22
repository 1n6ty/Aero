import sys
import numpy as np
from fluid import Fluid, fluid_compute
from network import Model, Model_Weights, backpropagation

# Main settings-------------------------

N = 10
EPOCHS = 200
EPSILON = 0.01
MAX_ITER = 40
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

# Main settings-------------------------

if __name__ == "__main__":
    #init initial enviroment and initial fluid object 
    
    if len(sys.argv) < 2:
        raise ValueError('Enviromental map must be provided')

    init_env = list(np.load(sys.argv[1]))

    init_fluid = Fluid(N, 1, 0.0017, D_T, 4)
    init_fluid.set_obj(init_env)

    init_objects = [
        [
            np.array([[0.5, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 1]]),
            np.array([[0.5, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 1]]),
            np.array([[0.5, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 1]]),
            np.array([[0.5, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 1]]),
            np.array([[0.5, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 1]]),
        ]
    ]

    # init main model
    main_model = [Model(WIDTH * HEIGHT * DEPTH, C_Vn=4), init_fluid.copy()]

    # setting weights if provided
    if len(sys.argv) > 2:
        model_weights = Model_Weights.load(sys.argv[2])
        main_model[0].set_weights(model_weights)

    # starting learning
    # Computing enviroment
    fluid_compute(EPSILON, MAX_ITER, init_fluid, VELOCITY_X, VELOCITY_Y, VELOCITY_Z)

    #starting epoch
    for epoch in range(EPOCHS):
        # Computing objects
        print("Epoch -", epoch + 1, '- Computing models:')
        for i in init_objects:
            for j in range(WIDTH):
                out = main_model[0].compute(main_model[1].data["Vx"],
                                            main_model[1].data["Vy"],
                                            main_model[1].data["Vz"],
                                            P_X, P_Y, P_Z,
                                            WIDTH, DEPTH, HEIGHT, j, N)
                backpropagation(main_model, out, i[j], epoch + 1, a = 0.3)

        print('\n\n\n')

    # Saving best weights
    main_model[0].get_weights().save("weights.npy")
