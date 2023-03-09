import sys
import argparse
import numpy as np

import torch
import torch.nn as nn

from fluid import Fluid, fluid_compute
from network import AeroNetwork

# Main settings-------------------------

N = 12
EPOCHS = 1000
EPSILON = 0.01
MAX_ITER = 40
D_T = 0.1
C_N = 4
VELOCITY_X = 100
VELOCITY_Y = 0
VELOCITY_Z = 0

# initial parallelipiped

P_X = 3
P_Y = 4
P_Z = 4
WIDTH = 5
DEPTH = 5
HEIGHT = 6

# Main settings-------------------------

if __name__ == "__main__":
    #init initial enviroment and initial fluid object 
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_path', default='./env/env_empty.npy', type=str, help='config path')
    parser.add_argument('--weight_path', default='none', type=str, help='config path')
    parser.add_argument('--output_path', default='./computed_weights/model.pt', type=str, help='config path')
    args = parser.parse_args()

    init_env = list(np.load(args.env_path))

    init_fluid = Fluid(N, 1, 0.0017, D_T, 4)
    init_fluid.set_obj(init_env)

    init_object = torch.tensor([[0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0,
                                0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0,
                                0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0,
                                0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0,
                                0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0,]]) # array with shape (1, DEPTH * (C_N * 4 + 2)); each subarray with (C_N * 4 + 2) items looks like [c_real ... (C_N * 2 + 1 times), c_imaginary ... (C_N * 2 + 1 times)]

    # init main model
    main_model = AeroNetwork(arr_shape=WIDTH * HEIGHT * DEPTH, Cn=C_N, d=DEPTH)

    # setting weights if provided
    if args.weight_path != 'none':
        main_model.load_state_dict(torch.load(args.weight_path))
    
    # setting optimizer
    criterion = nn.MSELoss()
    opt = torch.optim.SGD(main_model.parameters(), lr=0.005)

    # starting learning
    # Computing enviroment
    fluid_compute(EPSILON, MAX_ITER, init_fluid, -1, VELOCITY_X, VELOCITY_Y, VELOCITY_Z, dict(), ignore_epsilon=True)

    #starting epoch
    for epoch in range(EPOCHS):
        # Computing objects
        with torch.no_grad():
            inp = torch.reshape(AeroNetwork.prepare(init_fluid.data["Vx"],
                                        init_fluid.data["Vy"],
                                        init_fluid.data["Vz"],
                                        P_X, P_Y, P_Z,
                                        WIDTH, DEPTH, HEIGHT, N), (1, WIDTH * HEIGHT * DEPTH * 3))
        
        out = main_model(inp)
        loss = criterion(out, init_object)

        opt.zero_grad()

        loss.backward()
        opt.step()

        sys.stdout.write("\rEpoch - {0}, Loss - {1:.6f}".format(epoch + 1, loss.item()))
        sys.stdout.flush()

    # Saving weights
    torch.save(main_model.state_dict(), args.output_path)
