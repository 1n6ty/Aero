import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from network import _Funcs, AeroNetwork
from object_draw import obj_meshgrid, draw_contour
from fluid import Fluid, Arrow3D, add_velocities, fluid_compute, clear_env

# Main settings-------------------------

DRAW_ARROWS = False
DRAW_OBJ = True

D_T = 0.1
VELOCITY_X = 100
VELOCITY_Y = 0
VELOCITY_Z = 0
Cn = 4

EPSILON = 0.03
MAX_ITER = 40

P_X = 3
P_Y = 4
P_Z = 4
WIDTH = 5
DEPTH = 5
HEIGHT = 6

# Main settings-------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_path', default='./env/env_empty.npy', type=str, help='config path')
    parser.add_argument('--weight_path', default='none', type=str, help='config path')
    args = parser.parse_args()

    init_env = list(np.load(args.env_path))

    N = int(pow(len(init_env), 0.33)) + 1
    
    fluid = Fluid(N, 1, 0.0017, D_T, 4)
    fluid.set_obj(init_env)
    fluid_compute(EPSILON, MAX_ITER, fluid, -1, VELOCITY_X, VELOCITY_Y, VELOCITY_Z, dict(), ignore_epsilon=True)

    if args.weight_path != 'none':

        main_model = [AeroNetwork(WIDTH * HEIGHT * DEPTH, Cn=Cn, d = DEPTH), fluid]
        main_model[0].load_state_dict(torch.load(args.weight_path))
        
        c_part_inp = torch.reshape(AeroNetwork.prepare(
                                        main_model[1].data["Vx"],
                                        main_model[1].data["Vy"],
                                        main_model[1].data["Vz"],
                                        P_X, P_Y, P_Z,
                                        WIDTH, DEPTH, HEIGHT, N), (1, WIDTH * HEIGHT * DEPTH * 3))
        c_part = main_model[0](c_part_inp)
        c = []
        for j in range(DEPTH):
            c_cur = c_part[ : ,  : Cn * 4 + 2]
            cr = c_cur[0, : int(c_cur.shape[1] // 2)]
            ci = c_cur[0, int(c_cur.shape[1] // 2): ]
            
            c.append([cr[x].item() + ci[x].item() * 1j for x in range(int(c_cur.shape[1] // 2))])

            c_part = c_part[ : , Cn * 4 + 2 : ]
        draw_contour(main_model[1].data["obj"], c, P_X, P_Y, P_Z, WIDTH, DEPTH, HEIGHT, N, 0.1)

        clear_env(N, 1, [main_model], init_env)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    prev_force = np.array([EPSILON + 1, EPSILON + 1, EPSILON + 1])
    force = np.array([0, 0, 0])

    while np.max(np.abs(prev_force - force)) > EPSILON:
        prev_force = np.array(force)

        ax.cla()
        ax.set_xlim(1, N - 1)
        ax.set_ylim(1, N - 1)
        ax.set_zlim(1, N - 1)

        add_velocities(fluid, VELOCITY_X, VELOCITY_Y, VELOCITY_Z)
        fluid.step()
        
        force = fluid.forces_Newton()
        sys.stdout.write("\rForces - {0:.8f} {1:.8f} {2:.8f}".format(*force))
        sys.stdout.flush()

        if DRAW_ARROWS:
            u = fluid.data["Vx"]
            v = fluid.data["Vy"]
            w = fluid.data["Vz"]

            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    for k in range(1, N - 1):
                        if not fluid.data["obj"][_Funcs.IND(i, j, k, N)] and fluid.data["obj"][_Funcs.IND(i - 1, j, k, N)] + fluid.data["obj"][_Funcs.IND(i + 1, j, k, N)]\
                            + fluid.data["obj"][_Funcs.IND(i, j - 1, k, N)] + fluid.data["obj"][_Funcs.IND(i, j + 1, k, N)]\
                            + fluid.data["obj"][_Funcs.IND(i, j, k - 1, N)] + fluid.data["obj"][_Funcs.IND(i, j, k + 1, N)] != 0:
                            arrow = Arrow3D(i, j, k, u[_Funcs.IND(i, j, k, N)], v[_Funcs.IND(i, j, k, N)], w[_Funcs.IND(i, j, k, N)], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
                            ax.add_artist(arrow)

        if DRAW_OBJ:
            ax.scatter(*obj_meshgrid(fluid.data["obj"], N), c = "#ff0000")
        plt.pause(0.5)
    plt.show()