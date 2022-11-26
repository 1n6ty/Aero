import sys
import numpy as np
import matplotlib.pyplot as plt
from network import _Funcs, Model, Model_Weights
from object_draw import obj_meshgrid, draw_contour
from fluid import Fluid, Arrow3D, add_velocities, fluid_compute, clear_env

# Main settings-------------------------

DRAW_ARROWS = False
DRAW_OBJ = True

D_T = 0.1
VELOCITY_X = 2
VELOCITY_Y = 0
VELOCITY_Z = 0
Cn = 2

EPSILON = 0.01
MAX_ITER = 40

P_X = 3
P_Y = 3
P_Z = 3
WIDTH = 5
DEPTH = 5
HEIGHT = 5

# Main settings-------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError("Weights file should be provided")
    elif len(sys.argv) < 3:
        raise ValueError("Enviroment file should be provided")
    else:
        f = sys.argv[1]
        env_f = sys.argv[2]

    init_env = list(np.load(env_f))

    N = int(pow(len(init_env), 0.33)) + 1
    
    fluid = Fluid(N, 1, 0.0017, D_T, 4)
    fluid.set_obj(init_env)
    fluid_compute(EPSILON, MAX_ITER, fluid, -1, VELOCITY_X, VELOCITY_Y, VELOCITY_Z, dict(), ignore_epsilon=True)

    main_model = [Model(WIDTH * HEIGHT * DEPTH, Cn=Cn, d = DEPTH), fluid]
    main_model[0].set_weights(Model_Weights.load(f))
    
    c_part = main_model[0].compute(*_Funcs.norm_velocities(
                                    np.array(main_model[1].data["Vx"]),
                                    np.array(main_model[1].data["Vy"]),
                                    np.array(main_model[1].data["Vz"])),
                                    P_X, P_Y, P_Z,
                                    WIDTH, DEPTH, HEIGHT, N)
    c = []
    for j in range(DEPTH):
        c_cur = c_part[ : ,  : Cn * 4 + 2]
        cr = c_cur[0, : int(c_cur.shape[1] // 2)]
        ci = c_cur[0, int(c_cur.shape[1] // 2): ]
        
        c.append([cr[x] + ci[x] * 1j for x in range(int(c_cur.shape[1] // 2))])

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
        print('Forces -', force)

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