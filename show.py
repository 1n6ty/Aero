import numpy as np
import matplotlib.pyplot as plt
from network import _Funcs, Model, Model_Weights
from object_draw import obj_meshgrid, draw_contour
from fluid import Fluid, Arrow3D, add_velocities, fluid_compute, clear_env

# Main settings-------------------------

def metric(Cx, Cy, Cz): # Aim - to minimize it
    if Cz < 0 or Cx == 0: return float("inf")
    return np.log(Cz)

DRAW_ARROWS = False
DRAW_OBJ = True

D_T = 0.1
VELOCITY_X = 2
VELOCITY_Y = 0
VELOCITY_Z = 0

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

    f = input("Filename: ")

    env_f = input('Env: ')
    init_env = list(np.load(env_f))

    N = int(pow(len(init_env), 0.33)) + 1
    
    fluid = Fluid(N, 1, 0.0017, D_T, 4)
    fluid.set_obj(init_env)
    fluid_compute(EPSILON, MAX_ITER, fluid, VELOCITY_X, VELOCITY_Y, VELOCITY_Z)

    main_model = [Model(WIDTH * HEIGHT * DEPTH, C_Vn=4), fluid]
    main_model[0].set_weights(Model_Weights.load(f))
    
    contours = []
    for j in range(WIDTH):
        contours.append(main_model[0].compute(main_model[1].data["Vx"],
                                            main_model[1].data["Vy"],
                                            main_model[1].data["Vz"],
                                            P_X, P_Y, P_Z,
                                            WIDTH, DEPTH, HEIGHT, j, N))
    c, v = [], []
    for c_v in contours:
        cr = c_v[0, : int(c_v.shape[1] // 3)]
        ci = c_v[0, int(c_v.shape[1] // 3) : int(c_v.shape[1] * 2 // 3)]
        c.append([cr[i] + ci[i] * 1j for i in range(int(c_v.shape[1] // 3))])
        v.append(c_v[0, int(c_v.shape[1] * 2 // 3) :])
    
    print('C, V:', c, v, sep='\n\n')

    draw_contour(init_env, c, v, P_X, P_Y, P_Z, WIDTH, DEPTH, HEIGHT, N, 0.1)

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

            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        if not fluid.data["obj"][_Funcs.IND(i, j, k, N)]:
                            arrow = Arrow3D(i, j, k, u[_Funcs.IND(i, j, k, N)], v[_Funcs.IND(i, j, k, N)], w[_Funcs.IND(i, j, k, N)], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
                            ax.add_artist(arrow)

        if DRAW_OBJ:
            ax.scatter(*obj_meshgrid(fluid.data["obj"], N), c = "#ff0000")
        plt.pause(0.5)
    plt.show()