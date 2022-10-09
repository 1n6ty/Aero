import matplotlib.pyplot as plt
import numpy as np
from fluid import Fluid, Arrow3D

def __IND(x, y, z, N):
    return x + y * N + z * N * N

def set_obj():
    obj = [0 for i in range(6**3)]


    obj[__IND(2, 2, 3, 6)] = 1
    obj[__IND(2, 3, 3, 6)] = 1

    obj[__IND(3, 2, 2, 6)] = 1
    obj[__IND(3, 3, 2, 6)] = 1
    obj[__IND(3, 2, 3, 6)] = 1
    obj[__IND(3, 3, 3, 6)] = 1

    return obj

if __name__ == "__main__":
    fluid = Fluid(6, 1, 0.0017, 0.1, 4)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    fluid.set_obj(set_obj())

    for t in range(1000000):
        ax.cla()
        for k in range(1, 5):
            for j in range(1, 5):
                fluid.set_velocity(1, j, k, 1, 0, 0)
        fluid.step()
        
        u = fluid.data["Vx"]
        v = fluid.data["Vy"]
        w = fluid.data["Vz"]

        f = fluid.forces_Newton()
        force = Arrow3D((2, 2 + f[0]), (2, 2 + f[1]), (2, 2 + f[2]), mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
        print(f)
        
        for k in range(6):
            for i in range(6):
                for j in range(6):
                    if u[__IND(k, j, i, 6)] != 0 or v[__IND(k, j, i, 6)] != 0 or w[__IND(k, j, i, 6)] != 0:
                        arrow = Arrow3D((k, u[__IND(k, j, i, 6)] + k), (j, v[__IND(k, j, i, 6)] + j), (i, i + w[__IND(k, j, i, 6)]), mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
                        ax.add_artist(arrow)
        for k in range(0, 6):
            for i in range(0, 6):
                for j in range(0, 6):
                    if not fluid.data["obj"][__IND(i, j, k, 6)]:
                        ax.scatter(i, j, k, c = "#0000ff")
                    else:
                        ax.scatter(i, j, k, c = "#ff0000")
        plt.pause(0.05)

    plt.show()