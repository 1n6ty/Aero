import matplotlib.pyplot as plt
import numpy as np
from fluid import Fluid, Arrow3D

if __name__ == "__main__":
    fluid = Fluid(10, 1, 0.0017, 0.1, 4)

    fluid.set_obj([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(range(10), range(10))

    for t in range(1000000):
        ax.cla()
        ax.set_zlim(zmin=0.001, zmax=0.1)
        for i in range(0, 10):
            for j in range(0, 10):
                if not fluid.data["obj"][i][j]:
                    fluid.add_velocity(i, j, 4, 0)
        fluid.step()
        z = np.array(fluid.data["p"]).reshape((10, 10)) * 0
        u = np.array(fluid.data["Vx"]).reshape((10, 10))
        v = np.array(fluid.data["Vy"]).reshape((10, 10))
        for i in range(10):
            for j in range(10):
                if u[i][j] != 0 or v[i][j] != 0:
                    arrow = Arrow3D((x[i][j], u[i][j] + x[i][j]), (y[i][j], v[i][j] + y[i][j]), (z[i][j], z[i][j]), mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
                    ax.add_artist(arrow)
        for i in range(0, 10):
            for j in range(0, 10):
                if not fluid.data["obj"][i][j]:
                    ax.scatter(i, j, 0, c = "#0000ff")
                else:
                    ax.scatter(i, j, 0, c = "#ff0000")
        plt.pause(0.05)

    plt.show()