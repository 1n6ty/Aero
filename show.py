import numpy as np
import matplotlib.pyplot as plt
from fluid import Fluid, Arrow3D

# Main settings-------------------------

def metric(Cx, Cy, Cz): # Aim - to minimize it
    if Cz == 0 or Cx == 0: return 10**20
    return abs(Cx / Cz)

DRAW = False
DRAW_ARROWS = False
DRAW_OBJ = False
N = 6
D_T = 0.1
VELOCITY_X = 2
VELOCITY_Y = 0
VELOCITY_Z = 0

# Main settings-------------------------

def __IND(x, y, z, N):
    return x + y * N + z * N * N

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

if __name__ == "__main__":
    fluid = Fluid(N, 1, 0.0017, D_T, 4)

    f = input("Filename: ")
    obj = np.load(f)

    fluid.set_obj(obj)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    for t in range(20):
        ax.cla()
        ax.set_xlim(1, N - 1)
        ax.set_ylim(1, N - 1)
        ax.set_zlim(1, N - 1)

        add_velocities(fluid, VELOCITY_X, VELOCITY_Y, VELOCITY_Z)
        fluid.step()
        
        u = fluid.data["Vx"]
        v = fluid.data["Vy"]
        w = fluid.data["Vz"]

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if not fluid.data["obj"][__IND(i, j, k, N)]:
                        arrow = Arrow3D(i, j, k, u[__IND(i, j, k, N)], v[__IND(i, j, k, N)], w[__IND(i, j, k, N)], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
                        ax.add_artist(arrow)

        ax.scatter(*obj_meshgrid(fluid.data["obj"], N), c = "#ff0000")
        print("Forces: ", fluid.forces_Newton())
        plt.pause(0.5)

    plt.show()