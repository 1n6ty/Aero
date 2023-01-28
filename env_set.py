import bpy
import numpy as np

N = 10
output_path = "./env/env.npy"

res = [0 for i in range(N**3)]

def IND(x, y, z):
    return x + y * N + z * N * N

depsgraph = bpy.context.evaluated_depsgraph_get()

def fill_line(val, origin, step, low = -1, hig = N): # fill in (low, hig)
    point = origin[:]
    while low < point[0] < hig and low < point[1] < hig and low < point[2] < hig:
        res[IND(*point)] = val
        for i in range(3):
            point[i] += step[i]

def round(x):
    return int(x) + 1 if (x % 1) > 0.5 else int(x)

for x in range(N):
    for y in range(N):
        for z in range(N):
            if res[IND(x, y, z)] == 0:
                x_ray = bpy.context.scene.ray_cast(depsgraph, [x, y, z], [1, 0, 0])
                xn_ray = bpy.context.scene.ray_cast(depsgraph, [x, y, z], [-1, 0, 0])
                
                y_ray = bpy.context.scene.ray_cast(depsgraph, [x, y, z], [0, 1, 0])
                yn_ray = bpy.context.scene.ray_cast(depsgraph, [x, y, z], [0, -1, 0])
                
                z_ray = bpy.context.scene.ray_cast(depsgraph, [x, y, z], [0, 0, 1])
                zn_ray = bpy.context.scene.ray_cast(depsgraph, [x, y, z], [0, 0, -1])
                
                if x_ray[0] and xn_ray[0] and y_ray[0] and yn_ray[0] and z_ray[0] and zn_ray[0]:
                    x_max = round(x_ray[1][0])
                    fill_line(1, [x, y, z], [1, 0, 0], hig = (x_max if N > x_max else N))
                    
                    y_max = round(y_ray[1][1])
                    fill_line(1, [x, y, z], [0, 1, 0], hig = (y_max if N > y_max else N))
                    
                    z_max = round(z_ray[1][2])
                    fill_line(1, [x, y, z], [0, 0, 1], hig = (z_max if N > z_max else N))

np.save(output_path, res)