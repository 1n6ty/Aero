from network import _Funcs, Model
import cmath
import numpy as np

def series(c, t):
    return sum([c[i] * cmath.exp(2 * cmath.pi * (i - int(len(c) // 2)) * t * 1j) for i in range(len(c))])

def draw_line(map_3d, x1, z1, x2, z2, y, N):
    if x2 != x1:
        if x1 > x2:
            x1, x2 = x2, x1
            z1, z2 = z2, z1

        for x in range(x1, x2 + 1):
            z = z1 + int((x - x1) * (z2 - z1) / (x2 - x1))
            map_3d[_Funcs.IND(x, y, z, N)] = 1
    if z2 != z1:
        if z1 > z2:
            x1, x2 = x2, x1
            z1, z2 = z2, z1

        for z in range(z1, z2 + 1):
            x = x1 + int((z - z1) * (x2 - x1) / (z2 - z1))
            map_3d[_Funcs.IND(x, y, z, N)] = 1
    if x1 == x2 and z1 == z2:
        map_3d[_Funcs.IND(x1, y, z1, N)] = 1

def check_and_fill(map_3d, x, y, z, w, h, x0, z0, N):
    map_cp = map_3d[:]
    map_cp[_Funcs.IND(x0, y, z0, N)] = 1
    
    for i in range(x0 - 1, x - 1, -1):
        if i == x and map_cp[_Funcs.IND(i, y, z0, N)] == 0:
            return False
        if map_cp[_Funcs.IND(i, y, z0, N)] == 0:
            map_cp[_Funcs.IND(i, y, z0, N)] = 1
        else:
            break
    
    for i in range(x0 + 1, x + w):
        if i == x + w - 1 and map_cp[_Funcs.IND(i, y, z0, N)] == 0:
            return False
        if map_cp[_Funcs.IND(i, y, z0, N)] == 0:
            map_cp[_Funcs.IND(i, y, z0, N)] = 1
        else:
            break
    
    for i in range(z0 - 1, z - 1, -1):
        if i == z and map_cp[_Funcs.IND(x0, y, i, N)] == 0:
            return False
        if map_cp[_Funcs.IND(x0, y, i, N)] == 0:
            map_cp[_Funcs.IND(x0, y, i, N)] = 1
        else:
            break
    for i in range(z0 + 1, z + h):
        if i == z + h - 1 and map_cp[_Funcs.IND(x0, y, i, N)] == 0:
            return False
        if map_cp[_Funcs.IND(x0, y, i, N)] == 0:
            map_cp[_Funcs.IND(x0, y, i, N)] = 1
        else:
            break
    for i in range(len(map_3d)):
        map_3d[i] |= map_cp[i]
    return True

def fill_contour(map_3d, x, y, z, w, h, N):
    for row_id in range(z + 1, z + h - 1):
        for col_id in range(x + 1, x + w - 1):
            if map_3d[_Funcs.IND(col_id, y, row_id, N)] != 1:
                check_and_fill(map_3d, x, y, z, w, h, col_id, row_id, N)        

def edge(x, z, f, f1, w, h):
    x1 = int(x + f.real * w)
    if x1 > x + w - 1: x1 = x + w - 1
    if x1 < x: x1 = x
    x2 = int(x + f1.real * w)
    if x2 > x + w - 1: x2 = x + w - 1
    if x2 < x: x2 = x

    z1 = int(z + f.imag * h)
    if z1 > z + h - 1: z1 = z + h - 1
    if z1 < z: z1 = z
    z2 = int(z + f1.imag * h)
    if z2 > z + h - 1: z2 = z + h - 1
    if z2 < z: z2 = z
    return [x1, z1, x2, z2]

def draw_contour(map_3d, c, x, y, z, w, d, h, N, step = 0.1):
    for k in range(y, y + d):
        f0 = series(c[k - y], 0)
        f = f0
        for i in range(1, int(1 // step)):
            f1 = series(c[k - y], i * step)
            draw_line(map_3d, *edge(x, z, f, f1, w, h), k, N)
            f = f1
        
        draw_line(map_3d, *edge(x, z, f0, f, w, h), k, N)
        fill_contour(map_3d, x, k, z, w, h, N)

def compute_new_objects(POPULATION_SIZE, population, N, P_X, P_Y, P_Z, WIDTH, DEPTH, HEIGHT, step, env_obj, Cn):
    for i in range(POPULATION_SIZE):
        try:
            c_part = population[i][0].compute(*_Funcs.norm_velocities(
                                            np.array(env_obj.data["Vx"]),
                                            np.array(env_obj.data["Vy"]),
                                            np.array(env_obj.data["Vz"])),
                                            P_X, P_Y, P_Z,
                                            WIDTH, DEPTH, HEIGHT, N)
            c = []
            for j in range(DEPTH):
                c_cur = c_part[ : ,  : Cn * 4 + 2]
                cr = c_cur[0, : int(c_cur.shape[1] // 2)]
                ci = c_cur[0, int(c_cur.shape[1] // 2) : ]
                
                c.append([cr[x] + ci[x] * 1j for x in range(int(c_cur.shape[1] // 2))])

                c_part = c_part[ : , Cn * 4 + 2 : ]

            draw_contour(population[i][1].data["obj"], c, P_X, P_Y, P_Z, WIDTH, DEPTH, HEIGHT, N, step) 
        except:
            population[i] = [Model(WIDTH * HEIGHT * DEPTH, Cn=Cn, d = DEPTH), env_obj.copy(), float("inf")]

def obj_meshgrid(obj, N): # generating meshgrid of object to draw in plt
    x, y, z = [], [], []

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if obj[_Funcs.IND(i, j, k, N)]:
                    x.append(i)
                    y.append(j)
                    z.append(k)
    return [x, y, z]