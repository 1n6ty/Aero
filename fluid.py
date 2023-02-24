import math
import numpy as np

class Fluid:

    def __init__(self, N, diff, visc, dt, iter) -> None:
        self.N = N
        self.diff = diff
        self.visc = visc
        self.dt = dt
        self.iter = iter
        self.data = {
            "Vx0": [0 for i in range(N*N*N)],
            "Vx": [0 for i in range(N*N*N)],
            "Vy0": [0 for i in range(N*N*N)],
            "Vy": [0 for i in range(N*N*N)],
            "Vz0": [0 for i in range(N*N*N)],
            "Vz": [0 for i in range(N*N*N)],
            "density": [0 for i in range(N*N*N)],
            "s": [0 for i in range(N*N*N)],
            "div": [0 for i in range(N*N*N)],
            "p": [0 for i in range(N*N*N)],
            "obj": [0 for i in range(N*N*N)],
        }
    
    def copy(self):
        new_fluid = Fluid(self.N, self.diff, self.visc, self.dt, self.iter)
        new_fluid.data["obj"] = self.data["obj"][:]
        return new_fluid

    def __IND(self, x, y, z):
        return x + y * self.N + z * self.N * self.N
    
    def set_obj(self, matrix_1D):
        self.data["obj"] = matrix_1D

    def add_density(self, x, y, z, amount):
        self.data["density"][self.__IND(x, y, z)] += amount
    
    def add_velocity(self, x, y, z, amount_x, amount_y, amount_z):
        self.data["Vx"][self.__IND(x, y, z)] += amount_x
        self.data["Vy"][self.__IND(x, y, z)] += amount_y
        self.data["Vz"][self.__IND(x, y, z)] += amount_z
    
    def set_velocity(self, x, y, z, amount_x, amount_y, amount_z):
        self.data["Vx"][self.__IND(x, y, z)] = amount_x
        self.data["Vy"][self.__IND(x, y, z)] = amount_y
        self.data["Vz"][self.__IND(x, y, z)] = amount_z

    def set_bnd(self, b, x):
        for k in range(1, self.N - 1):
            for i in range(1, self.N - 1):
                self.data[x][self.__IND(i, 0, k)] = self.data[x][self.__IND(i, 1, k)]
                self.data[x][self.__IND(i, self.N - 1, k)] = self.data[x][self.__IND(i, self.N - 2, k)]

        for k in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                self.data[x][self.__IND(0, j, k)] = self.data[x][self.__IND(1, j, k)]
                self.data[x][self.__IND(self.N - 1, j, k)] = self.data[x][self.__IND(self.N - 2, j, k)]
        
        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                self.data[x][self.__IND(i, j, 0)] = self.data[x][self.__IND(i, j, 1)]
                self.data[x][self.__IND(i, j, self.N - 1)] = self.data[x][self.__IND(i, j, self.N - 2)]

        
        for k in range(1, self.N - 1):
            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    if self.data["obj"][self.__IND(i, j, k)]:
                        if not self.data["obj"][self.__IND(i - 1, j, k)]:
                            self.data["Vx"][self.__IND(i, j, k)] = (-1 if b == 1 else 1) * self.data["Vx"][self.__IND(i - 1, j, k)]
                        if not self.data["obj"][self.__IND(i + 1, j, k)]:
                            self.data["Vx"][self.__IND(i, j, k)] = (-1 if b == 1 else 1) * self.data["Vx"][self.__IND(i + 1, j, k)]
                        
                        if not self.data["obj"][self.__IND(i, j - 1, k)]:
                            self.data["Vy"][self.__IND(i, j, k)] = (-1 if b == 2 else 1) * self.data["Vy"][self.__IND(i, j - 1, k)]
                        if not self.data["obj"][self.__IND(i, j + 1, k)]:
                            self.data["Vy"][self.__IND(i, j, k)] = (-1 if b == 2 else 1) * self.data["Vy"][self.__IND(i, j + 1, k)]
                        
                        if not self.data["obj"][self.__IND(i, j, k - 1)]:
                            self.data["Vz"][self.__IND(i, j, k)] = (-1 if b == 3 else 1) * self.data["Vz"][self.__IND(i, j, k - 1)]
                        if not self.data["obj"][self.__IND(i, j + 1, k)]:
                            self.data["Vz"][self.__IND(i, j, k)] = (-1 if b == 3 else 1) * self.data["Vz"][self.__IND(i, j, k + 1)]
                            

    def linear_solver(self, b, a, c, x, x0):
        c = 1 / c
        for t in range(self.iter):
            for k in range(1, self.N - 1):
                for i in range(1, self.N - 1):
                    for j in range(1, self.N - 1):
                        if self.data["obj"][self.__IND(i, j, k)]:
                            continue
                        self.data[x][self.__IND(i, j, k)] = c * (self.data[x0][self.__IND(i, j, k)] + a * (self.data[x][self.__IND(i + 1, j, k)] + self.data[x][self.__IND(i - 1, j, k)] + self.data[x][self.__IND(i, j + 1, k)] + self.data[x][self.__IND(i, j - 1, k)] + self.data[x][self.__IND(i, j, k + 1)] + self.data[x][self.__IND(i, j, k - 1)]))
            self.set_bnd(b, x)

    def diffuse(self, b, x, x0):
        a = self.dt * self.diff * (self.N - 2) * (self.N - 2)
        self.linear_solver(b, a, 1 + 6 * a, x, x0)
    
    def project(self, Vx, Vy, Vz, p, div):
        for k in range(1, self.N - 1):
            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    if self.data["obj"][self.__IND(i, j, k)]:
                        continue
                    self.data[div][self.__IND(i, j, k)] = -0.5 * (self.data[Vx][self.__IND(i + 1, j, k)] - self.data[Vx][self.__IND(i - 1, j, k)] + self.data[Vy][self.__IND(i, j + 1, k)] - self.data[Vy][self.__IND(i, j - 1, k)] + self.data[Vy][self.__IND(i, j, k + 1)] - self.data[Vy][self.__IND(i, j, k - 1)]) / self.N
                    self.data[p][self.__IND(i, j, k)] = 0
        self.set_bnd(0, div)
        self.set_bnd(0, p)
        self.linear_solver(0, 1, 6, p, div)

        for k in range(1, self.N - 1):
            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    if self.data["obj"][self.__IND(i, j, k)]:
                        continue
                    self.data[Vx][self.__IND(i, j, k)] -= 0.5 * (self.data[p][self.__IND(i + 1, j, k)] - self.data[p][self.__IND(i - 1, j, k)]) * self.N
                    self.data[Vy][self.__IND(i, j, k)] -= 0.5 * (self.data[p][self.__IND(i, j + 1, k)] - self.data[p][self.__IND(i, j - 1, k)]) * self.N
                    self.data[Vz][self.__IND(i, j, k)] -= 0.5 * (self.data[p][self.__IND(i, j, k + 1)] - self.data[p][self.__IND(i, j, k - 1)]) * self.N
        self.set_bnd(1, Vx)
        self.set_bnd(2, Vy)
        self.set_bnd(3, Vz)
    
    def _search4edge(self, x, y, z):
        res = [[], [], []]
        for i in range(x, -1, -1):
            if self.data["obj"][self.__IND(i, y, z)] or i == 0:
                res[0].append(i)
                break
        for i in range(x, self.N):
            if self.data["obj"][self.__IND(i, y, z)] or i == self.N - 2:
                res[0].append(i)
                break
        
        for j in range(y, -1, -1):
            if self.data["obj"][self.__IND(x, j, z)] or j == 0:
                res[1].append(j)
                break
        for j in range(y, self.N):
            if self.data["obj"][self.__IND(x, j, z)] or j == self.N - 2:
                res[1].append(j)
                break
        
        for k in range(z, -1, -1):
            if self.data["obj"][self.__IND(x, y, k)] or k == 0:
                res[2].append(k)
                break
        for k in range(z, self.N):
            if self.data["obj"][self.__IND(x, y, k)] or k == self.N - 2:
                res[2].append(k)
                break
        
        return res

    def advect(self, b, d, d0, Vx, Vy, Vz):
        for k in range(1, self.N - 1):
            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    if self.data["obj"][self.__IND(i, j, k)]:
                        continue
                    tmp1 = self.dt * (self.N - 2) * self.data[Vx][self.__IND(i, j, k)]
                    tmp2 = self.dt * (self.N - 2) * self.data[Vy][self.__IND(i, j, k)]
                    tmp3 = self.dt * (self.N - 2) * self.data[Vz][self.__IND(i, j, k)]
                    x = i - tmp1
                    y = j - tmp2
                    z = k - tmp3

                    edge = self._search4edge(i, j, k)
                    if x < edge[0][0]: x = edge[0][0]
                    if x > edge[0][1]: x = edge[0][1]
                    i0 = math.floor(x)
                    i1 = i0 + 1
                    if y < edge[1][0]: y = edge[1][0]
                    if y > edge[1][1]: y = edge[1][1]
                    j0 = math.floor(y)
                    j1 = j0 + 1
                    if z < edge[2][0]: z = edge[2][0]
                    if z > edge[2][1]: z = edge[2][1]
                    k0 = math.floor(z)
                    k1 = k0 + 1

                    s1 = x - i0
                    s0 = 1 - s1
                    t1 = y - j0
                    t0 = 1 - t1
                    u1 = z - k0
                    u0 = 1 - u1

                    self.data[d][self.__IND(i, j, k)] = s0 * (t0 * (u0 * self.data[d0][self.__IND(i0, j0, k0)] + u1 * self.data[d0][self.__IND(i0, j0, k1)]) + t1 * (u0 * self.data[d0][self.__IND(i0, j1, k0)] + u1 * self.data[d0][self.__IND(i0, j1, k1)])) + s1 * (t0 * (u0 * self.data[d0][self.__IND(i1, j0, k0)] + u1 * self.data[d0][self.__IND(i1, j0, k1)]) + t1 * (u0 * self.data[d0][self.__IND(i1, j1, k0)] + u1 * self.data[d0][self.__IND(i1, j1, k1)]))
        self.set_bnd(b, d)

    def step(self):
        self.diffuse(1, "Vx0", "Vx")
        self.diffuse(2, "Vy0", "Vy")
        self.diffuse(3, "Vz0", "Vz")

        self.project("Vx0", "Vy0", "Vz0", "p", "div")
        
        self.advect(1, "Vx", "Vx0", "Vx0", "Vy0", "Vz0")
        self.advect(2, "Vy", "Vy0", "Vx0", "Vy0", "Vz0")
        self.advect(3, "Vz", "Vz0", "Vx0", "Vy0", "Vz0")

        self.project("Vx", "Vy", "Vz", "p", "div")

        self.diffuse(0, "s", "density")
        self.advect(0, "density", "s", "Vx", "Vy", "Vz")
    
    def forces_Newton(self):
        f_x, f_y, f_z = 0, 0, 0
        m = 0.0013

        for k in range(1, self.N - 1):
            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    if self.data["obj"][self.__IND(i, j, k)]:
                        if not self.data["obj"][self.__IND(i - 1, j, k)]:
                            f_x += (self.data["Vx"][self.__IND(i - 1, j, k)] if self.data["Vx"][self.__IND(i - 1, j, k)] > 0 else 0) * m / self.dt
                        if not self.data["obj"][self.__IND(i - 1, j, k)]:
                            f_x += (self.data["Vx"][self.__IND(i + 1, j, k)] if self.data["Vx"][self.__IND(i + 1, j, k)] < 0 else 0) * m / self.dt

                        if not self.data["obj"][self.__IND(i, j - 1, k)]:
                            f_y += (self.data["Vy"][self.__IND(i, j - 1, k)] if self.data["Vy"][self.__IND(i, j - 1, k)] > 0 else 0) * m / self.dt
                        if not self.data["obj"][self.__IND(i, j + 1, k)]:
                            f_y += (self.data["Vy"][self.__IND(i, j + 1, k)] if self.data["Vy"][self.__IND(i, j + 1, k)] < 0 else 0) * m / self.dt

                        if not self.data["obj"][self.__IND(i, j, k - 1)]:
                            f_z += (self.data["Vz"][self.__IND(i, j, k - 1)] if self.data["Vz"][self.__IND(i, j, k - 1)] > 0 else 0) * m / self.dt
                        if not self.data["obj"][self.__IND(i, j, k + 1)]:
                            f_z += (self.data["Vz"][self.__IND(i, j, k + 1)] if self.data["Vz"][self.__IND(i, j, k + 1)] < 0 else 0) * m / self.dt
                        
        return np.array([f_x, f_y, f_z])


from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def get_arrow_artists(fluid: Fluid, N):
    res = []

    u = fluid.data["Vx"]
    v = fluid.data["Vy"]
    w = fluid.data["Vz"]

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if not fluid.data["obj"][_Funcs.IND(i, j, k, N)]:
                    res.append(Arrow3D(i, j, k, u[_Funcs.IND(i, j, k, N)], v[_Funcs.IND(i, j, k, N)], w[_Funcs.IND(i, j, k, N)], mutation_scale=20, lw=1, arrowstyle="-|>", color="k"))
    return res

from network import _Funcs

def add_velocities(fluid: Fluid, x_amount, y_amount, z_amount): # adding velocities to field
    new_field_x = [0 for i in range(fluid.N ** 3)]
    new_field_y = [0 for i in range(fluid.N ** 3)]
    new_field_z = [0 for i in range(fluid.N ** 3)] 

    for k in range(1, fluid.N - 1):
        for i in range(1, fluid.N - 1):
            if y_amount > 0:
                for j in range(1, fluid.N - 1):
                    new_field_y[_Funcs.IND(i, j - 1, k, fluid.N)] = y_amount
                    if fluid.data["obj"][_Funcs.IND(i, j, k, fluid.N)]:
                        break
            else:
                for j in range(fluid.N - 2, 0, -1):
                    new_field_y[_Funcs.IND(i, j + 1, k, fluid.N)] = y_amount
                    if fluid.data["obj"][_Funcs.IND(i, j, k, fluid.N)]:
                        break
    
    for k in range(1, fluid.N - 1):
        for j in range(1, fluid.N - 1):
            if x_amount > 0:
                for i in range(1, fluid.N - 1):
                    new_field_x[_Funcs.IND(i - 1, j, k, fluid.N)] = x_amount
                    if fluid.data["obj"][_Funcs.IND(i, j, k, fluid.N)]:
                        break
            else:
                for i in range(fluid.N - 2, 0, -1):
                    new_field_x[_Funcs.IND(i + 1, j, k, fluid.N)] = x_amount
                    if fluid.data["obj"][_Funcs.IND(i, j, k, fluid.N)]:
                        break
    
    for i in range(1, fluid.N - 1):
        for j in range(1, fluid.N - 1):
            if z_amount > 0:
                for k in range(1, fluid.N - 1):
                    new_field_z[_Funcs.IND(i, j, k - 1, fluid.N)] = z_amount
                    if fluid.data["obj"][_Funcs.IND(i, j, k, fluid.N)]:
                        break
            else:
                for k in range(fluid.N - 2, 0, -1):
                    new_field_z[_Funcs.IND(i, j, k + 1, fluid.N)] = z_amount
                    if fluid.data["obj"][_Funcs.IND(i, j, k, fluid.N)]:
                        break

    for i in range(1, fluid.N - 1):
        for j in range(1, fluid.N - 1):
            for k in range(1, fluid.N - 1):
                fluid.data['Vx'][_Funcs.IND(i, j, k, fluid.N)] += new_field_x[_Funcs.IND(i, j, k, fluid.N)]
                fluid.data['Vy'][_Funcs.IND(i, j, k, fluid.N)] += new_field_y[_Funcs.IND(i, j, k, fluid.N)]
                fluid.data['Vz'][_Funcs.IND(i, j, k, fluid.N)] += new_field_z[_Funcs.IND(i, j, k, fluid.N)]

def clear_env(N, POPULATION_SIZE, population, init_env):
    for i in range(POPULATION_SIZE):
        population[i][1].data["Vx0"] = [0 for j in range(N*N*N)]
        population[i][1].data["Vy0"] = [0 for j in range(N*N*N)]
        population[i][1].data["Vz0"] = [0 for j in range(N*N*N)]
        population[i][1].data["Vx"] = [0 for j in range(N*N*N)]
        population[i][1].data["Vy"] = [0 for j in range(N*N*N)]
        population[i][1].data["Vz"] = [0 for j in range(N*N*N)]
        population[i][1].data["density"] = [0 for j in range(N*N*N)]
        population[i][1].data["s"] = [0 for j in range(N*N*N)]
        population[i][1].data["div"] = [0 for j in range(N*N*N)]
        population[i][1].data["p"] = [0 for j in range(N*N*N)]
        population[i][1].data["obj"] = init_env[:]

def fluid_compute(epsilon, max_iter, fluid: Fluid, j, v_x, v_y, v_z, manager_dict, ignore_epsilon = False):
    prev_force = np.array([epsilon + 1, epsilon + 1, epsilon + 1])
    force = np.array([0, 0, 0])

    iter = 0

    while (ignore_epsilon or np.max(np.abs(prev_force - force)) > epsilon) and iter < max_iter:
        prev_force = np.array(force)

        add_velocities(fluid, v_x, v_y, v_z)
        fluid.step()

        force = fluid.forces_Newton()

        iter += 1
    
    manager_dict[j] = force