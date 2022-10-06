import math

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

    #fix bnd for objects
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
                        v = self.data[x][self.__IND(i, j - 1, k)] + self.data[x][self.__IND(i, j + 1, k)]
                        if not self.data["obj"][self.__IND(i, j - 1, k)] and not self.data["obj"][self.__IND(i, j - 1, k)]: v /= 2
                        self.data[x][self.__IND(i, j, k)] = -v if b == 2 else v
                        break
                for j in range(self.N - 2, 0, -1):
                    if self.data["obj"][self.__IND(i, j, k)]:
                        v = (self.data[x][self.__IND(i, j - 1, k)] if not self.data["obj"][self.__IND(i, j - 1, k)] else 0) + (self.data[x][self.__IND(i, j + 1, k)] if not self.data["obj"][self.__IND(i, j + 1, k)] else 0)
                        self.data[x][self.__IND(i, j, k)] = -v if b == 2 else v
                        break
        
        for k in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                for i in range(1, self.N - 1):
                    if self.data["obj"][self.__IND(i, j, k)]:
                        v = (self.data[x][self.__IND(i - 1, j, k)] if not self.data["obj"][self.__IND(i - 1, j, k)] else 0) + (self.data[x][self.__IND(i + 1, j, k)] if not self.data["obj"][self.__IND(i + 1, j, k)] else 0)
                        self.data[x][self.__IND(i, j, k)] = -v if b == 1 else v
                        print(i, j, k, self.data[x][self.__IND(i, j, k)])
                        break
                for i in range(self.N - 2, 0, -1):
                    if self.data["obj"][self.__IND(i, j, k)]:
                        v = (self.data[x][self.__IND(i - 1, j, k)] if not self.data["obj"][self.__IND(i - 1, j, k)] else 0) + (self.data[x][self.__IND(i + 1, j, k)] if not self.data["obj"][self.__IND(i + 1, j, k)] else 0)
                        self.data[x][self.__IND(i, j, k)] = -v if b == 1 else v
                        break
        
        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                for k in range(1, self.N - 1):
                    if self.data["obj"][self.__IND(i, j, k)]:
                        v = (self.data[x][self.__IND(i, j, k - 1)] if not self.data["obj"][self.__IND(i, j, k - 1)] else 0) + (self.data[x][self.__IND(i, j, k + 1)] if not self.data["obj"][self.__IND(i, j, k + 1)] else 0)
                        self.data[x][self.__IND(i, j, k)] = -v if b == 3 else v
                        break
                for k in range(self.N - 2, 0, -1):
                    if self.data["obj"][self.__IND(i, j, k)]:
                        v = (self.data[x][self.__IND(i, j, k - 1)] if not self.data["obj"][self.__IND(i, j, k - 1)] else 0) + (self.data[x][self.__IND(i, j, k + 1)] if not self.data["obj"][self.__IND(i, j, k + 1)] else 0)
                        self.data[x][self.__IND(i, j, k)] = -v if b == 3 else v
                        break

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

                    if x < 0.5: x = 0.5 
                    if x > self.N - 1.5: x = self.N - 1.5
                    i0 = math.floor(x)
                    i1 = i0 + 1
                    if y < 0.5: y = 0.5 
                    if y > self.N - 1.5: y = self.N - 1.5
                    j0 = math.floor(y)
                    j1 = j0 + 1
                    if z < 0.5: z = 0.5 
                    if z > self.N - 1.5: z = self.N - 1.5
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


from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)