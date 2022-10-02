import math

class Fluid:

    def __init__(self, N, diff, visc, dt, iter) -> None:
        self.N = N
        self.diff = diff
        self.visc = visc
        self.dt = dt
        self.iter = iter
        self.data = {
            "Vx0": [0 for i in range(N*N)],
            "Vx": [0 for i in range(N*N)],
            "Vy0": [0 for i in range(N*N)],
            "Vy": [0 for i in range(N*N)],
            "density": [0 for i in range(N*N)],
            "s": [0 for i in range(N*N)],
            "div": [0 for i in range(N*N)],
            "p": [0 for i in range(N*N)],
            "obj": [[0 for j in range(N)] for i in range(N)]
        }
    
    def __IND(self, x, y):
        return x + y * self.N
    
    def set_obj(self, matrix_2D):
        self.data["obj"] = matrix_2D

    def add_density(self, x, y, amount):
        self.data["density"][self.__IND(x, y)] += amount
    
    def add_velocity(self, x, y, amount_x, amount_y):
        self.data["Vx"][self.__IND(x, y)] += amount_x
        self.data["Vy"][self.__IND(x, y)] += amount_y

    def set_bnd(self, b, x):
        for i in range(1, self.N - 1):
            self.data[x][self.__IND(i, 0)] = self.data[x][self.__IND(i, 1)]
            self.data[x][self.__IND(i, self.N - 1)] = self.data[x][self.__IND(i, self.N - 2)]

        for j in range(1, self.N - 1):
            self.data[x][self.__IND(0, j)] = self.data[x][self.__IND(1, j)]
            self.data[x][self.__IND(self.N - 1, j)] = self.data[x][self.__IND(self.N - 2, j)]

        self.data[x][self.__IND(0, 0)] = 0.5 * (self.data[x][self.__IND(1, 0)] + self.data[x][self.__IND(0, 1)])
        self.data[x][self.__IND(0, self.N - 1)] = 0.5 * (self.data[x][self.__IND(1, self.N - 1)] + self.data[x][self.__IND(0, self.N - 2)])
        self.data[x][self.__IND(self.N - 1, 0)] = 0.5 * (self.data[x][self.__IND(self.N - 2, 0)] + self.data[x][self.__IND(self.N - 1, 1)])
        self.data[x][self.__IND(self.N - 1, self.N - 1)] = 0.5 * (self.data[x][self.__IND(self.N - 2, self.N - 1)] + self.data[x][self.__IND(self.N - 1, self.N - 2)])


        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                if self.data["obj"][j][i]:
                    v = (self.data[x][self.__IND(i, j - 1)] if not self.data["obj"][j - 1][i] else 0) + (self.data[x][self.__IND(i, j + 1)] if not self.data["obj"][j + 1][i] else 0)
                    self.data[x][self.__IND(i, j)] = -v if b == 2 else v
                    break
            for j in range(self.N - 2, 0, -1):
                if self.data["obj"][j][i]:
                    v = (self.data[x][self.__IND(i, j - 1)] if not self.data["obj"][j - 1][i] else 0) + (self.data[x][self.__IND(i, j + 1)] if not self.data["obj"][j + 1][i] else 0)
                    self.data[x][self.__IND(i, j)] = -v if b == 2 else v
                    break
        
        for j in range(1, self.N - 1):
            for i in range(1, self.N - 1):
                if self.data["obj"][j][i]:
                    v = (self.data[x][self.__IND(i - 1, j)] if not self.data["obj"][j][i - 1] else 0) + (self.data[x][self.__IND(i - 1, j)] if not self.data["obj"][j][i + 1] else 0)
                    self.data[x][self.__IND(i, j)] = -v if b == 1 else v
                    break
            for i in range(self.N - 2, 0, -1):
                if self.data["obj"][j][i]:
                    v = (self.data[x][self.__IND(i - 1, j)] if not self.data["obj"][j][i - 1] else 0) + (self.data[x][self.__IND(i - 1, j)] if not self.data["obj"][j][i + 1] else 0)
                    self.data[x][self.__IND(i, j)] = -v if b == 1 else v
                    break

    def linear_solver(self, b, a, c, x, x0):
        c = 1 / c
        for k in range(self.iter):
            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    if self.data["obj"][j][i]:
                        continue
                    self.data[x][self.__IND(i, j)] = c * (self.data[x0][self.__IND(i, j)] + a * (self.data[x][self.__IND(i + 1, j)] + self.data[x][self.__IND(i - 1, j)] + self.data[x][self.__IND(i, j + 1)] + self.data[x][self.__IND(i, j - 1)]))
            self.set_bnd(b, x)

    def diffuse(self, b, x, x0):
        a = self.dt * self.diff * (self.N - 2)
        self.linear_solver(b, a, 1 + 6 * a, x, x0)
    
    def project(self, Vx, Vy, p, div):
        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                if self.data["obj"][j][i]:
                    continue
                self.data[div][self.__IND(i, j)] = -0.5 * (self.data[Vx][self.__IND(i + 1, j)] - self.data[Vx][self.__IND(i - 1, j)] + self.data[Vy][self.__IND(i + 1, j)] - self.data[Vy][self.__IND(i - 1, j)]) / self.N
                self.data[p][self.__IND(i, j)] = 0
        self.set_bnd(0, div)
        self.set_bnd(0, p)
        self.linear_solver(0, 1, 6, p, div)
        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                if self.data["obj"][j][i]:
                    continue
                self.data[Vx][self.__IND(i, j)] -= 0.5 * (self.data[p][self.__IND(i + 1, j)] - self.data[p][self.__IND(i - 1, j)]) * self.N
                self.data[Vy][self.__IND(i, j)] -= 0.5 * (self.data[p][self.__IND(i, j + 1)] - self.data[p][self.__IND(i, j - 1)]) * self.N
        self.set_bnd(1, Vx)
        self.set_bnd(2, Vy)
    
    def advect(self, b, d, d0, Vx, Vy):
        for i in range(1, self.N - 1):
            i_f = i
            for j in range(1, self.N - 1):
                if self.data["obj"][j][i]:
                    continue
                j_f = j
                tmp1 = self.dt * (self.N - 2) * self.data[Vx][self.__IND(i, j)]
                tmp2 = self.dt * (self.N - 2) * self.data[Vy][self.__IND(i, j)]
                x = i_f - tmp1
                y = j_f - tmp2

                if x < 0.5: x = 0.5 
                if x > self.N - 1.5: x = self.N - 1.5
                i0 = math.floor(x)
                i1 = i0 + 1
                if y < 0.5: y = 0.5 
                if y > self.N - 1.5: y = self.N - 1.5
                j0 = math.floor(y)
                j1 = j0 + 1

                s1 = x - i0
                s0 = 1 - s1
                t1 = y - j0
                t0 = 1 - t1

                self.data[d][self.__IND(i, j)] = s0 * (t0 * self.data[d0][self.__IND(i0, j0)] + t1 * self.data[d0][self.__IND(i0, j1)]) + s1 * (t0 * self.data[d0][self.__IND(i1, j0)] + t1 * self.data[d0][self.__IND(i1, j1)])
        self.set_bnd(b, d)

    def step(self):
        self.diffuse(1, "Vx0", "Vx")
        self.diffuse(2, "Vy0", "Vy")

        self.project("Vx0", "Vy0", "p", "div")

        self.advect(1, "Vx", "Vx0", "Vx0", "Vy0")
        self.advect(2, "Vy", "Vy0", "Vx0", "Vy0")

        self.project("Vx", "Vy", "p", "div")

        self.diffuse(0, "s", "density")
        self.advect(0, "density", "s", "Vx", "Vy")


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