import numpy as np
N = 10
def IND(x, y, z):
    return x + N * y + z * N * N

new_env = [0 for i in range(N ** 3)]

for i in range(1, N - 1):
    for j in range(1, N - 1):
        for k in range(1, N - 1):
            if i == 2:
                if 3 <= j <= 7:
                    new_env[IND(i, j, k)] = 1
            if i == 3:
                if j == 1:
                    new_env[IND(i, j, k)] = 1
            if i == 4:
                if j == 1 or j == 9 or j == 2 or j == 8:
                    new_env[IND(i, j, k)] = 1

np.save('env.npy', new_env)