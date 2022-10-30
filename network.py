from random import random
import numpy as np

class _Funcs:
    @staticmethod
    def relu(x):
        k = np.copy(x)
        np.place(k, k >= 0, [1])
        np.place(k, k < 0, [0.01])
        return x * k
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    @staticmethod
    def IND(x, y, z, N):
        return x + y * N + z * N * N


class Simple_layer:
    # input_shape = (1, N1); output_shape = (1, N2); weights_shape = (N1, N2);
    def __init__(self, input_size = None, output_size = None, activation_func = "sigm") -> None:
        if not (input_size and output_size): raise ValueError("input_size and output_size must be provided")

        self.input_size = input_size
        self.output_size = output_size
        self.activation_func = activation_func
        
        self.weights = np.random.random((input_size + 1, output_size)) * 2 - 1

    def forward(self, inp):
        # pass input forward throw weighting and activation
        self.sum_x = np.dot(np.append(inp, 1), self.weights)
        if self.activation_func == "sigm":
            x_active = _Funcs.sigmoid(self.sum_x)
        if self.activation_func == "relu":
            x_active = _Funcs.relu(self.sum_x)
        if self.activation_func == "tanh":
            x_active = _Funcs.tanh(self.sum_x)
        return x_active


class GRU_unit:
    def __init__(self, init_state = np.array([[0, 0, 0]])) -> None:
        self.state = init_state
        self.weights = (np.random.random((1, 9)) * 2 - 1) * 0.0001
    
    def forward(self, x):
        z = _Funcs.sigmoid(self.weights[0, 0] * x + self.weights[0, 3] * self.state + self.weights[0, 6])
        r = _Funcs.sigmoid(self.weights[0, 1] * x + self.weights[0, 4] * self.state + self.weights[0, 7])
        h = _Funcs.tanh(self.weights[0, 2] * x + self.weights[0, 5] * self.state * r + self.weights[0, 8])
        self.state = z * h + (1 - z) * self.state
        return self.state


class MD_GRU:
    @staticmethod
    def compute(arr_1D, N, gru_unit: GRU_unit):
        new_arr = [np.zeros((1, 3)) for i in range((N - 2)**3)]
        
        state_arr = np.zeros((N, N, N, 3))
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    gru_unit.state = state_arr[k, j, i - 1] + state_arr[k, j - 1, i] + state_arr[k - 1, j, i]
                    c = gru_unit.forward(np.array([[
                        arr_1D[_Funcs.IND(i - 1, j, k, N)], arr_1D[_Funcs.IND(i, j - 1, k, N)], arr_1D[_Funcs.IND(i, j, k - 1, N)]
                    ]]))
                    new_arr[_Funcs.IND(i - 1, j - 1, k - 1, N - 2)] += c
                    state_arr[k, j, i] = c
        state_arr = np.zeros((N, N, N, 3))
        for i in range(N - 2, 0, -1):
            for j in range(N - 2, 0, -1):
                for k in range(N - 2, 0, -1):
                    gru_unit.state = state_arr[k, j, i + 1] + state_arr[k, j + 1, i] + state_arr[k + 1, j, i]
                    c = gru_unit.forward(np.array([[
                        arr_1D[_Funcs.IND(i + 1, j, k, N)], arr_1D[_Funcs.IND(i, j + 1, k, N)], arr_1D[_Funcs.IND(i, j, k + 1, N)]
                    ]]))
                    new_arr[_Funcs.IND(i - 1, j - 1, k - 1, N - 2)] += c
                    state_arr[k, j, i] = c
        
        state_arr = np.zeros((N, N, N, 3))
        for i in range(N - 2, 0, -1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    gru_unit.state = state_arr[k, j, i + 1] + state_arr[k, j - 1, i] + state_arr[k - 1, j, i]
                    c = gru_unit.forward(np.array([[
                        arr_1D[_Funcs.IND(i + 1, j, k, N)], arr_1D[_Funcs.IND(i, j - 1, k, N)], arr_1D[_Funcs.IND(i, j, k - 1, N)]
                    ]]))
                    new_arr[_Funcs.IND(i - 1, j - 1, k - 1, N - 2)] += c
                    state_arr[k, j, i] = c
        state_arr = np.zeros((N, N, N, 3))
        for i in range(1, N - 1):
            for j in range(N - 2, 0, -1):
                for k in range(N - 2, 0, -1):
                    gru_unit.state = state_arr[k, j, i - 1] + state_arr[k, j + 1, i] + state_arr[k + 1, j, i]
                    c = gru_unit.forward(np.array([[
                        arr_1D[_Funcs.IND(i - 1, j, k, N)], arr_1D[_Funcs.IND(i, j + 1, k, N)], arr_1D[_Funcs.IND(i, j, k + 1, N)]
                    ]]))
                    new_arr[_Funcs.IND(i - 1, j - 1, k - 1, N - 2)] += c
                    state_arr[k, j, i] = c

        state_arr = np.zeros((N, N, N, 3))
        for i in range(1, N - 1):
            for j in range(N - 2, 0, -1):
                for k in range(1, N - 1):
                    gru_unit.state = state_arr[k, j, i - 1] + state_arr[k, j + 1, i] + state_arr[k - 1, j, i]
                    c = gru_unit.forward(np.array([[
                        arr_1D[_Funcs.IND(i - 1, j, k, N)], arr_1D[_Funcs.IND(i, j + 1, k, N)], arr_1D[_Funcs.IND(i, j, k - 1, N)]
                    ]]))
                    new_arr[_Funcs.IND(i - 1, j - 1, k - 1, N - 2)] += c
                    state_arr[k, j, i] = c
        state_arr = np.zeros((N, N, N, 3))
        for i in range(N - 2, 0, -1):
            for j in range(1, N - 1):
                for k in range(N - 2, 0, -1):
                    gru_unit.state = state_arr[k, j, i + 1] + state_arr[k, j - 1, i] + state_arr[k + 1, j, i]
                    c = gru_unit.forward(np.array([[
                        arr_1D[_Funcs.IND(i + 1, j, k, N)], arr_1D[_Funcs.IND(i, j - 1, k, N)], arr_1D[_Funcs.IND(i, j, k + 1, N)]
                    ]]))
                    new_arr[_Funcs.IND(i - 1, j - 1, k - 1, N - 2)] += c
                    state_arr[k, j, i] = c
        
        state_arr = np.zeros((N, N, N, 3))
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(N - 2, 0, -1):
                    gru_unit.state = state_arr[k, j, i - 1] + state_arr[k, j - 1, i] + state_arr[k + 1, j, i]
                    c = gru_unit.forward(np.array([[
                        arr_1D[_Funcs.IND(i - 1, j, k, N)], arr_1D[_Funcs.IND(i, j - 1, k, N)], arr_1D[_Funcs.IND(i, j, k + 1, N)]
                    ]]))
                    new_arr[_Funcs.IND(i - 1, j - 1, k - 1, N - 2)] += c
                    state_arr[k, j, i] = c
        state_arr = np.zeros((N, N, N, 3))
        for i in range(N - 2, 0, -1):
            for j in range(N - 2, 0, -1):
                for k in range(1, N - 1):
                    gru_unit.state = state_arr[k, j, i + 1] + state_arr[k, j + 1, i] + state_arr[k - 1, j, i]
                    c = gru_unit.forward(np.array([[
                        arr_1D[_Funcs.IND(i + 1, j, k, N)], arr_1D[_Funcs.IND(i, j + 1, k, N)], arr_1D[_Funcs.IND(i, j, k - 1, N)]
                    ]]))
                    new_arr[_Funcs.IND(i - 1, j - 1, k - 1, N - 2)] += c
                    state_arr[k, j, i] = c

        return np.array(new_arr)
        

class Genetic:
    @staticmethod
    def mutate_GRU(obj: GRU_unit, a = 0.1):
        ind = int(random() * 9)
        obj.weights[0, ind] += a * (random() * 2 - 1)
    
    @staticmethod
    def mutate_Simple_Layer(obj: Simple_layer, a = 0.1):
        ind_x = int(random() * obj.input_size + 1)
        ind_y = int(random() * obj.output_size)
        obj.weights[ind_x, ind_y] += a * (random() * 2 - 1)

    @staticmethod
    def cross_GRU(obj1: GRU_unit, obj2: GRU_unit, a = 0.5):
        new_weights = np.zeros((1, 9))
        for i in range(9):
            if random() > a:
                new_weights[0, i] = obj2.weights[0, i]
            else:
                new_weights[0, i] = obj1.weights[0, i]
        n = GRU_unit()
        n.weights = new_weights
        return n
    
    @staticmethod
    def cross_Simple_Layer(obj1: Simple_layer, obj2: Simple_layer, a = 0.5):
        if obj1.input_size != obj2.input_size or obj1.output_size != obj2.output_size: raise ValueError("Size must be similar")

        new_weights = np.zeros((obj1.input_size + 1, obj1.output_size))
        for i in range(obj1.input_size):
            for j in range(obj1.output_size):
                if random() > a:
                    new_weights[i, j] = obj2.weights[i, j]
                else:
                    new_weights[i, j] = obj1.weights[i, j]
        n = Simple_layer(input_size=obj1.input_size, output_size=obj1.output_size, activation_func=obj1.activation_func)
        n.weights = new_weights
        return n


class Model_Weights:
    def __init__(self, gru_u_weights, gru_v_weights, gru_w_weights,
                simple_relu_weights, simple_sigm_weights, simple_out_weights) -> None:
        self.gru_u_weights = gru_u_weights
        self.gru_v_weights = gru_v_weights
        self.gru_w_weights = gru_w_weights
        self.simple_relu_weights = simple_relu_weights
        self.simple_sigm_weights = simple_sigm_weights
        self.simple_out_weights = simple_out_weights

    def save(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.gru_u_weights)
            np.save(f, self.gru_v_weights)
            np.save(f, self.gru_w_weights)
            np.save(f, self.simple_relu_weights)
            np.save(f, self.simple_sigm_weights)
            np.save(f, self.simple_out_weights)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            gru_u_weights = np.load(f)
            gru_v_weights = np.load(f)
            gru_w_weights = np.load(f)
            simple_relu_weights = np.load(f)
            simple_sigm_weights = np.load(f)
            simple_out_weights = np.load(f)
            return Model_Weights(gru_u_weights, gru_v_weights, gru_w_weights, simple_relu_weights, simple_sigm_weights, simple_out_weights)


class Model:
    def __init__(self) -> None:
        self.gru_unit_u = GRU_unit()
        self.gru_unit_v = GRU_unit()
        self.gru_unit_w = GRU_unit()

        self.simple_relu = Simple_layer(input_size=9, output_size=5, activation_func="relu")
        self.simple_sigmoid = Simple_layer(input_size=5, output_size=3, activation_func="sigm")
        self.simple_out = Simple_layer(input_size=3, output_size=1, activation_func="sigm")

    def get_weights(self):
        return Model_Weights(
                    self.gru_unit_u.weights,
                    self.gru_unit_v.weights,
                    self.gru_unit_w.weights,
                    self.simple_relu.weights,
                    self.simple_sigmoid.weights,
                    self.simple_out.weights)

    def compute(self, V_x_1D, V_y_1D, V_z_1D, N, x, y, z, a, b, c):
        V_x = MD_GRU.compute(V_x_1D, N, self.gru_unit_u)
        V_y = MD_GRU.compute(V_y_1D, N, self.gru_unit_v)
        V_z = MD_GRU.compute(V_z_1D, N, self.gru_unit_w)

        res = [0 for i in range(N**3)]
        for i in range(x, x + a):
            for j in range(y, y + b):
                for k in range(z, z + c):
                    inp = np.append(np.append(np.array(V_x[_Funcs.IND(i, j, k, N - 2)]), V_y[_Funcs.IND(i, j, k, N - 2)]), V_z[_Funcs.IND(i, j, k, N - 2)])
                    fst = self.simple_relu.forward(inp)
                    sec = self.simple_sigmoid.forward(fst)
                    res[_Funcs.IND(i, j, k, N)] = 1 if self.simple_out.forward(sec) >= 0.5 else 0
        return res