from random import random, choice
import numpy as np

class _Funcs:
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def dlinear(x):
        return np.ones_like(x)
    
    @staticmethod
    def relu(x):
        return x * (x > 0)
    
    @staticmethod
    def drelu(x):
        return 1 * (x > 0)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def dsigmoid(x):
        return _Funcs.sigmoid(x) * (1 - _Funcs.sigmoid(x))

    @staticmethod
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    @staticmethod
    def dtanh(x):
        return 1 - _Funcs.tanh(x)**2
    
    @staticmethod
    def softmax(x):
        e_x = np.exp(x)
        return e_x / np.sum(e_x, axis = 1, keepdims = True)

    @staticmethod
    def dsoftmax(x):
        return _Funcs.softmax(x) * (1 - _Funcs.softmax(x))
    
    @staticmethod
    def IND(x, y, z, N):
        return x + y * N + z * N * N
    
    @staticmethod
    def norm_velocities(V_x: np.ndarray, V_y: np.ndarray, V_z: np.ndarray):
        ma = max(np.max(V_x), np.max(V_y), np.max(V_z))
        mi = min(np.min(V_x), np.min(V_y), np.min(V_z))
        d = ma - mi
        if d != 0:
            return [
                (V_x - mi) / d,
                (V_y - mi) / d,
                (V_z - mi) / d
            ]
        else:
            return [np.ones_like(V_x), np.ones_like(V_y), np.ones_like(V_z)]
    
    @staticmethod
    def MSE(x, y):
        return np.sum((y - x)**2) / x.shape[-1]
    
    @staticmethod
    def dMSE(x, y):
        return (-2 * (y - x)) / x.shape[-1]


class Simple_layer:
    # input_shape = (1, N1 + bias); output_shape = (1, N2); weights_shape = (N1 + bias, N2);
    def __init__(self, input_size = None, output_size = None, activation_func = "sigm") -> None:
        if not (input_size and output_size): raise ValueError("input_size and output_size must be provided")

        self.input_size = input_size
        self.output_size = output_size
        self.activation_func = activation_func
        self.inp = np.zeros((1, input_size))
        
        self.m_weights = np.zeros((input_size, output_size))
        self.v_weights = np.zeros((input_size, output_size))

        self.m_bias = np.zeros((1, output_size))
        self.v_bias = np.zeros((1, output_size))

        self.weights = (np.random.random((input_size, output_size)) * 2 - 1) * 0.01
        self.bias = (np.random.random((1, output_size)) * 2 - 1) * 0.01

    def forward(self, inp):
        # pass input forward throw weighting and activation
        self.inp = inp
        self.sum_x = np.dot(self.inp, self.weights) + self.bias
        if self.activation_func == "sigm":
            x_active = _Funcs.sigmoid(self.sum_x)
        if self.activation_func == "softmax":
            x_active = _Funcs.softmax(self.sum_x)
        if self.activation_func == "relu":
            x_active = _Funcs.relu(self.sum_x)
        if self.activation_func == "tanh":
            x_active = _Funcs.tanh(self.sum_x)
        if self.activation_func == "linear":
            x_active = _Funcs.linear(self.sum_x)
        return x_active
    
    def backward(self, dl, t, a = 0.01, beta_1 = 0.998, beta_2 = 0.7):
        if self.activation_func == "sigm":
            d_act_x = dl * _Funcs.dsigmoid(self.sum_x)
        if self.activation_func == "softmax":
            d_act_x = dl * _Funcs.dsoftmax(self.sum_x)
        if self.activation_func == "relu":
            d_act_x = dl * _Funcs.drelu(self.sum_x)
        if self.activation_func == "tanh":
            d_act_x = dl * _Funcs.dtanh(self.sum_x)
        if self.activation_func == "linear":
            d_act_x = dl * _Funcs.dlinear(self.sum_x)
        d_weights = np.dot((self.inp.T), d_act_x) # compute weights gradient

        self.m_weights = beta_1 * self.m_weights + (1 - beta_1) * d_weights
        self.v_weights = beta_2 * self.v_weights + (1 - beta_2) * np.power(d_weights, 2)
        m_hat = self.m_weights / (1 - np.power(beta_1, t))
        v_hat = self.v_weights / (1 - np.power(beta_2, t))
        self.weights -= a * m_hat / (np.sqrt(v_hat) + 0.00001)

        self.m_bias = beta_1 * self.m_bias + (1 - beta_1) * d_act_x
        self.v_bias = beta_2 * self.v_bias + (1 - beta_2) * np.power(d_act_x, 2)
        m_hat = self.m_bias / (1 - np.power(beta_1, t))
        v_hat = self.v_bias / (1 - np.power(beta_2, t))
        self.bias -= a * m_hat / (np.sqrt(v_hat) + 0.00001)

        return np.dot(d_act_x, (self.weights.T)) # derivative of second layer


class GRU_unit:
    def __init__(self, init_state = np.zeros((1, 1))) -> None:
        self.history = []
        self.state = init_state
        self.weights = (np.random.random((1, 9)) * 2 - 1) * 0.01
    
    def forward(self, x):
        self.r = _Funcs.sigmoid(self.weights[0, 0] * x + self.weights[0, 3] * self.state + self.weights[0, 6])
        self.z = _Funcs.sigmoid(self.weights[0, 1] * x + self.weights[0, 4] * self.state + self.weights[0, 7])
        self.h = _Funcs.tanh(self.weights[0, 2] * x + self.weights[0, 5] * self.state * self.r + self.weights[0, 8])
        self.history.append({
            'prev_state': self.state,
            'z': self.z,
            'r': self.r,
            'h': self.h,
            'x': x
        })
        self.state = self.z * self.h + (1 - self.z) * self.state
        return self.state
    
    def backward(self, dl, a = 0.1):
        d0 = dl
        while len(self.history) > 0:
            last = self.history.pop()
            d1 = last["z"] * d0
            d2 = last["prev_state"] * d0
            d3 = last['h'] * d0
            d4 = -1 * d3
            d5 = d2 + d4
            d6 = (1 - last['z']) * d0
            d7 = d5 * (last['z'] * (1 - last['z']))
            d8 = d6 * (1 - last['h']**2)
            d9 = d8 * self.weights[0, 2]
            d10 = d8 * self.weights[0, 5]
            d11 = d7 * self.weights[0, 1]
            d12 = d7 * self.weights[0, 4]
            d14 = d10 * last['r']
            d15 = d10 * last['h']
            d16 = d15 * (last['r'] * (1 - last['r']))
            d13 = d16 * self.weights[0, 0]
            d17 = d16 * self.weights[0, 3]
            d0 = d12 + d14 + d1 + d17

            self.weights[0, 0] -= a * last['x'] * d16
            self.weights[0, 1] -= a * last['x'] * d7
            self.weights[0, 2] -= a * last['x'] * d8
            self.weights[0, 3] -= a * last['prev_state'] * d16
            self.weights[0, 4] -= a * last['prev_state'] * d7
            self.weights[0, 5] -= a * last['prev_state'] * last['r'] * d8
            self.weights[0, 6] -= a * d16
            self.weights[0, 7] -= a * d7
            self.weights[0, 8] -= a * d8


    def compute(self, arr_1D, x, y, z, w, d, h, N):
        self.state = np.zeros((1, 1))
        for k in range(z, z + h):
            for j in range(y, y + d):
                for i in range(x, x + w):
                    self.forward(arr_1D[_Funcs.IND(i, j, k, N)])
        
        return self.state


class Model_Weights:
    def __init__(self, gru_u_weights, gru_v_weights, gru_w_weights,
                layer_1_weights, layer_2_weights, layer_3_weights,
                layer_out_weights) -> None:
        self.gru_u_weights = gru_u_weights
        self.gru_v_weights = gru_v_weights
        self.gru_w_weights = gru_w_weights

        self.layer_1_weights = layer_1_weights
        self.layer_2_weights = layer_2_weights
        self.layer_3_weights = layer_3_weights
        self.layer_out_weights = layer_out_weights

    def save(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.gru_u_weights)
            np.save(f, self.gru_v_weights)
            np.save(f, self.gru_w_weights)

            np.save(f, self.layer_1_weights)
            np.save(f, self.layer_2_weights)
            np.save(f, self.layer_3_weights)
            np.save(f, self.layer_out_weights)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            gru_u_weights = np.load(f)
            gru_v_weights = np.load(f)
            gru_w_weights = np.load(f)

            layer_1_weights = np.load(f)
            layer_2_weights = np.load(f)
            layer_3_weights = np.load(f)
            layer_out_weights = np.load(f)

        return Model_Weights(gru_u_weights, gru_v_weights, gru_w_weights,
                            layer_1_weights, layer_2_weights,
                            layer_3_weights, layer_out_weights)

class Model:
    def __init__(self, C_Vn) -> None:
        self.gru_unit_u = GRU_unit()
        self.gru_unit_v = GRU_unit()
        self.gru_unit_w = GRU_unit()

        self.layer_1 = Simple_layer(input_size=6, output_size=64, activation_func="sigm")
        self.layer_2 = Simple_layer(input_size=64, output_size=64, activation_func="relu")
        self.layer_3 = Simple_layer(input_size=64, output_size=32, activation_func='tanh')
        self.layer_out = Simple_layer(input_size=32, output_size=3 * C_Vn, activation_func='linear')

    def set_weights(self, model: Model_Weights):
        self.gru_unit_u.weights = model.gru_u_weights
        self.gru_unit_v.weights = model.gru_v_weights
        self.gru_unit_w.weights = model.gru_w_weights

        self.layer_1.weights = model.layer_1_weights
        self.layer_2.weights = model.layer_2_weights
        self.layer_3.weights = model.layer_3_weights
        self.layer_out.weights = model.layer_out_weights

    def get_weights(self):
        return Model_Weights(
                    self.gru_unit_u.weights,
                    self.gru_unit_v.weights,
                    self.gru_unit_w.weights,
                    self.layer_1.weights,
                    self.layer_2.weights,
                    self.layer_3.weights,
                    self.layer_out.weights)

    def compute(self, V_x_1D, V_y_1D, V_z_1D, x, y, z, w, d, h, t, N):
        V_x = self.gru_unit_u.compute(V_x_1D, x, y, z, w, d, h, N)
        V_y = self.gru_unit_v.compute(V_y_1D, x, y, z, w, d, h, N)
        V_z = self.gru_unit_w.compute(V_z_1D, x, y, z, w, d, h, N)

        inp = np.array([[
            V_x[0, 0],
            V_y[0, 0],
            V_z[0, 0],
            (t + 1) / d,
            w,
            h
        ]])

        fst = self.layer_1.forward(inp)
        sec = self.layer_2.forward(fst)
        thr = self.layer_3.forward(sec)
        out = self.layer_out.forward(thr)
        
        return out

class Genetic:
    @staticmethod
    def mutate(obj: Model, a = 0.1):
        ch = choice([
            obj.gru_unit_u, obj.gru_unit_v, obj.gru_unit_w,
            obj.layer_1, obj.layer_2, obj.layer_3, obj.layer_out
        ])
        ind_row = int(random() * ch.weights.shape[0])
        ind_col = int(random() * ch.weights.shape[1])

        ch.weights[ind_row, ind_col] += a * (random() * 2 - 1)

    @staticmethod
    def cross(layer_1, layer_2, distr_index):
        u = random()
        if u < 0.5:
            B = pow(2 * u, 1 / (distr_index + 1))
        else:
            B = pow(1 / 2 * (1 - u), 1 / (distr_index + 1))
        
        return [
            0.5 * (1 + B) * layer_1.weights + (1 - B) * layer_2.weights,
            0.5 * (1 - B) * layer_1.weights + (1 + B) * layer_2.weights
        ]
    
    @staticmethod
    def update_population(POOLING_SIZE, POPULATION_SIZE, population, CROSS_DISTRIBUTION_INDEX,
                            MUTATION_PROBABILITY, MUTATION_C):
        for i in range(POOLING_SIZE, POPULATION_SIZE - 1, 2):
            parent_1 = choice([j for j in range(POOLING_SIZE)])
            parent_2 = choice([j for j in range(POOLING_SIZE) if j != parent_1])

            population[i][0].gru_unit_u.weights, population[i + 1][0].gru_unit_u.weights = Genetic.cross(population[parent_1][0].gru_unit_u, population[parent_2][0].gru_unit_u, CROSS_DISTRIBUTION_INDEX)
            population[i][0].gru_unit_v.weights, population[i + 1][0].gru_unit_v.weights = Genetic.cross(population[parent_1][0].gru_unit_v, population[parent_2][0].gru_unit_v, CROSS_DISTRIBUTION_INDEX)
            population[i][0].gru_unit_w.weights, population[i + 1][0].gru_unit_w.weights = Genetic.cross(population[parent_1][0].gru_unit_w, population[parent_2][0].gru_unit_w, CROSS_DISTRIBUTION_INDEX)
            population[i][0].layer_1.weights, population[i + 1][0].layer_1.weights = Genetic.cross(population[parent_1][0].layer_1, population[parent_2][0].layer_1, CROSS_DISTRIBUTION_INDEX)
            population[i][0].layer_2.weights, population[i + 1][0].layer_2.weights = Genetic.cross(population[parent_1][0].layer_2, population[parent_2][0].layer_2, CROSS_DISTRIBUTION_INDEX)
            population[i][0].layer_3.weights, population[i + 1][0].layer_3.weights = Genetic.cross(population[parent_1][0].layer_3, population[parent_2][0].layer_3, CROSS_DISTRIBUTION_INDEX)
            population[i][0].layer_out.weights, population[i + 1][0].layer_out.weights = Genetic.cross(population[parent_1][0].layer_out, population[parent_2][0].layer_out, CROSS_DISTRIBUTION_INDEX)

            if random() < MUTATION_PROBABILITY:
                Genetic.mutate(population[i][0], a = MUTATION_C)
            if random() < MUTATION_PROBABILITY:
                Genetic.mutate(population[i + 1][0], a = MUTATION_C)

def backpropagation(obj, out, y, t, a = 0.1, beta_1 = 0.01, beta_2 = 0.7):
    print('Error -', _Funcs.MSE(out, np.array(y)))
    dl = _Funcs.dMSE(out, np.array(y))

    dl1 = obj[0].layer_out.backward(dl, t, a, beta_1 = beta_1, beta_2 = beta_2)
    dl2 = obj[0].layer_3.backward(dl1, t, a, beta_1 = beta_1, beta_2 = beta_2)
    dl3 = obj[0].layer_2.backward(dl2, t, a, beta_1 = beta_1, beta_2 = beta_2)
    dl4 = obj[0].layer_1.backward(dl3, t, a, beta_1 = beta_1, beta_2 = beta_2)

    obj[0].gru_unit_u.backward(dl4[0, 0], a)
    obj[0].gru_unit_v.backward(dl4[0, 1], a)
    obj[0].gru_unit_w.backward(dl4[0, 2], a)