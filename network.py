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

        self.weights = (np.random.random((input_size, output_size)) * 2 - 1) * 0.0001
        self.bias = (np.random.random((1, output_size)) * 2 - 1) * 0.0001

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
    
    def backward(self, dl, t, a = 0.01):
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

        self.weights -= a * d_weights
        self.bias -= a * d_act_x

        return np.dot(d_act_x, (self.weights.T)) # derivative of second layer


class Model_Weights:
    def __init__(self, layer_1_weights, layer_2_weights, layer_3_weights, layer_4_weights, layer_out_weights) -> None:

        self.layer_1_weights = layer_1_weights
        self.layer_2_weights = layer_2_weights
        self.layer_3_weights = layer_3_weights
        self.layer_4_weights = layer_4_weights
        self.layer_out_weights = layer_out_weights

    def save(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.layer_1_weights)
            np.save(f, self.layer_2_weights)
            np.save(f, self.layer_3_weights)
            np.save(f, self.layer_4_weights)
            np.save(f, self.layer_out_weights)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            layer_1_weights = np.load(f)
            layer_2_weights = np.load(f)
            layer_3_weights = np.load(f)
            layer_4_weights = np.load(f)
            layer_out_weights = np.load(f)

        return Model_Weights(layer_1_weights, layer_2_weights, layer_3_weights, layer_4_weights, layer_out_weights)

class Model:
    def __init__(self, arr_shape, Cn, d) -> None:
        self.layer_1 = Simple_layer(input_size=arr_shape * 3, output_size=128, activation_func='relu')
        self.layer_2 = Simple_layer(input_size=128, output_size=128, activation_func='sigm')
        self.layer_3 = Simple_layer(input_size=128, output_size=128, activation_func='relu')
        self.layer_4 = Simple_layer(input_size=128, output_size=32, activation_func='sigm')
        self.layer_out = Simple_layer(input_size=32, output_size=(Cn * 4 + 2) * d, activation_func='linear')

    def set_weights(self, model: Model_Weights):
        self.layer_1.weights = model.layer_1_weights
        self.layer_2.weights = model.layer_2_weights
        self.layer_3.weights = model.layer_3_weights
        self.layer_4.weights = model.layer_4_weights
        self.layer_out.weights = model.layer_out_weights

    def get_weights(self):
        return Model_Weights(
                    self.layer_1.weights,
                    self.layer_2.weights,
                    self.layer_3.weights,
                    self.layer_4.weights,
                    self.layer_out.weights)

    def compute(self, V_x_1D, V_y_1D, V_z_1D, x, y, z, w, d, h, N):
        V_n_x, V_n_y, V_n_z = [], [], []
        for k in range(z, z + h):
            for j in range(y, y + d):
                for i in range(x, x + w):
                    V_n_x.append(V_x_1D[_Funcs.IND(i, j, k, N)])
                    V_n_y.append(V_y_1D[_Funcs.IND(i, j, k, N)])
                    V_n_z.append(V_z_1D[_Funcs.IND(i, j, k, N)])

        V_n_x, V_n_y, V_n_z = _Funcs.norm_velocities(V_n_x, V_n_y, V_n_z)

        inp = np.array([np.append(np.append(V_n_x, V_n_y), V_n_z)])

        l_1 = self.layer_1.forward(inp)
        l_2 = self.layer_2.forward(l_1)
        l_3 = self.layer_3.forward(l_2)
        l_4 = self.layer_4.forward(l_3)
        out = self.layer_out.forward(l_4)
        
        return out

class Genetic:
    @staticmethod
    def mutate(obj: Model, a = 0.1):
        ch = choice([
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

            population[i][0].layer_1.weights, population[i + 1][0].layer_1.weights = Genetic.cross(population[parent_1][0].layer_1, population[parent_2][0].layer_1, CROSS_DISTRIBUTION_INDEX)
            population[i][0].layer_2.weights, population[i + 1][0].layer_2.weights = Genetic.cross(population[parent_1][0].layer_2, population[parent_2][0].layer_2, CROSS_DISTRIBUTION_INDEX)
            population[i][0].layer_3.weights, population[i + 1][0].layer_3.weights = Genetic.cross(population[parent_1][0].layer_3, population[parent_2][0].layer_3, CROSS_DISTRIBUTION_INDEX)
            population[i][0].layer_4.weights, population[i + 1][0].layer_4.weights = Genetic.cross(population[parent_1][0].layer_4, population[parent_2][0].layer_4, CROSS_DISTRIBUTION_INDEX)
            population[i][0].layer_out.weights, population[i + 1][0].layer_out.weights = Genetic.cross(population[parent_1][0].layer_out, population[parent_2][0].layer_out, CROSS_DISTRIBUTION_INDEX)

            if random() < MUTATION_PROBABILITY:
                Genetic.mutate(population[i][0], a = MUTATION_C)
            if random() < MUTATION_PROBABILITY:
                Genetic.mutate(population[i + 1][0], a = MUTATION_C)

def backpropagation(obj, out, y, t, a = 0.1):
    print('Error -', _Funcs.MSE(out, np.array(y)))
    dl = _Funcs.dMSE(out, np.array(y))

    dl1 = obj[0].layer_out.backward(dl, t, a)
    dl2 = obj[0].layer_4.backward(dl1, t, a)
    dl3 = obj[0].layer_3.backward(dl2, t, a)
    dl4 = obj[0].layer_2.backward(dl3, t, a)
    dl5 = obj[0].layer_1.backward(dl4, t, a)