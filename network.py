from random import random, choice
import torch
import torch.nn as nn

class _Funcs:
    @staticmethod
    def IND(x, y, z, N):
        return x + y * N + z * N * N

class AeroNetwork(nn.Module):
    def __init__(self, arr_shape, Cn, d) -> None:
        super(AeroNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(arr_shape * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, (Cn * 4 + 2) * d)
        )
    
    def forward(self, x):
        return self.net(x)
    
    @staticmethod
    def prepare(V_x_1D, V_y_1D, V_z_1D, x, y, z, w, d, h, N):
        V_n = torch.tensor([
            [],
            [],
            [],
        ])
        for k in range(z, z + h):
            for j in range(y, y + d):
                for i in range(x, x + w):
                    V_n = torch.cat((V_n, torch.tensor([
                                [V_x_1D[_Funcs.IND(i, j, k, N)], ],
                                [V_y_1D[_Funcs.IND(i, j, k, N)], ],
                                [V_z_1D[_Funcs.IND(i, j, k, N)], ],
                            ])), dim=1)
        return nn.functional.normalize(V_n)


class Genetic:
    @staticmethod
    def mutate(model : AeroNetwork, MUTATION_PROBABILITY = 0.1, a = 0.1):
        state_dict = model.state_dict()

        for name, param in state_dict.items():
            if random() < MUTATION_PROBABILITY:
                transformed_param = param
                
                if len(param.shape) == 2:
                    transformed_param[int(random() * param.shape[0])][int(random() * param.shape[1])] *= a
                elif len(param.shape) == 1:
                    transformed_param[int(random() * param.shape[0])] *= a

                param.copy_(transformed_param)

    @staticmethod
    def cross(p1: AeroNetwork, p2: AeroNetwork, ch1: AeroNetwork, ch2: AeroNetwork, distr_index):
        u = random()
        if u < 0.5:
            B = pow(2 * u, 1 / (distr_index + 1))
        else:
            B = pow(1 / 2 * (1 - u), 1 / (distr_index + 1))
        
        for dp1, dp2, dch1, dch2 in zip(p1.state_dict().items(), p2.state_dict().items(), ch1.state_dict().items(), ch2.state_dict().items()):
            transformed_param1, transformed_param2 = 0.5 * (1 + B) * dp1[1] + (1 - B) * dp2[1], 0.5 * (1 - B) * dp1[1] + (1 + B) * dp2[1]
            dch1[1].copy_(transformed_param1)
            dch2[1].copy_(transformed_param2)
    
    @staticmethod
    def update_population(POOLING_SIZE, POPULATION_SIZE, population, CROSS_DISTRIBUTION_INDEX,
                            MUTATION_PROBABILITY, MUTATION_C):
        with torch.no_grad():
            for i in range(POOLING_SIZE, POPULATION_SIZE - 1, 2):
                parent_1 = choice([j for j in range(POOLING_SIZE)])
                parent_2 = choice([j for j in range(POOLING_SIZE) if j != parent_1])

                Genetic.cross(population[parent_1][0], population[parent_2][0], population[i][0], population[i + 1][0], CROSS_DISTRIBUTION_INDEX)

                
                Genetic.mutate(population[i][0], MUTATION_PROBABILITY=MUTATION_PROBABILITY, a = MUTATION_C)
                Genetic.mutate(population[i + 1][0], MUTATION_PROBABILITY=MUTATION_PROBABILITY, a = MUTATION_C)