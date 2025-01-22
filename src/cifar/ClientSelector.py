import numpy as np
import matplotlib.pyplot as plt

class ClientSelector:
    def __init__(self, config):
        self.num_clients = config['K']
        self.sample_size = int(max(config['C'] * self.num_clients, 1))
        self.client_indices = np.arange(self.num_clients)

        if config['participation'] == 'uniform':
            self.probabilities = None
        else:
            self.probabilities = np.random.dirichlet(
                alpha=np.full(self.num_clients, 1/config['gamma'])
            )

    def select(self):
        return np.random.choice(
            self.client_indices,
            size=self.sample_size,
            p=self.probabilities,
            replace=False
        )