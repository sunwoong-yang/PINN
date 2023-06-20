import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_layers=[100], activation='tanh'):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = nn.ModuleList()

        self.hidden_layers.append(nn.Linear(input_dim, hidden_layers[0]))

        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        self.final_layer = nn.Linear(hidden_layers[-1], output_dim)

        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'SiLU':
            self.activation = nn.SiLU()
        else:
            assert ('Invalid activation!'
                    'Options: ReLU, LeakyReLU, Tanh, Sigmoid, GELU, SiLU')

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = self.activation(x)
        x = self.final_layer(x)
        return x
