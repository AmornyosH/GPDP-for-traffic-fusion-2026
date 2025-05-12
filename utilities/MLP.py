import torch

class MLP(torch.nn.Module):
    def __init__(self, num_hiddens:int, num_nodes:int, input_dim:int, output_dim:int):
        super(MLP, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_nodes = num_nodes
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(self.num_hiddens):
            _input_nodes = num_nodes if i > 0 else input_dim
            _output_nodes = num_nodes if i < (self.num_hiddens-1) else output_dim
            self.hidden_layers.append(torch.nn.Linear(_input_nodes, _output_nodes))
    def forward(self, input):
        for j in range(self.num_hiddens):
            output = torch.nn.functional.mish(self.hidden_layers[j](input if j==0 else output))
        return output