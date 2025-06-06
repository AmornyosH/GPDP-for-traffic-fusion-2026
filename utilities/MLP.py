import torch

class MLP(torch.nn.Module):
    def __init__(self, num_hiddens:int, num_nodes:int, input_dim:int, output_dim:int):
        super(MLP, self).__init__()
        # self.num_hiddens = num_hiddens
        # self.num_nodes = num_nodes
        # self.hidden_layers = torch.nn.ModuleList()
        # for i in range(self.num_hiddens):
        #     _input_nodes = num_nodes if i > 0 else input_dim
        #     _output_nodes = num_nodes if i < (self.num_hiddens-1) else output_dim
        #     self.hidden_layers.append(torch.nn.Linear(_input_nodes, _output_nodes))
        
        self.fc1 = torch.nn.Linear(input_dim, num_nodes)
        self.fc2 = torch.nn.Linear(num_nodes, num_nodes)
        self.fc3 = torch.nn.Linear(num_nodes, num_nodes)
        self.output = torch.nn.Linear(num_nodes, output_dim)

    def forward(self, input):
        # for j in range(self.num_hiddens):
        #     output = torch.nn.functional.mish(self.hidden_layers[j](input if j==0 else output))
        # return output
        act1 = torch.nn.functional.mish(self.fc1(input))
        act2 = torch.nn.functional.mish(self.fc2(act1))
        act3 = torch.nn.functional.mish(self.fc3(act2))
        # act1 = torch.nn.functional.relu(self.fc1(input))
        # act2 = torch.nn.functional.relu(self.fc2(act1))
        # act3 = torch.nn.functional.relu(self.fc3(act2))
        output = self.output(act3)
        return output