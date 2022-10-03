import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

    def __init__(self, type, n_nodes, memory_dimension, message_dimension=None,
                 device="cpu", combination_method='sum'):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension

        self.message_dimension = message_dimension
        self.device = device

        self.type = type

        self.combination_method = combination_method

        self.__init_memory__()

    def __init_memory__(self):
        """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
        # Treat memory as parameter so that it is saved and loaded together with the model
        self.memory = nn.ParameterDict(
            {t: nn.Parameter(torch.zeros((self.n_nodes[t], self.memory_dimension)).to(self.device),
                             requires_grad=False) for t in self.type})
        self.last_update = nn.ParameterDict({t: nn.Parameter(torch.zeros(self.n_nodes[t]).to(self.device),
                                                             requires_grad=False) for t in self.type})


    def get_memory(self, type, node_idxs):
        return self.memory[type][node_idxs, :]

    def set_memory(self, type, node_idxs, values):
        # print("previous memory type:{}, node{}: \n{}".format(type, node_idxs, self.memory[type][node_idxs,:]))
        self.memory[type][node_idxs, :] = values
        # print("now memory{}".format(self.memory[type][node_idxs,:]))

    def get_last_update(self, type, node_idxs):
        return self.last_update[type][node_idxs]

    def backup_memory(self):
        memory = {t: self.memory[t].data.clone() for t in self.type}
        last_up = {t: self.last_update[t].data.clone() for t in self.type}
        return memory, last_up

    def restore_memory(self, memory_backup):
        for t in self.type:
            self.memory[t].data, self.last_update[t].data = memory_backup[0][t].clone(), memory_backup[1][t].clone()
            # self.memory[t].data = memory_backup[t].clone()


    def detach_memory(self):
        for t in self.type:
            self.memory[t].detach_()

