from torch import nn
import torch


class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        pass


class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.device = device

    def update_memory(self, unique_node_ids, unique_messages, type, timestamps=None):
        if len(unique_node_ids) <= 0:
            return
        if timestamps is not None:
            assert (self.memory.get_last_update(type, unique_node_ids) <= timestamps[unique_node_ids]).all().item(), "Trying to " \
                                                                                              "update memory to time in the past"
            self.memory.last_update[type][unique_node_ids] = timestamps[unique_node_ids].float()

        self.memory.set_memory(type, unique_node_ids, unique_messages[unique_node_ids])

    def get_updated_memory(self, unique_node_ids, unique_messages, type, timestamps=None):
        if len(unique_node_ids) <= 0:
            return self.memory.memory[type].data.clone(), self.memory.last_update[type].data.clone()

        assert (self.memory.get_last_update(type, unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                                "update memory to time in the past"
        updated_memory = self.memory.memory[type].data.clone()
        updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

        updated_last_update = self.memory.last_update[type].data.clone()
        updated_last_update[unique_node_ids] = timestamps.float()

        return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

        self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

        self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
    if module_type == "gru":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
