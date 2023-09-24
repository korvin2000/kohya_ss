import abc
class AttentionStore :

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def get_empty_store(self):
        return {}

    def store(self, attn, layer_name):
        if layer_name not in self.step_store.keys() :
            self.step_store[layer_name] = []
        self.step_store[layer_name].append(attn)
        return attn

    def reset(self):
        self.step_store = self.get_empty_store()
        self.attention_store = {}
