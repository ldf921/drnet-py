import torch
from utils import utils

class Model:
    def named_modules(self):
        return [ (name, getattr(self, name)) for name in self._modules]

    def modules(self):
        return map(lambda name : getattr(self, name), self._modules)

    def __iter__(self):
        return self.modules()

    def build_optimizer(self):
        for name, module in self.named_modules():
            optim_name = name.replace('net', 'optimizer')
            setattr(self, optim_name, utils.get_optimizer(self.opt, module))

    def save(self, cp_path):
        torch.save(dict(self.named_modules()), cp_path)
