import torch
from utils import utils


class Model(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._modules = []
        self._criterions = []

    def named_modules(self):
        return [(name, getattr(self, name)) for name in self._modules]

    def modules(self):
        return map(lambda name: getattr(self, name), self._modules)

    def named_criterions(self):
        return [(name, getattr(self, name)) for name in self._criterions]

    def criterions(self):
        return map(lambda name: getattr(self, name), self._criterions)

    def cuda(self):
        for module in self.modules():
            module.cuda()
        for criterion in self.criterions():
            criterion.cuda()

    def set_all_train(self):
        for module in self.modules():
            module.train()

    def set_all_eval(self):
        for module in self.modules():
            module.eval()

    def build_optimizer(self):
        for name, module in self.named_modules():
            optim_name = name.replace('net', 'optimizer')
            setattr(self, optim_name, utils.get_optimizer(self.opt, module))

    def save(self, cp_path):
        torch.save(dict(self.named_modules()), cp_path)
