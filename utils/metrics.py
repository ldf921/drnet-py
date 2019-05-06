from collections import OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Summary:
    """
    Automatically keeps a dictionary of named metric values
    """
    def __init__(self):
        self.dict = OrderedDict()

    def update(self, metrics, n=1):
        for k, v in metrics.items():
            meter = self.dict.setdefault(k, AverageMeter())  # set a new meter if no meter found
            meter.update(v.item(), n=n)

    def format(self):
        msg = ''
        for k, v in self.dict.items():
            msg += '| {}: {:.4f} '.format(k.replace('_', ' '), v.avg)
        return msg
