import torch


class InputNormCollector:
    def __init__(self, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.sum_sq = None
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        if inp is None:
            return
        x = inp.detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            pass
        else:
            x = x.reshape(-1, x.shape[-1])

        x = x.to(device=self.device, dtype=self.dtype)
        cur = x.pow(2).sum(dim=0)
        n = x.shape[0]

        if self.sum_sq is None:
            self.sum_sq = cur
            self.nsamples = n
        else:
            self.sum_sq += cur
            self.nsamples += n

    def mean_l2_sq(self, eps: float = 1e-6):
        if self.sum_sq is None or self.nsamples <= 0:
            return None
        return self.sum_sq / max(1, self.nsamples)

    def rms(self, eps: float = 1e-6):
        val = self.mean_l2_sq(eps=eps)
        if val is None:
            return None
        return torch.sqrt(val.clamp_min(eps))


class InputStatCollector:
    def __init__(self, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.sum_ = None
        self.sum_sq = None
        self.count = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        if inp is None:
            return
        x = inp.detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            pass
        else:
            x = x.reshape(-1, x.shape[-1])

        x = x.to(device=self.device, dtype=self.dtype)
        cur_sum = x.sum(dim=0)
        cur_sq = x.pow(2).sum(dim=0)
        n = x.shape[0]

        if self.sum_ is None:
            self.sum_ = cur_sum
            self.sum_sq = cur_sq
            self.count = n
        else:
            self.sum_ += cur_sum
            self.sum_sq += cur_sq
            self.count += n

    def mean(self):
        if self.sum_ is None or self.count <= 0:
            return None
        return self.sum_ / max(1, self.count)

    def var(self, eps: float = 1e-6):
        if self.sum_ is None or self.count <= 0:
            return None
        mean = self.mean()
        ex2 = self.sum_sq / max(1, self.count)
        var = ex2 - mean.pow(2)
        return var.clamp_min(0.0)

    def std(self, eps: float = 1e-6):
        val = self.var(eps=eps)
        if val is None:
            return None
        return torch.sqrt(val.clamp_min(eps))
