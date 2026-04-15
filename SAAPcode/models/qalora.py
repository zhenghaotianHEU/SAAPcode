import torch
import torch.nn as nn
import torch.nn.functional as F


class QALoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.05, group_size: int = 32):
        super().__init__()
        if not hasattr(base_layer, 'weight'):
            raise ValueError('QALoRALinear requires a linear-like layer with weight')
        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.r = int(r)
        self.alpha = int(alpha)
        self.group_size = max(1, int(group_size))
        self.scaling = self.alpha / max(1, self.r)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.reduced_in_features = self.in_features // self.group_size
        if self.reduced_in_features <= 0 or self.in_features % self.group_size != 0:
            raise ValueError(
                f'QALoRALinear requires in_features divisible by group_size, got in_features={self.in_features}, group_size={self.group_size}'
            )

        self.lora_A = nn.Linear(self.reduced_in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, self.out_features, bias=False)
        nn.init.xavier_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

        if hasattr(self.base_layer, 'weight') and self.base_layer.weight is not None:
            self.base_layer.weight.requires_grad = False
        if hasattr(self.base_layer, 'bias') and self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x):
        result = self.base_layer(x)
        previous_dtype = result.dtype
        target_dtype = self.lora_A.weight.dtype

        x_d = self.dropout(x)
        if x_d.dtype != target_dtype:
            x_d = x_d.to(dtype=target_dtype)

        orig_shape = x_d.shape[:-1]
        pooled = x_d.reshape(-1, self.reduced_in_features, self.group_size).mean(dim=-1)
        adapter = self.lora_B(self.lora_A(pooled))
        adapter.mul_(self.scaling)
        adapter = adapter.reshape(*orig_shape, self.out_features)
        if adapter.dtype != previous_dtype:
            adapter = adapter.to(dtype=previous_dtype)
        return result + adapter


def replace_with_qalora(model: nn.Module, target_suffixes, r: int = 8, alpha: int = 16, dropout: float = 0.05, group_size: int = 32, logger=print):
    replaced = []
    for module_name, module in list(model.named_modules()):
        if not any(module_name.endswith(sfx) for sfx in target_suffixes):
            continue
        if not hasattr(module, 'in_features') or not hasattr(module, 'out_features'):
            continue
        if int(module.in_features) % int(group_size) != 0:
            logger(f'[qalora] skip module={module_name} | in_features={module.in_features} not divisible by group_size={group_size}')
            continue
        parent = model
        parts = module_name.split('.')
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]
        wrapped = QALoRALinear(module, r=r, alpha=alpha, dropout=dropout, group_size=group_size)
        setattr(parent, child_name, wrapped)
        replaced.append(module_name)
    logger(f'[qalora] replaced modules count={len(replaced)} | modules={replaced[:12]}{" ..." if len(replaced) > 12 else ""}')
    return replaced
