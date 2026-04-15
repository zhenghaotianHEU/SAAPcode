import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.cpp_ops.pruned_int4_loader import load_pruned_int4_ext
from core.models.pruned_int4_triton import TritonPrunedInt4Linear


def quantize_tensor_int4_symmetric(weight: torch.Tensor, group_size: int = 64):
    w = weight.detach().to(dtype=torch.float32, device='cpu').contiguous()
    out_features, in_features = w.shape
    if in_features % group_size != 0:
        pad = group_size - (in_features % group_size)
        w = F.pad(w, (0, pad))
        in_features_padded = in_features + pad
    else:
        pad = 0
        in_features_padded = in_features
    num_groups = in_features_padded // group_size
    wg = w.view(out_features, num_groups, group_size)
    scales = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
    q = torch.clamp(torch.round(wg / scales), -8, 7).to(torch.int8)
    return q, scales.squeeze(-1).to(torch.float16), in_features, in_features_padded, pad, group_size


def dequantize_tensor_int4_symmetric(qweight: torch.Tensor, scales: torch.Tensor, in_features: int, in_features_padded: int, group_size: int):
    w = qweight.to(torch.float16) * scales.unsqueeze(-1).to(torch.float16)
    w = w.view(qweight.shape[0], in_features_padded)
    return w[:, :in_features].contiguous()


class PrunedInt4Linear(nn.Module):
    def __init__(self, linear_module: nn.Module, group_size: int = 64, use_cpp_ext: bool = True, use_triton: bool = True):
        super().__init__()
        if not hasattr(linear_module, 'weight'):
            raise ValueError('PrunedInt4Linear requires a linear-like module with weight')
        self.in_features = int(linear_module.in_features)
        self.out_features = int(linear_module.out_features)
        self.group_size = int(group_size)
        self.source_dtype = linear_module.weight.dtype
        self.use_cpp_ext = bool(use_cpp_ext)
        self.use_triton = bool(use_triton)
        qweight, scales, in_features, in_features_padded, pad, group_size = quantize_tensor_int4_symmetric(
            linear_module.weight.data, group_size=self.group_size
        )
        self.register_buffer('qweight', qweight, persistent=True)
        self.register_buffer('scales', scales, persistent=True)
        self.in_features_orig = in_features
        self.in_features_padded = in_features_padded
        self.pad = pad
        self.weight = nn.Parameter(linear_module.weight.detach().to(torch.float16), requires_grad=False)
        if linear_module.bias is not None:
            self.bias = nn.Parameter(linear_module.bias.detach().to(torch.float16), requires_grad=False)
        else:
            self.bias = None
        self._cached_weight = None
        self._cached_weight_device = None
        self._cached_weight_dtype = None
        self._cached_bias = None
        self._cached_bias_device = None
        self._cached_bias_dtype = None
        self._cache_hits = 0
        self._cache_misses = 0
        self._cpp_ext = None
        self._qweight_cuda = None
        self._qweight_cuda_device = None
        self._scales_cuda_fp16 = None
        self._scales_cuda_device = None
        self._triton_qweight_packed = None
        self._triton_scales_packed = None
        self._triton_pack_device = None

    def _get_qweight_cuda(self, device):
        device_key = str(device)
        if self._qweight_cuda is not None and self._qweight_cuda_device == device_key:
            return self._qweight_cuda
        q = self.qweight if self.qweight.device == device else self.qweight.to(device=device)
        self._qweight_cuda = q
        self._qweight_cuda_device = device_key
        return q

    def _get_scales_cuda_fp16(self, device):
        device_key = str(device)
        if self._scales_cuda_fp16 is not None and self._scales_cuda_device == device_key:
            return self._scales_cuda_fp16
        if self.scales.device == device and self.scales.dtype == torch.float16:
            s = self.scales
        else:
            s = self.scales.to(device=device, dtype=torch.float16)
        self._scales_cuda_fp16 = s
        self._scales_cuda_device = device_key
        return s

    def _get_dequantized_weight(self, device, dtype):
        if self._cached_weight is not None and self._cached_weight_device == str(device) and self._cached_weight_dtype == str(dtype):
            self._cache_hits += 1
            return self._cached_weight
        self._cache_misses += 1
        q = self._get_qweight_cuda(device)
        s = self._get_scales_cuda_fp16(device)
        w = q.to(torch.float16) * s.unsqueeze(-1)
        w = w.view(q.shape[0], self.in_features_padded)[:, :self.in_features_orig].contiguous()
        if w.dtype != dtype:
            w = w.to(dtype=dtype)
        self._cached_weight = w
        self._cached_weight_device = str(device)
        self._cached_weight_dtype = str(dtype)
        return w

    def _get_triton_packed_weights(self, device):
        if self._triton_qweight_packed is not None and self._triton_pack_device == str(device):
            return self._triton_qweight_packed, self._triton_scales_packed
        q = self.qweight.to(device=device).contiguous()
        s = self.scales.to(device=device, dtype=torch.float16).contiguous()
        q_packed = q.permute(1, 2, 0).contiguous()
        s_packed = s.transpose(0, 1).contiguous()
        self._triton_qweight_packed = q_packed
        self._triton_scales_packed = s_packed
        self._triton_pack_device = str(device)
        return q_packed, s_packed

    def _try_triton_linear(self, x: torch.Tensor):
        if not self.use_triton:
            return None
        q_packed, s_packed = self._get_triton_packed_weights(x.device)
        bias = self.bias if self.bias is not None else None
        return TritonPrunedInt4Linear.forward(
            x,
            q_packed,
            s_packed,
            bias,
            int(self.in_features_orig),
            int(self.in_features_padded),
        )

    def _try_cpp_linear(self, x: torch.Tensor):
        if not self.use_cpp_ext:
            return None
        if x.device.type != 'cuda':
            return None
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return None
        if self._cpp_ext is None:
            self._cpp_ext = load_pruned_int4_ext(verbose=False)
        if self._cpp_ext is None:
            return None
        bias = self.bias if self.bias is not None else None
        return self._cpp_ext.pruned_int4_linear_forward(
            x,
            self._get_qweight_cuda(x.device),
            self._get_scales_cuda_fp16(x.device),
            bias,
            int(self.in_features_orig),
            int(self.in_features_padded),
        )

    def _blockwise_linear(self, x: torch.Tensor):
        out = self._try_triton_linear(x)
        if out is not None:
            return out
        out = self._try_cpp_linear(x)
        if out is not None:
            return out
        weight = self._get_dequantized_weight(device=x.device, dtype=x.dtype)
        bias = self._get_cached_bias(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)

    def _get_cached_bias(self, device, dtype):
        if self.bias is None:
            return None
        device_key = str(device)
        dtype_key = str(dtype)
        if self._cached_bias is not None and self._cached_bias_device == device_key and self._cached_bias_dtype == dtype_key:
            return self._cached_bias
        if self.bias.device == device and self.bias.dtype == dtype:
            b = self.bias
        else:
            b = self.bias.to(device=device, dtype=dtype)
        self._cached_bias = b
        self._cached_bias_device = device_key
        self._cached_bias_dtype = dtype_key
        return b

    def clear_cache(self):
        self._cached_weight = None
        self._cached_weight_device = None
        self._cached_weight_dtype = None
        self._cached_bias = None
        self._cached_bias_device = None
        self._cached_bias_dtype = None
        self._qweight_cuda = None
        self._qweight_cuda_device = None
        self._scales_cuda_fp16 = None
        self._scales_cuda_device = None
        self._triton_qweight_packed = None
        self._triton_scales_packed = None
        self._triton_pack_device = None

    def warmup_cache(self, device, dtype):
        self._get_dequantized_weight(device=device, dtype=dtype)
        self._get_cached_bias(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        return self._blockwise_linear(x)


def replace_modules_with_pruned_int4(model: nn.Module, target_suffixes, group_size: int = 64, logger=print):
    replaced = []
    for module_name, module in list(model.named_modules()):
        if not any(module_name.endswith(sfx) for sfx in target_suffixes):
            continue
        if not hasattr(module, 'in_features') or not hasattr(module, 'out_features') or not hasattr(module, 'weight'):
            continue
        parent = model
        parts = module_name.split('.')
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]
        wrapped = PrunedInt4Linear(module, group_size=group_size)
        setattr(parent, child_name, wrapped)
        replaced.append(module_name)
    logger(f'[pruned_int4] replaced modules count={len(replaced)} | modules={replaced[:12]}{" ..." if len(replaced) > 12 else ""}')
    return replaced
