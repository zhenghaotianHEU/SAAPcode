import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _rowwise_dequant_kernel(
        q_ptr, s_ptr, w_ptr,
        out_features, num_groups, group_size, in_features,
        stride_q0, stride_q1, stride_q2,
        stride_s0, stride_s1,
        stride_w0, stride_w1,
        BLOCK_G: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_g = tl.program_id(axis=0)
        pid_grp = tl.program_id(axis=1)

        offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        offs_k = tl.arange(0, BLOCK_K)

        mask_g = offs_g < out_features
        mask_k = offs_k < group_size

        q_ptrs = q_ptr + offs_g[:, None] * stride_q0 + pid_grp * stride_q1 + offs_k[None, :] * stride_q2
        q = tl.load(q_ptrs, mask=mask_g[:, None] & mask_k[None, :], other=0)

        s_ptrs = s_ptr + offs_g * stride_s0 + pid_grp * stride_s1
        s = tl.load(s_ptrs, mask=mask_g, other=0)

        w = q.to(tl.float16) * s[:, None].to(tl.float16)

        out_cols = pid_grp * group_size + offs_k
        out_ptrs = w_ptr + offs_g[:, None] * stride_w0 + out_cols[None, :] * stride_w1
        out_mask = mask_g[:, None] & (out_cols[None, :] < in_features)
        tl.store(out_ptrs, w, mask=out_mask)


class TritonPrunedInt4Linear:
    @staticmethod
    def available():
        return HAS_TRITON and torch.cuda.is_available()

    @staticmethod
    def _dequantize_to_weight(qweight, scales, in_features, in_features_padded, dtype=torch.float16):
        if not TritonPrunedInt4Linear.available():
            return None
        if qweight.device.type != 'cuda':
            return None
        if dtype != torch.float16:
            return None
        if qweight.dim() != 3:
            return None

        out_features = qweight.shape[0]
        num_groups = qweight.shape[1]
        group_size = qweight.shape[2]
        weight = torch.empty((out_features, in_features), device=qweight.device, dtype=dtype)

        grid = (triton.cdiv(out_features, 32), num_groups)
        _rowwise_dequant_kernel[grid](
            qweight, scales, weight,
            out_features, num_groups, group_size, in_features,
            qweight.stride(0), qweight.stride(1), qweight.stride(2),
            scales.stride(0), scales.stride(1),
            weight.stride(0), weight.stride(1),
            BLOCK_G=32,
            BLOCK_K=64,
        )
        return weight

    @staticmethod
    def forward(x, qweight, scales, bias, in_features, in_features_padded):
        if not TritonPrunedInt4Linear.available():
            return None
        if x.device.type != 'cuda':
            return None
        if x.dtype != torch.float16:
            return None
        if qweight.dtype != torch.int8:
            return None
        if scales.dtype not in (torch.float16, torch.float32):
            return None

        weight = TritonPrunedInt4Linear._dequantize_to_weight(
            qweight.to(device=x.device),
            scales.to(device=x.device, dtype=torch.float16),
            in_features,
            in_features_padded,
            dtype=x.dtype,
        )
        if weight is None:
            return None

        x2d = x.contiguous().view(-1, in_features)
        y2d = torch.matmul(x2d, weight.transpose(0, 1))
        if bias is not None:
            y2d = y2d + bias.to(device=x.device, dtype=x.dtype)
        out_shape = list(x.shape)
        out_shape[-1] = weight.shape[0]
        return y2d.view(*out_shape)
