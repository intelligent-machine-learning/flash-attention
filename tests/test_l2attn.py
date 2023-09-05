import math
from functools import partial

import torch
import torch.nn.functional as F

import pytest

from einops import rearrange, repeat

from flash_attn.flash_attn_interface import l2attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis

try:
    from flash_attn.flash_attn_triton import flash_attn_func
except (ImportError, AttributeError):  # Older version of Triton doesn't have tl.constexpr
    flash_attn_func = None


is_sm75 = torch.cuda.get_device_capability('cuda') == (7, 5)
is_sm80 = torch.cuda.get_device_capability('cuda') == (8, 0)


def generate_random_padding_mask(max_seqlen, batch_size, device, mode='random'):
    assert mode in ['full', 'random', 'third', 'split']
    if mode == 'full':
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == 'random':
        lengths = torch.randint(max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device)
    elif mode == 'third':
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    elif mode == 'split':
        lengths0 = torch.randint(min(128, max_seqlen), max_seqlen + 1,
                                 (batch_size // 4 * 3, 1), device=device)
        lengths1 = torch.randint(min(max(1, max_seqlen - 20), 128), min(max_seqlen, 128) + 1,
                                 (batch_size - batch_size // 4 * 3, 1), device=device)
        lengths = torch.cat([lengths0, lengths1], dim=0)
    padding_mask = repeat(torch.arange(max_seqlen, device=device), 's -> b s', b=batch_size) < lengths
    return padding_mask


def generate_qkv(x, Wqkv, nheads, query_padding_mask=None, key_padding_mask=None,
                 kvpacked=False, qkvpacked=False):
    """
    Arguments:
        x: (batch_size, seqlen, nheads * d)
        Wqkv: nn.Linear(nheads * d, 3 * nheads * d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen, dim = x.shape
    q, k, v = Wqkv(x).chunk(3, dim=-1)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        q_unpad = rearrange(q_unpad, 'nnz (h d) -> nnz h d', h=nheads)
        output_pad_fn = lambda output_unpad: rearrange(
            pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'), indices_q, batch_size, seqlen),
            'b s (h d) -> b s h d', h=nheads
        )
    else:
        q_unpad = rearrange(q, 'b s (h d) -> (b s) h d', h=nheads)
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                    device=q_unpad.device)
        max_seqlen_q = seqlen
        output_pad_fn = lambda output_unpad: rearrange(output_unpad, '(b s) h d -> b s h d', b=batch_size)

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        k_unpad = rearrange(k_unpad, 'nnz (h d) -> nnz h d', h=nheads)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
        v_unpad = rearrange(v_unpad, 'nnz (h d) -> nnz h d', h=nheads)
    else:
        k_unpad = rearrange(k, 'b s (h d) -> (b s) h d', h=nheads)
        v_unpad = rearrange(v, 'b s (h d) -> (b s) h d', h=nheads)
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                    device=q_unpad.device)
        max_seqlen_k = seqlen

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = rearrange(torch.stack([q, k, v], dim=2), 'b s t (h d) -> b s t h d', h=nheads)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                pad_input(rearrange(dqkv_unpad, 'nnz t h d -> nnz (t h d)'), indices_q, batch_size, seqlen),
                'b s (t h d) -> b s t h d', t=3, h=nheads
            )
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(dqkv_unpad, '(b s) t h d -> b s t h d', b=batch_size)
        return (qkv_unpad.detach().requires_grad_(), cu_seqlens_q, max_seqlen_q,
                qkv.detach().requires_grad_(), output_pad_fn, dqkv_pad_fn)
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        q = rearrange(q, 'b s (h d) -> b s h d', h=nheads)
        kv = rearrange(torch.stack([k, v], dim=2), 'b s t (h d) -> b s t h d', h=nheads)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                pad_input(rearrange(dkv_unpad, 'nnz t h d -> nnz (t h d)'), indices_k, batch_size, seqlen),
                'b s (t h d) -> b s t h d', t=2, h=nheads
            )
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(dkv_unpad, '(b s) t h d -> b s t h d', b=batch_size)
        return (q_unpad.detach().requires_grad_(), kv_unpad.detach().requires_grad_(),
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                q.detach().requires_grad_(), kv.detach().requires_grad_(),
                output_pad_fn, dq_pad_fn, dkv_pad_fn)
    else:
        q, k, v = [rearrange(z, 'b s (h d) -> b s h d', h=nheads).detach().requires_grad_()
                   for z in [q, k, v]]
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: rearrange(
                pad_input(rearrange(dk_unpad, 'nnz h d -> nnz (h d)'), indices_k, batch_size, seqlen),
                'b s (h d) -> b s h d', h=nheads
            )
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, '(b s) h d -> b s h d', b=batch_size)
        return (q_unpad.detach().requires_grad_(), k_unpad.detach().requires_grad_(),
                v_unpad.detach().requires_grad_(),
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                q, k, v,
                output_pad_fn, dq_pad_fn, dk_pad_fn)


def l2attention_ref(q, k, v, query_padding_mask=None, key_padding_mask=None, dropout_p=0.0,
                  dropout_mask=None, causal=False, bias=None, upcast=True, reorder_ops=False):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads, head_dim)
        v: (batch_size, seqlen_k, nheads, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        bias: (batch_size, nheads, seqlen_q, seqlen_k)
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    d = q.shape[-1]
    # if not reorder_ops:
    #     scores = torch.einsum('bthd,bshd->bhts', q / math.sqrt(d), k)
    # else:
    #     scores = torch.einsum('bthd,bshd->bhts', q, k / math.sqrt(d))
    scores = -torch.cdist(q.permute(0,2,1,3), k.permute(0,2,1,3)) / math.sqrt(d)
    if bias is not None:
        scores = (scores + bias).to(dtype=scores.dtype)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, 'b s -> b 1 1 s'), float('-inf'))
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device), 1)
        scores.masked_fill_(causal_mask, float('-inf'))
    attention = torch.softmax(scores, dim=-1)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum('bhts,bshd->bthd', attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, 'b s -> b s 1 1'), 0.0)
        attention = attention.masked_fill(rearrange(~query_padding_mask, 'b s -> b 1 s 1'), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def l2attention_qkvpacked_ref(qkv, key_padding_mask=None, dropout_p=0.0,
                            dropout_mask=None, causal=False, upcast=True, reorder_ops=False):
    return l2attention_ref(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], key_padding_mask,
                         key_padding_mask, dropout_p, dropout_mask, upcast=upcast, causal=causal,
                         reorder_ops=reorder_ops)


def convert_flash_attn_S_to_softmax(S, query_padding_mask, key_padding_mask, head_dim, is_dropout,
                                    causal=False):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q, seqlen_k)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
    """
    S_flat = rearrange(S, 'b h t s -> b h (t s)')
    seqlen_q, seqlen_k = S.shape[-2:]
    block_size = _get_block_size(S.device, head_dim, is_dropout)
    loop_steps = (seqlen_k + block_size - 1) // block_size
    warps_n = 4
    mmas_n = (seqlen_k // warps_n // 16) if seqlen_k <= block_size else (block_size // warps_n // 16)
    S_converted = rearrange(S_flat, 'b h (loop nsteps mmas_n warps_n eight t r c0 c1) -> b h (nsteps r eight) (loop mmas_n warps_n c0 t c1)',
                            loop=loop_steps, nsteps=seqlen_q // 16, mmas_n=mmas_n, warps_n=warps_n, eight=8, t=4,
                            r=2, c0=2, c1=2)

    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    seqlen_q_og = query_padding_mask.shape[-1]
    if seqlen_q_og < seqlen_q:
        query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q - seqlen_q_og))
    else:
        query_padding_mask = query_padding_mask[:, :seqlen_q]
    S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, 'b s -> b 1 s 1'), 0.0)
    seqlen_k_og = key_padding_mask.shape[-1]
    if seqlen_k_og < seqlen_k:
        key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k - seqlen_k_og))
    else:
        key_padding_mask = key_padding_mask[:, :seqlen_k]
    S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, 'b s -> b 1 1 s'), 0.0)
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=S.device), 1)
        S_converted.masked_fill_(causal_mask, 0.0)
    if seqlen_q_og < seqlen_q:
        S_converted = S_converted[:, :, :seqlen_q_og, :]
    else:
        S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q))
    if seqlen_k_og < seqlen_k:
        S_converted = S_converted[:, :, :, :seqlen_k_og]
    else:
        S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k))
    return S_converted


def normalize_l2attn_S(attn_unnorm, q, k, v, query_padding_mask=None, key_padding_mask=None,
                           is_dropout=False, causal=False):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k, v: (batch_size, seqlen_k, nheads, head_dim)
        key_padding_mask: (batch_size, seqlen_q)
    Output:
        softmax_lse: (batch_size, nheads, seqlen_q)
        softmax_max: (batch_size, nheads, seqlen_q)
    """
    q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, _, head_dim = q.shape
    seqlen_k = k.shape[1]
    # scores = torch.einsum('bthd,bshd->bhts', q / math.sqrt(head_dim), k)
    scores = -torch.cdist(q.permute(0,2,1,3), k.permute(0,2,1,3)) / math.sqrt(head_dim)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, 'b s -> b 1 1 s'), float('-inf'))
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device), 1)
        scores.masked_fill_(causal_mask, float('-inf'))
    block_size = _get_block_size(scores.device, head_dim, is_dropout)
    scores_block = scores.split(block_size, dim=-1)
    lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
    lcse_block = torch.logcumsumexp(lse_block, dim=-1).unbind(dim=-1)
    scores_max_block = ([torch.amax(scores_block[0], dim=-1)]
                        + [torch.maximum(torch.amax(s, dim=-1), lcse)
                           for s, lcse in zip(scores_block[1:], lcse_block[:-1])])
    attn_unnorm_block = attn_unnorm.split(block_size, dim=-1)
    attn_norm = torch.cat([a / rearrange(torch.exp(lcse_block[-1] - m), 'b h s -> b h s 1')
                           for a, m in zip(attn_unnorm_block, scores_max_block)], dim=-1)
    if query_padding_mask is not None:
        attn_norm.masked_fill_(rearrange(~query_padding_mask, 'b s -> b 1 s 1'), 0.0)
    return attn_norm.to(dtype=attn_unnorm.dtype)


def get_dropout_fraction(dropout_mask, query_padding_mask=None, key_padding_mask=None, causal=False):
    """
    dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k), bool. True means keep, False means drop.
    query_padding_mask: (batch_size, seqlen_q)
    key_padding_mask: (batch_size, seqlen_k)
    """
    batch_size, nheads, seqlen_q, seqlen_k = dropout_mask.shape
    dropped = ~dropout_mask
    if query_padding_mask is not None:
        dropped.masked_fill_(rearrange(~query_padding_mask, 'b s -> b 1 s 1'), False)
    if key_padding_mask is not None:
        dropped.masked_fill_(rearrange(~key_padding_mask, 'b s -> b 1 1 s'), False)
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool,
                                            device=dropout_mask.device), 1)
        dropped.masked_fill_(causal_mask, False)
    dropped_total = dropped.sum()
    query_lengths = (query_padding_mask.sum(dim=-1) if query_padding_mask is not None
                     else torch.full((batch_size,), seqlen_q, device=dropout_mask.device))
    key_lengths = (key_padding_mask.sum(dim=-1) if key_padding_mask is not None
                   else torch.full((batch_size,), seqlen_k, device=dropout_mask.device))
    if not causal:
        numel_per_batch = query_lengths * key_lengths
    else:
        numel_per_batch = torch.where(
            query_lengths <= key_lengths,
            query_lengths * (query_lengths + 1) / 2,
            query_lengths * key_lengths - (key_lengths * (key_lengths - 1) / 2)
        )
    return dropped_total / (numel_per_batch.sum() * nheads)


# @pytest.mark.parametrize('dtype', ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('causal', [False, True])
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('d', [128, 80, 64, 40, 32, 16])
# @pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize('seqlen', [97, 128, 200, 256, 257, 384, 512, 768, 1024, 1025, 2048])
# @pytest.mark.parametrize('seqlen', [128])
@pytest.mark.parametrize('dropout_p', [0.0, 0.17])
# @pytest.mark.parametrize('dropout_p', [0.0])
def test_l2attn(seqlen, d, dropout_p, causal, dtype):
    if seqlen >= 2048 and torch.cuda.get_device_properties('cuda').total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = 'cuda'
    # if dtype == torch.float16:
    #     rtol, atol = (1e-3, 3e-4) if not causal else (1e-3, 1e-3)
    # else:  # torch.bfloat16
    #     rtol, atol = (3e-3, 3e-3) if not causal else (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    # Set smaller batch size so it would trigger num_splits > 1
    batch_size = 8
    nheads = 4
    x = torch.randn(batch_size, seqlen, nheads * d, device=device, dtype=dtype, requires_grad=True)
    Wqkv = torch.nn.Linear(nheads * d, 3 * nheads * d, device=device, dtype=dtype)

    key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode='random')
    # key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode='full')

    qkv_unpad, cu_seqlens, max_seqlen, qkv, output_pad_fn, dqkv_pad_fn = generate_qkv(
        x, Wqkv, nheads, key_padding_mask, key_padding_mask, qkvpacked=True
    )

    output_unpad, sm_lse, S_dmask = l2attn_unpadded_qkvpacked_func(
        qkv_unpad, cu_seqlens, max_seqlen, dropout_p, return_attn_probs=True, causal=causal
    )
    output = output_pad_fn(output_unpad)
    S_dmask_converted = convert_flash_attn_S_to_softmax(
        S_dmask, key_padding_mask, key_padding_mask, d, dropout_p > 0.0, causal=causal
    )
    dropout_mask = S_dmask_converted >= 0
    attn_unnorm = S_dmask_converted.abs()
    attn = normalize_l2attn_S(attn_unnorm, qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2],
                                  key_padding_mask, key_padding_mask, dropout_p > 0.0, causal=causal)
    dropout_fraction = get_dropout_fraction(dropout_mask, key_padding_mask, key_padding_mask,
                                            causal=causal).item()

    output_ref, attn_ref = l2attention_qkvpacked_ref(qkv, key_padding_mask, dropout_p, dropout_mask,
                                                   causal=causal)
    output_pt, attn_pt = l2attention_qkvpacked_ref(qkv, key_padding_mask, dropout_p, dropout_mask,
                                                 causal=causal, upcast=False, reorder_ops=True)
    print(f'Actual dropout fraction: {dropout_fraction}')
    print(f'Output max diff: {(output - output_ref).abs().max().item()}')
    print(f'Output mean diff: {(output - output_ref).abs().mean().item()}')
    print(f'Pytorch max diff: {(output_pt - output_ref).abs().max().item()}')
    print(f'Pytorch mean diff: {(output_pt - output_ref).abs().mean().item()}')
    print(f'Attention max diff: {(attn - attn_ref).abs().max().item()}')
    print(f'Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}')

    if is_sm80 or d <= 64:  # Only run backward for d=128 on A100
        g = torch.randn_like(output)
        dqkv_unpad, = torch.autograd.grad(output, qkv_unpad, g)
        dqkv = dqkv_pad_fn(dqkv_unpad)
        dqkv_ref, = torch.autograd.grad(output_ref, qkv, g)
        dqkv_pt, = torch.autograd.grad(output_pt, qkv, g)
        print(f'dQ max diff: {(dqkv[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}')
        print(f'dK max diff: {(dqkv[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}')
        print(f'dV max diff: {(dqkv[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}')
        print(f'dQKV mean diff: {(dqkv - dqkv_ref).abs().mean().item()}')
        print(f'dQ Pytorch max diff: {(dqkv_pt[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}')
        print(f'dK Pytorch max diff: {(dqkv_pt[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}')
        print(f'dV Pytorch max diff: {(dqkv_pt[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}')
        print(f'dQKV Pytorch mean diff: {(dqkv_pt - dqkv_ref).abs().mean().item()}')

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (output - output_ref).abs().max().item() <= 2 * (output_pt - output_ref).abs().max().item()
    # assert torch.allclose(output, output_ref, rtol=rtol, atol=atol)
    assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
    # assert torch.allclose(attn, attn_ref, rtol=rtol, atol=atol)
    if dropout_p == 0.0:
        assert dropout_mask.all()
    else:
        assert 0.98 <= dropout_fraction / dropout_p <= 1.02

    if is_sm80 or d <= 64:  # Only run backward for d=128 on A100
        # Error for dK and dV could be a bit higher if we're splitting along seqlen_q dimension
        assert (dqkv - dqkv_ref).abs().max().item() <= 4 * (dqkv_pt - dqkv_ref).abs().max().item()
        # assert torch.allclose(dqkv, dqkv_ref, rtol=rtol, atol=atol)
