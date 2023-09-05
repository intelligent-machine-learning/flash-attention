import ctypes
import math
import os
import unittest
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward, _l2attn_forward, _l2attn_backward


_cudart = ctypes.CDLL('libcudart.so')

def start():
    # As shown at http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html,
    # the return value will unconditionally be 0. This check is just in case it changes in
    # the future.
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)

def stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)


def cos_sim(a, b):
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return F.cosine_similarity(a, b).detach().cpu().item()


def py_mha(qkv, d):
    q = qkv[:, :, 0, :, :].permute(0,2,1,3)
    k = qkv[:, :, 1, :, :].permute(0,2,1,3)
    v = qkv[:, :, 2, :, :].permute(0,2,1,3)
    # p = torch.matmul(q, k.permute(0,1,3,2)).float()
    p = torch.matmul(q.float(), k.permute(0,1,3,2).float())
    p_masked = p / math.sqrt(d) # + (1.0 - amask) * -10000.0
    s = torch.softmax(p_masked, -1).to(qkv.dtype)
    ctx = torch.matmul(s, v)
    ctx = ctx.permute(0,2,1,3).contiguous()
    ctx.retain_grad()
    return ctx, p


def py_l2_attn(qkv, d):
    q = qkv[:, :, 0, :, :].permute(0,2,1,3)
    k = qkv[:, :, 1, :, :].permute(0,2,1,3)
    v = qkv[:, :, 2, :, :].permute(0,2,1,3)
    p = -torch.cdist(q.float(),k.float(),2)
    p_masked = p / math.sqrt(d) # + (1.0 - amask) * -10000.0
    s = torch.softmax(p_masked, -1).to(qkv.dtype)
    ctx = torch.matmul(s, v)
    ctx = ctx.permute(0,2,1,3).contiguous()
    ctx.retain_grad()
    return ctx, p


class TestL2Attn(unittest.TestCase):    
    def run_acc(self, s, b, h, d, l2_attn: bool=True):
        print(f'**************************** Test s={s}, b={b}, h={h}, d={d}')

        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)

        device = torch.device('cuda')
        dtype = torch.float16

        qkv_raw = torch.randn(
            (b, s, 3, h, d),
            device=device,
            dtype=dtype
        )
        qkv_raw.requires_grad = True

        batch_size = qkv_raw.shape[0]
        seqlen = qkv_raw.shape[1]
        qkv = rearrange(qkv_raw, 'b s ... -> (b s) ...')
        max_s = seqlen
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                device=qkv_raw.device)
        softmax_scale = qkv.shape[-1] ** (-0.5)
        dropout_p = 0.0

        if l2_attn:
            # start()
            output, softmax_lse, S_dmask = _l2attn_forward(
                qkv[:, 0], qkv[:, 1], qkv[:, 2], torch.empty_like(qkv[:, 0]), cu_seqlens, cu_seqlens,
                max_s, max_s, dropout_p, softmax_scale, causal=False,
                return_softmax=False, attn_mask=None, attn_bias=None
            )
            # end()
            output_ref, l2 = py_l2_attn(qkv_raw, d)
        else:
            # start()
            output, softmax_lse, S_dmask = _flash_attn_forward(
                qkv[:, 0], qkv[:, 1], qkv[:, 2], torch.empty_like(qkv[:, 0]), cu_seqlens, cu_seqlens,
                max_s, max_s, dropout_p, softmax_scale, causal=False,
                return_softmax=False, attn_mask=None, attn_bias=None
            )
            # end()
            output_ref, l2 = py_mha(qkv_raw, d)

        output_eval = rearrange(output, '(b s) ... -> b s ...', b=batch_size)

        print("cos(output_eval, output_ref):", cos_sim(output_eval, output_ref))
        print(f'Output max diff: {(output_eval - output_ref).abs().max().item()}')
        print(f'Output mean diff: {(output_eval - output_ref).abs().mean().item()}')
        self.assertTrue(torch.allclose(output_ref.float(), output_eval.float(), atol=1e-3))

        labels = torch.randn_like(output_ref.float())
        diff = output_ref.float() - labels
        l = (diff * diff).sum() / b
        l.backward()

        dout = output_ref.grad.clone().detach().view(b*s, h, d).contiguous()

        dqkv = torch.empty_like(qkv)
        torch.cuda.synchronize()
        t0 = time()
        if l2_attn:
            _, _, _, _, dbias = _l2attn_backward(
                dout, qkv[:, 0], qkv[:, 1], qkv[:, 2], output, softmax_lse,
                dqkv[:, 0], dqkv[:, 1], dqkv[:, 2], cu_seqlens, cu_seqlens,
                max_s, max_s, dropout_p, softmax_scale, False,
                attn_mask=None, attn_bias=None
            )
        else:
            _, _, _, _, dbias = _flash_attn_backward(
                dout, qkv[:, 0], qkv[:, 1], qkv[:, 2], output, softmax_lse,
                dqkv[:, 0], dqkv[:, 1], dqkv[:, 2], cu_seqlens, cu_seqlens,
                max_s, max_s, dropout_p, softmax_scale, False,
                attn_mask=None, attn_bias=None
            )
        torch.cuda.synchronize()
        t1 = time()
        print(f"bwd op cost {t1-t0}ms.")

        dqkv = dqkv.view(b,s,3,h,d)

        print("cos(dQ, dQ_ref):", cos_sim(dqkv[:,:,0,:,:], qkv_raw.grad[:,:,0,:,:]))
        print("cos(dK, dK_ref):", cos_sim(dqkv[:,:,1,:,:], qkv_raw.grad[:,:,1,:,:]))
        print("cos(dV, dV_ref):", cos_sim(dqkv[:,:,2,:,:], qkv_raw.grad[:,:,2,:,:]))
        print(f'dQ max diff: {(dqkv[:,:,0,:,:] - qkv_raw.grad[:,:,0,:,:]).abs().max().item()}')
        print(f'dQ mean diff: {(dqkv[:,:,0,:,:] - qkv_raw.grad[:,:,0,:,:]).abs().mean().item()}')
        print(f'dK max diff: {(dqkv[:,:,1,:,:] - qkv_raw.grad[:,:,1,:,:]).abs().max().item()}')
        print(f'dK mean diff: {(dqkv[:,:,1,:,:] - qkv_raw.grad[:,:,1,:,:]).abs().mean().item()}')
        print(f'dV max diff: {(dqkv[:,:,2,:,:] - qkv_raw.grad[:,:,2,:,:]).abs().max().item()}')
        print(f'dV mean diff: {(dqkv[:,:,2,:,:] - qkv_raw.grad[:,:,2,:,:]).abs().mean().item()}')
        self.assertTrue(torch.allclose(qkv_raw.grad.float(), dqkv.float(), atol=1e-3))


    def test_acc(self):
        dim_list = [32, 64, 128]
        seq_list = [128, 256, 512, 1024, 2048, 4096]
        batch_list = [4, 8, 16]
        for d in dim_list:
            for s in seq_list:
                for b in batch_list:
                    self.run_acc(s, b, 8, d)


def run_op(s, b, h, d, fun_name: str="fused", process: str="fwd"):
    """
        fun_name: {"fused", "cdist"}
        process: {"fwd", "bwd"}
    """
    print(f'Test s={s}, b={b}, h={h}, d={d}')
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    dtype = torch.float16
    device = torch.device('cuda')

    qkv_raw = torch.randn(
        (b, s, 3, h, d),
        device=device,
        dtype=dtype
    )
    qkv_raw.requires_grad = True

    batch_size = qkv_raw.shape[0]
    seqlen = qkv_raw.shape[1]
    qkv = rearrange(qkv_raw, 'b s ... -> (b s) ...')
    max_s = seqlen
    cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                            device=qkv_raw.device)
    softmax_scale = qkv.shape[-1] ** (-0.5)

    def warmup(fun, count=10):
        for i in range(count):
            fun()

    if fun_name == "fused":
        if process == "fwd":
            fun = lambda : _l2attn_forward(
                    qkv[:, 0], qkv[:, 1], qkv[:, 2], torch.empty_like(qkv[:, 0]), cu_seqlens, cu_seqlens,
                    max_s, max_s, 0.0, softmax_scale, causal=False,
                    return_softmax=False, attn_mask=None, attn_bias=None
                )
        else:
            dout = torch.randn_like(qkv[:,0])
            output = torch.randn_like(qkv[:, 0])
            softmax_lse = torch.randn([b, h, max_s]).float().to(device)
            dqkv = torch.empty_like(qkv)
            fun = lambda : _l2attn_backward(
                dout, qkv[:, 0], qkv[:, 1], qkv[:, 2], output, softmax_lse,
                dqkv[:, 0], dqkv[:, 1], dqkv[:, 2], cu_seqlens, cu_seqlens,
                max_s, max_s, 0.0, softmax_scale, False,
                attn_mask=None, attn_bias=None
            )
    else:
        if process == "fwd":
            fun = lambda : py_l2_attn(qkv_raw, d)
        else:
            output_ref, l2 = py_l2_attn(qkv_raw, d)
            labels = torch.randn_like(output_ref.float())
            diff = output_ref.float() - labels
            l = (diff * diff).sum() / b
            fun = lambda : l.backward()
    if fun_name == "fused" or process == "fwd":
        warmup(fun)

    torch.cuda.synchronize()
    t0 = time()
    fun()
    torch.cuda.synchronize()
    t1 = time()
    gpu_mem = torch.cuda.memory_reserved(device)
    print(f"{fun_name}:{process} cost(ms): {(t1-t0)*1000:.3f}, gpu mem(GB): {gpu_mem/1024/1024/1024:.3f}")
    return t1-t0, gpu_mem


def test_op_perf():
    def record(res):
        for r in res:
            print(f"{r:.3f}\t", end="")
        print()
    def record_percent(res):
        for r in res:
            print(f"{r:.2%}\t", end="")
        print()
    seq_list = [256, 512, 1024, 2048]  #, 4096]
    batch_list = [4, 8, 16]  #,32,64,128]
    for s in seq_list:
        print(f"================================= s={s} ======================")
        fused_cost_all, fused_gpumem_all = [], []
        baseline_cost_all, baseline_gpumem_all = [], []
        for b in batch_list:
            print(f"=============== b={b}")
            bc, bm = run_op(s, b, 8, 64, "cdist", "bwd")
            baseline_cost_all.append(bc*1000)
            baseline_gpumem_all.append(bm/1024/1024/1024)

            torch.cuda.empty_cache()
            fc, fm = run_op(s, b, 8, 64, "fused", "bwd")
            fused_cost_all.append(fc*1000)
            fused_gpumem_all.append(fm/1024/1024/1024)
        for b in batch_list:
            print(f"{b}\t", end="")
        print()
        print("GPU mem(GB):")
        record(baseline_gpumem_all)
        record(fused_gpumem_all)
        record_percent([(f-b)/b for b,f in zip(baseline_gpumem_all, fused_gpumem_all)])
        print("cost(ms):")
        record(baseline_cost_all)
        record(fused_cost_all)
        record_percent([(b-f)/f for b,f in zip(baseline_cost_all, fused_cost_all)])


if __name__ == "__main__":
    # for cuda-gdb
    # a = input(f"Current PID is {os.getpid()} . Please enter any key after running gdb attach.")
    
    unittest.main()
    # test_op_perf()
