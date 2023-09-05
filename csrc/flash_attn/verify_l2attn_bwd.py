import math

import torch


def mul_fwd(qk):
    q = qk[:, :, 0, :, :].permute(0,2,1,3)  # b,h,s,d
    k = qk[:, :, 1, :, :].permute(0,2,1,3)  # b,h,s,d
    S = torch.matmul(q, k.permute(0,1,3,2))
    S.retain_grad()
    return S


def mul_bwd(qk, S, dS):
    q = qk[:, :, 0, :, :].permute(0,2,1,3)  # b,h,s,d
    k = qk[:, :, 1, :, :].permute(0,2,1,3)  # b,h,s,d
    dQ = torch.matmul(dS, k)
    dK = torch.matmul(dS.transpose(2,3), q)
    return torch.stack([dQ, dK], 0).permute(1,3,0,2,4)


def test_mul_bwd():
    device = torch.device("cuda")
    dtype = torch.float32
    b, s, h, d = 4, 128, 8, 64
    # q = torch.arange(0, h * d).type(dtype).to(device)
    # k = 2 * torch.arange(0, h * d).type(dtype).to(device)
    # qk = torch.stack([q,k], 0).unsqueeze(0).unsqueeze(0).repeat(b,s,1,1).view(b,s,2,h,d)\
    qk = torch.randn(
        (b, s, 2, h, d),
        device=device,
        dtype=dtype
    )
    qk.requires_grad = True
    S = mul_fwd(qk)
    labels = torch.randn_like(S.float())
    diff = S.float() - labels
    loss = (diff * diff).sum() / b
    loss.backward()
    
    dS_ref = S.grad
    dQK_ref = qk.grad
    
    dQK = mul_bwd(qk, S, dS_ref)
    # import pdb; pdb.set_trace()
    assert (dQK - dQK_ref).abs().max() < 1e-3


def cdist_fwd(qk):
    q = qk[:, :, 0, :, :].permute(0,2,1,3)  # b,h,s,d
    k = qk[:, :, 1, :, :].permute(0,2,1,3)  # b,h,s,d
    b,h,s,d = q.shape
    S = -torch.cdist(q.float(), k.float(), 2) / math.sqrt(d)
    S.retain_grad()
    return S


def cdist_bwd(qk, S, dS):
    q = qk[:, :, 0, :, :].permute(0,2,1,3)  # b,h,s,d
    k = qk[:, :, 1, :, :].permute(0,2,1,3)  # b,h,s,d
    b,h,s,d = q.shape
    div_dS_S = dS / S  # b,h,s,s
    dot_dS_S_K = torch.matmul(div_dS_S, k)  # b,h,s,d
    rowsum = torch.sum(div_dS_S, 3)
    dQ = torch.empty_like(q)
    for bb in range(b):
        for hh in range(h):
            for i in range(s):
                dQ[bb,hh,i,:] = dot_dS_S_K[bb,hh,i,:] - q[bb,hh,i,:] * rowsum[bb,hh,i]
    # dQ = torch.zeros_like(q)
    # for bb in range(b):
    #     for hh in range(h):
    #         for i in range(s):
    #             for t in range(d):
    #                 for j in range(s):
    #                     dQ[bb,hh,i,t] += (k[bb,hh,j,t]-q[bb,hh,i,t])*dS[bb,hh,i,j]/S[bb,hh,i,j]
    
    dot_dS_S_T_Q = torch.matmul(div_dS_S.transpose(2,3), q)
    colsum = torch.sum(div_dS_S, 2)
    dK = torch.empty_like(k)
    for bb in range(b):
        for hh in range(h):
            for j in range(s):
                dK[bb,hh,j,:] = dot_dS_S_T_Q[bb,hh,j,:] - k[bb,hh,j,:] * colsum[bb,hh,j]
    # dK = torch.zeros_like(k)
    # for bb in range(b):
    #     for hh in range(h):
    #         for j in range(s):
    #             for t in range(d):
    #                 for i in range(s):
    #                     dK[bb,hh,j,t] += (q[bb,hh,i,t]-k[bb,hh,j,t])*dS[bb,hh,i,j]/S[bb,hh,i,j]
    dQ = -dQ/d
    dK = -dK/d
    return torch.stack([dQ, dK], 0).permute(1,3,0,2,4)  # b,s,2,h,d


def test_cdist_bwd():
    device = torch.device("cuda")
    dtype = torch.float32
    b, s, h, d = 16, 512, 8, 64
    # q = torch.arange(0, h * d).type(dtype).to(device)
    # k = 2 * torch.arange(0, h * d).type(dtype).to(device)
    # qk = torch.stack([q,k], 0).unsqueeze(0).unsqueeze(0).repeat(b,s,1,1).view(b,s,2,h,d)
    qk = torch.randn(
        (b, s, 2, h, d),
        device=device,
        dtype=dtype
    )
    qk.requires_grad = True
    S = cdist_fwd(qk)
    labels = torch.randn_like(S.float())
    diff = S.float() - labels
    loss = (diff * diff).sum() / b
    loss.backward()
    
    dS_ref = S.grad
    dQK_ref = qk.grad
    
    # import pdb; pdb.set_trace()
    dQK = cdist_bwd(qk, S, dS_ref)

    assert (dQK - dQK_ref).abs().max() < 1e-3


if __name__ == "__main__":
    # test_mul_bwd()
    test_cdist_bwd()
