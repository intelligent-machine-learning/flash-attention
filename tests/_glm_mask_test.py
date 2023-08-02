import torch
import math
from torch.testing._internal.common_utils import freeze_rng_state
from flash_attn import flash_attn_func, flash_attn_kvpacked_func, flash_attn_qkvpacked_func


device = 'cuda'
# set seed
torch.random.manual_seed(0)
#torch.backends.cudnn.deterministic = True
#batch_size = 4
#nheads = 1
#seqlen = 20
#d = 32
batch_size = 4
nheads = 16
seqlen = 512
# d = 32
d = 128
#dtype=torch.bfloat16
dtype=torch.float16

qkv = torch.randn(batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype,
                  requires_grad=True)
glm_mask = torch.randint(10, seqlen - 10, (batch_size,), device=device, dtype=torch.int32)
# glm_mask = torch.tensor([200, 200, 200, 200], device=device, dtype=torch.int32)

with freeze_rng_state():
    out, lse, S_dmask = flash_attn_qkvpacked_func(
        qkv, 0.0, return_attn_probs=True, causal=True, glm_mask=glm_mask
    )
    
    
# k = 10
# k = 200
Q = qkv[:, :, 0, :, :]
K = qkv[:, :, 1, :, :]
V = qkv[:, :, 2, :, :]


# conventional transformer
def build_mask_matrix(hidden_states, seq_length, sep, memory_length=0):
    m = hidden_states.new_ones((1, seq_length, seq_length))
    m = torch.tril(m)
    if False:  # is_scalar:
        m[0, :, :int(sep)] = 1
    else:
        m = m.expand(batch_size, -1, -1)
        ids = torch.arange(seq_length, device=sep.device, dtype=sep.dtype).view(1, -1)
        mask = ids < sep.view(-1, 1)
        m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
    if memory_length > 0:
        m = m.expand(batch_size, -1, -1)
        m = torch.cat((hidden_states.new_ones((batch_size, seq_length, memory_length)), m), dim=2)
    m = m.unsqueeze(1)
    return m

def ref_attn_compute(Q, K, V, glm_mask):
    attn_mask = build_mask_matrix(Q, seqlen, glm_mask)
    # attn_mask = torch.ones(seqlen, seqlen, dtype=torch.float32, device=device).tril(diagonal=0)
    # attn_mask[:, :k] = 1.
    attn_mask = (1-attn_mask)*(-60000.)
    # attn_mask = attn_mask.to(dtype)
    #import pdb;pdb.set_trace()
    #attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) + attn_mask, dim=-1)
    #attn = attn_weight @ V
    attn_weight = torch.softmax(torch.matmul(Q.permute(0, 2, 1, 3), K.permute(0, 2, 3, 1)) / math.sqrt(Q.size(-1)) + attn_mask, dim=-1)
    attn = attn_weight @ V.permute(0, 2, 1, 3)
    attn = attn.permute(0, 2, 1, 3)
    return attn






#with freeze_rng_state():
with freeze_rng_state(), torch.cuda.amp.autocast(dtype=dtype):
    autocast_attn = ref_attn_compute(Q, K, V, glm_mask)
fp32_attn = ref_attn_compute(Q.to(torch.float32), K.to(torch.float32), V.to(torch.float32), glm_mask)

out = torch.squeeze(out)

print(f'Output max diff: {(out - fp32_attn).abs().max().item()}')
print(f'Output mean diff: {(out - fp32_attn).abs().mean().item()}')
print(f'Pytorch max diff: {(autocast_attn - fp32_attn).abs().max().item()}')
print(f'Pytorch mean diff: {(autocast_attn - fp32_attn).abs().mean().item()}')

g = torch.randn_like(out)
dqkv, = torch.autograd.grad(out, qkv, g)
dqkv_ref, = torch.autograd.grad(fp32_attn, qkv, g)
dqkv_pt, = torch.autograd.grad(autocast_attn, qkv, g)

print(f'dQ max diff: {(dqkv[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}')
print(f'dK max diff: {(dqkv[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}')
print(f'dV max diff: {(dqkv[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}')
print(f'dQKV mean diff: {(dqkv - dqkv_ref).abs().mean().item()}')
print(f'dQ Pytorch max diff: {(dqkv_pt[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}')
print(f'dK Pytorch max diff: {(dqkv_pt[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}')
print(f'dV Pytorch max diff: {(dqkv_pt[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}')
print(f'dQKV Pytorch mean diff: {(dqkv_pt - dqkv_ref).abs().mean().item()}')


#print((out-attn).abs().max())
import pdb;pdb.set_trace()
# print(torch.allclose(attn, out, atol=1e-3, rtol=1e-3))