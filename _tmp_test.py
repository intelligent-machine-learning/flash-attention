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
d = 32
#dtype=torch.bfloat16
dtype=torch.float16

qkv = torch.randn(batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype,
                  requires_grad=True)
with freeze_rng_state():
    out, lse, S_dmask = flash_attn_qkvpacked_func(
        qkv, 0.0, return_attn_probs=True, causal=True,
        glm_mask=torch.tensor(
            [
                [[0, 50, 150], [20, 100, 300]],
                [[0, 50, 150], [20, 100, 300]],
                [[0, 50, 150], [20, 100, 300]],
                [[0, 50, 150], [20, 100, 300]],
            ]
        ).to(torch.int32).to(device)
    )
k = 10
k = 200
k = 0  # not glm yet

# assume pack_attention_mask: startpoint/endpoint: (0, 20) (50, 100)(150, 300) other 3
# pack_glm_mask = torch.tensor(
#     [
#         [[0, 20], [50, 100], [150, 300]],
#         [[0, 31], [122, 200], [360, 500]],
#         [[0, 21], [51, 100], [150, 301]],
#         [[0, 30], [90, 100], [250, 300]],
#     ]
# )
pack_glm_mask = torch.tensor(
    [
        [[0, 20], [50, 100], [150, 300]],
        [[0, 20], [50, 100], [150, 300]],
        [[0, 20], [50, 100], [150, 300]],
        [[0, 20], [50, 100], [150, 300]],
    ]
)

def mock_mask(device):
    pack_attention_mask = torch.tril(
                torch.ones([seqlen, seqlen]))
    def build_mask_matrix(seq_length, sep):
        # https://github.com/pytorch/pytorch/issues/101932, fix triu/tril bf16 support
        m = torch.ones((1, seq_length, seq_length))
        mask = torch.arange(
            1, m.shape[-1] + 1).reshape(1, -1, 1).to(m.device)
        ids = torch.arange(
            1, m.shape[-1] + 1).reshape(1, 1, -1).expand(1, m.shape[-1], -1).to(m.device)
        m = (ids <= mask).type_as(m)
        m[0, :, :int(sep)] = 1
        m = m.squeeze(0)
        return m
    attention_mask_list = []
    for len_data, len_input in [(50, 20), (150 - 50, 100 - 50), (seqlen - 150, 300 - 150)]:
        attention_mask = build_mask_matrix(len_data, len_input)
        attention_mask_list.append(attention_mask)
    total_len = 0
    for i in range(len(attention_mask_list)):
        attention_mask = attention_mask_list[i]
        pack_attention_mask[total_len:total_len + attention_mask.shape[0],
                            total_len:total_len + attention_mask.shape[0]] = attention_mask
        total_len += len(attention_mask_list[i])
    return pack_attention_mask.to(device)


Q = qkv[:, :, 0, :, :]
K = qkv[:, :, 1, :, :]
V = qkv[:, :, 2, :, :]


def ref_attn_compute(Q, K, V, mask=None):
    attn_mask = torch.ones(seqlen, seqlen, dtype=torch.float32, device=device).tril(diagonal=0)
    if mask is not None:
        attn_mask = attn_mask.bool() | mask.bool()
    attn_mask = (1-attn_mask.float())*(-60000.)
    attn_mask = attn_mask.to(dtype)
    #import pdb;pdb.set_trace()
    #attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) + attn_mask, dim=-1)
    #attn = attn_weight @ V
    attn_weight = torch.softmax(torch.matmul(Q.permute(0, 2, 1, 3), K.permute(0, 2, 3, 1)) / math.sqrt(Q.size(-1)) + attn_mask, dim=-1)
    attn = attn_weight @ V.permute(0, 2, 1, 3)
    attn = attn.permute(0, 2, 1, 3)
    return attn






#with freeze_rng_state():
with freeze_rng_state(), torch.cuda.amp.autocast(dtype=dtype):
    autocast_attn = ref_attn_compute(Q, K, V, mask=mock_mask(device))
fp32_attn = ref_attn_compute(Q.to(torch.float32), K.to(torch.float32), V.to(torch.float32), mask=mock_mask(device))

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
print(torch.allclose(attn, out, atol=1e-3, rtol=1e-3))