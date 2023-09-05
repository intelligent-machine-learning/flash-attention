
import torch
from flash_attn.l2_attention import L2MHA, L2Attention

def test_L2MHA():
    # Replace this with your correct GPU device
    device = "cuda"

    # Create attention layer. This is similar to torch.nn.MultiheadAttention,
    # and it includes the input and output linear layers
    l2mha = L2MHA(
        embed_dim=512, # total channels (= num_heads * head_dim)
        num_heads=8, # number of heads
        device=device,
        dtype=torch.float16,
    )

    # Run forward pass with dummy data
    x = torch.randn(
        (32, 1024, 512), # (batch, seqlen, embed_dim)
        device=device,
        dtype=torch.float16
    )

    output = l2mha(x)[0]
    print(f"output shape: {output.shape}")


def test_L2Attention():
    # Replace this with your correct GPU device
    device = "cuda"
    dtype = torch.float16

    b = 32
    s = 1024
    h = 8
    d = 64

    qkv = torch.randn(
        (b, s, 3, h, d),
        device=device,
        dtype=dtype
    )

    l2attention = L2Attention()
    output = l2attention(qkv)[0]
    print(f"output shape: {output.shape}")


if __name__ == "__main__":
    test_L2MHA()
    test_L2Attention()
