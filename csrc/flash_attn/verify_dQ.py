import torch


def test_dQ():
    Q = torch.arange(64).type(torch.float16).unsqueeze(0).repeat(16, 1)
    K = torch.arange(64).type(torch.float16).unsqueeze(0).repeat(256, 1)
    K += 1
    S = torch.empty([16, 256], dtype=torch.float16)
    dS = torch.empty_like(S)
    for i in range(16):
        for j in range(256):
            warp = j % 128
            ii = i % 16
            jj = j % 16
            lane = 4 * (ii if ii < 8 else ii - 8) + (jj if jj < 8 else jj - 8) // 2
            ri = (jj // 8) * 4 + (ii // 8) * 2 + (jj & 0x1)
            S[i,j] = lane * 10 + ri + 1
            dS[i,j] = lane
    
    dQ = torch.zeros([16, 64*8], dtype=Q.dtype)
    for ki in range(2):
        for i in range(16):
            for t in range(64):
                for warp in range(8):
                    for j in range(16):
                        k_row = ki * 128 + warp * 16 + j
                        dQ[i, warp * 64 + t] += (K[k_row,t] - Q[i,t]) * dS[i,k_row]/S[i,k_row]
    for tidx in range(256):
        warp = tidx // 32
        lane = tidx % 32
        for ni in range(4):
            acc_dq = []
            for ri in range(8):
                i = (lane >> 2) + 4 * (ri & 0x2)
                j = lane % 4 * 2 + 2 * (ri & 0x4) + (ri & 0x1)
                acc_dq.append(dQ[i, warp * 64 + ni * 16 + j].item())
            print(f"tidx= {tidx}, acc_dp[0][{ni}]={acc_dq}")


if __name__ == "__main__":
    test_dQ()
