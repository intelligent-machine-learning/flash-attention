// /***************************************************************************************************
//  * Copyright (c) 2022, Tri Dao.
//  * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
//  * 
//  * Redistribution and use in source and binary forms, with or without
//  * modification, are permitted provided that the following conditions are met:
//  *     * Redistributions of source code must retain the above copyright
//  *       notice, this list of conditions and the following disclaimer.
//  *     * Redistributions in binary form must reproduce the above copyright
//  *       notice, this list of conditions and the following disclaimer in the
//  *       documentation and/or other materials provided with the distribution.
//  *     * Neither the name of the NVIDIA CORPORATION nor the
//  *       names of its contributors may be used to endorse or promote products
//  *       derived from this software without specific prior written permission.
//  * 
//  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//  * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//  * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
//  * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//  *
//  ******************************************************************************/

#pragma once

#include "fmha_kernel.h"
#include <fmha/kernel_traits.h>
#include <fmha/utils.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename elem_type_=__half>
struct L2Dist_Q_K {
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;

    using elem_type = elem_type_;

    static constexpr bool SHARE_SMEM_FOR_K_AND_V = Kernel_traits::SHARE_SMEM_FOR_K_AND_V;
    static constexpr bool V_IN_REGS = Kernel_traits::V_IN_REGS;
    static_assert(V_IN_REGS || !SHARE_SMEM_FOR_K_AND_V);

    static constexpr int SMEM_OFFSET_V = Smem_tile_q::BYTES_PER_TILE + (SHARE_SMEM_FOR_K_AND_V ? 0 : Smem_tile_k::BYTES_PER_TILE);
    static_assert(Smem_tile_v::BYTES_PER_TILE == (int) Smem_tile_k::BYTES_PER_TILE);
    static constexpr int SMEM_OFFSET_O = SMEM_OFFSET_V + Smem_tile_v::BYTES_PER_TILE;
    static constexpr int SMEM_OFFSET_SOFTMAX = SMEM_OFFSET_O + Smem_tile_o::BYTES_PER_TILE;
    
    static constexpr int SMEM_BYTES_SOFTMAX = Cta_tile_p::M * Cta_tile_p::WARPS_N * sizeof(float) * 2;

    // If V_IN_REGS and SHARE_SMEM_FOR_K_AND_V:      Q | K/V | O | SOFTMAX
    // If !V_IN_REGS (then !SHARE_SMEM_FOR_K_AND_V): Q | K   | V | O | SOFTMAX
    static constexpr int SMEM_BYTES = Smem_tile_q::BYTES_PER_TILE
                                    + (SHARE_SMEM_FOR_K_AND_V ? 1 : 2) * Smem_tile_k::BYTES_PER_TILE 
                                    + Smem_tile_o::BYTES_PER_TILE + SMEM_BYTES_SOFTMAX;

    static constexpr int WARP_SIZE = Cta_tile_p::THREADS_PER_WARP;  // 32

    static constexpr int ELTS_PER_WARP_Q = Mma_tile_p::M_PER_MMA * Mma_tile_p::K_PER_MMA;  // 16*16 = 256

    static constexpr int ELTS_PER_WARP_ROW_Q = ELTS_PER_WARP_Q * Mma_tile_p::MMAS_K;  // 256*4 = 1024

    static constexpr int ELTS_PER_CTA_ROW_Q = ELTS_PER_WARP_ROW_Q * Cta_tile_p::WARPS_M;  // 1024*1 = 1024

    static constexpr int ELTS_PER_WARP_K = Mma_tile_p::N_PER_MMA * Mma_tile_p::K_PER_MMA;  // 16*16 = 256

    static constexpr int ELTS_PER_WARP_ROW_K = ELTS_PER_WARP_K * Mma_tile_p::MMAS_K;  // 256*4 = 1024

    static constexpr int ELTS_PER_CTA_ROW_K = ELTS_PER_WARP_ROW_K * Cta_tile_p::WARPS_N;  // 1024*4 = 4096

    static constexpr int ELTS_PER_ROW_Q_K = Cta_tile_p::K;  // 64 TODO: should consider shared memory

    // number of elements loaded by each thread
    static constexpr int ELTS_PER_THREAD = ELTS_PER_ROW_Q_K / WARP_SIZE; // 2

    static constexpr int ELTS_PER_ROW_P = Mma_tile_p::N_PER_MMA * Cta_tile_p::WARPS_N;  // 16*8 = 128

    __device__ inline L2Dist_Q_K(char * smem_, const int tidx)
        : smem_q(smem_, tidx), smem_k(smem_ + Smem_tile_q::BYTES_PER_TILE, tidx) {
        smem_q_ptr_ = (half *)smem_;
        smem_k_ptr_ = (half *)(smem_ + Smem_tile_q::BYTES_PER_TILE);
        // if (blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && tidx==0) {
        //     printf("*** smem size (KB): Q=%d, K/V=%d, O=%d, S=%dB, total=%d\n", 
        //         Smem_tile_q::BYTES_PER_TILE/1024, (SHARE_SMEM_FOR_K_AND_V ? 1 : 2) * Smem_tile_k::BYTES_PER_TILE / 1024,
        //         Smem_tile_o::BYTES_PER_TILE/1024, SMEM_BYTES_SOFTMAX, SMEM_BYTES/1024
        //     );
        // }
    }

    template<typename Acc, int M, int N>
    __device__ inline void operator()(Acc (&acc_p)[M][N], int tidx, int l) {
        // Do this part of P = l2(Q - K).
        int warp_id = tidx / WARP_SIZE;
        int lane_id = tidx % WARP_SIZE;
        half *warp_row_base_q = smem_q_ptr_;
        half *warp_row_base_k = smem_k_ptr_ + warp_id * ELTS_PER_WARP_ROW_K/*1024*/;

        // TODO: solve bank conflicts
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) {
            #pragma unroll
            for (int mi = 0; mi < M; ++mi) {
                half *this_warp_q = warp_row_base_q + mi * ELTS_PER_CTA_ROW_Q/*1024*/ 
                                            + ki * Mma_tile_p::K_PER_MMA/*16*/;
                #pragma unroll
                for (int ni = 0; ni < N; ++ni) {
                    half *this_warp_k = warp_row_base_k + ni * ELTS_PER_CTA_ROW_K/*4096*/
                                            + ki * Mma_tile_p::K_PER_MMA/*16*/;
                    /**
                     * corresponding to mma.m16n8k16
                     * q:16x16, k:8*16
                    */
                    #pragma unroll
                    for (int di = 0; di < 4; ++di) {
                        int row = di < 2 ? lane_id >> 2 : (lane_id >> 2) + 8;
                        int col = lane_id % 4 * 2 + (di & 0x1);
                        half *q_row = this_warp_q + row * ELTS_PER_ROW_Q_K/*64*/;
                        half *k_row = this_warp_k + col * ELTS_PER_ROW_Q_K;
                        cal_dist_pow_sum(&acc_p[mi][ni].elt(di), q_row, k_row, Mma_tile_p::K_PER_MMA/*16*/);
                        // float temp = acc_p[mi][ni].elt(di);
                        // if (blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && tidx==0 && l==0 && ki==3) {
                        //     printf("cal_l2: tidx= %d lane_id= %d l= %d, ki= %d acc_p[%d][%d].elt(%d)=%.3f, q_row[%d]={%.3f,%.3f,%3.f,%.3f,%.3f,%.3f,%3.f,%.3f}, k_row[%d]={%.3f,%.3f,%3.f,%.3f,%.3f,%.3f,%3.f,%.3f}\n",
                        //         tidx,lane_id,l,ki,mi,ni,di,temp,row,
                        //         __half2float(q_row[0]),__half2float(q_row[1]),__half2float(q_row[2]),__half2float(q_row[3]),
                        //         __half2float(q_row[12]),__half2float(q_row[13]),__half2float(q_row[14]),__half2float(q_row[15]),col,
                        //         __half2float(k_row[0]),__half2float(k_row[1]),__half2float(k_row[2]),__half2float(k_row[3]),
                        //         __half2float(k_row[12]),__half2float(k_row[13]),__half2float(k_row[14]),__half2float(k_row[15])
                        //     );
                        // }
                    }
                    /**
                     * for next 8*16 K matrix, corresponding to next mma.m16n8k16
                     * q:16x16, k:8*16
                    */
                    this_warp_k += 8 * ELTS_PER_ROW_Q_K;  // 8 * 64
                    #pragma unroll
                    for (int di = 0; di < 4; ++di) {
                        int row = di < 2 ? lane_id >> 2 : (lane_id >> 2) + 8; // shit!!! the priority of "+" is greater than ">>"
                        int col = lane_id % 4 * 2 + (di & 0x1);
                        half *q_row = this_warp_q + row * ELTS_PER_ROW_Q_K;
                        half *k_row = this_warp_k + col * ELTS_PER_ROW_Q_K;
                        cal_dist_pow_sum(&acc_p[mi][ni].elt(4 + di), q_row, k_row, Mma_tile_p::K_PER_MMA/*16*/);
                    }
                }
            }
        }

        #pragma unroll
        for (int mi = 0; mi < M; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < N; ++ni) {
                for (int di = 0; di < 8; ++di) {
                    acc_p[mi][ni].elt(di) = -std::sqrt(acc_p[mi][ni].elt(di));
                    // asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(acc_p[mi][ni].elt(di)), "+f"(acc_p[mi][ni].elt(di))); // Little acceleration effect
                    // acc_p[mi][ni].elt(di) = -acc_p[mi][ni].elt(di);
                }
            }
        }
    }

    half *smem_q_ptr_, *smem_k_ptr_;
    Smem_tile_q smem_q;
    Smem_tile_k smem_k;

private:
    __device__ inline void cal_dist_pow_sum(float *d, const half *a, const half *b, int k){
        for (int i = 0; i < k; ++i) {
            // TODO: use cvt instruct to speed up
            *d += __half2float((a[i] - b[i])) * __half2float((a[i] - b[i]));
        }
    }
};

template<typename Kernel_traits>
constexpr size_t get_dynamic_smem_size_for_l2attn() {
    return L2Dist_Q_K<Kernel_traits>::SMEM_BYTES;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha

