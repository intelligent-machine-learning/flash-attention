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
#include "fmha/gemm.h"

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename elem_type_=__half>
struct L2Dist_dQ_dK {
    // using elem_type = elem_type_;
    using elem_type = __half;
    
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    using Cta_tile_dq = typename Kernel_traits::Cta_tile_o;
    using Cta_tile_dk =
        fmha::Cta_tile_extd<Cta_tile_p::N, Cta_tile_p::K, Cta_tile_p::M, Cta_tile_p::WARPS_N, 1, Cta_tile_p::WARPS_M>;
    static_assert(Cta_tile_dk::M == 512 ||  Cta_tile_dk::M == 256 || Cta_tile_dk::M == 128);
    static_assert(Cta_tile_dk::N == 16 || Cta_tile_dk::N == 32 || Cta_tile_dk::N == 64 || Cta_tile_dk::N == 128);
    static_assert(Cta_tile_dk::K == 16);

    using Mma_tile_dq = fmha::Hmma_tile<Cta_tile_dq>;
    using Mma_tile_dk = fmha::Hmma_tile<Cta_tile_dk>;

    static constexpr int WARP_SIZE = Cta_tile_dq::THREADS_PER_WARP;  // 32

    ///////////////////////////////////////////////////
    // for dK
    static constexpr int DQ_ELTS_PER_ROW_P = Cta_tile_dq::K;  // 256
    static constexpr int DQ_ELTS_PER_WARP_ROW_P = DQ_ELTS_PER_ROW_P * Mma_tile_dq::M_PER_MMA;  // 256*16 = 4096
    static constexpr int DQ_ELTS_PER_CTA_ROW_P = DQ_ELTS_PER_WARP_ROW_P * Cta_tile_dq::WARPS_M;  // 4096*1 = 4096

    static constexpr int DQ_ELTS_PER_ROW_K = Cta_tile_dq::N;  // 64
    static constexpr int DQ_ELTS_PER_WARP_ROW_K = DQ_ELTS_PER_ROW_K * Mma_tile_dq::K_PER_MMA;  // 64*16 = 1024
    static constexpr int DQ_ELTS_PER_CTA_ROW_K = DQ_ELTS_PER_WARP_ROW_K * Cta_tile_dq::WARPS_K;  // 1024*8 = 8192

    static constexpr int DQ_ELTS_PER_ROW_Q = Cta_tile_dq::N;  // 64
    static constexpr int DQ_ELTS_PER_WARP_ROW_Q = DQ_ELTS_PER_ROW_Q * Mma_tile_dq::M_PER_MMA;  // 64*16 = 1024
    static constexpr int DQ_ELTS_PER_CTA_ROW_Q = DQ_ELTS_PER_WARP_ROW_Q * Cta_tile_dq::WARPS_M;  // 1024*1 = 1024
    ///////////////////////////////////////////////////

    ///////////////////////////////////////////////////
    // for dK
    static constexpr int DK_ELTS_PER_ROW_P = Cta_tile_dk::M;  // 512
    static constexpr int DK_ELTS_PER_WARP_ROW_P = DK_ELTS_PER_ROW_P * Mma_tile_dk::K_PER_MMA;  // 512*16 = 8192
    static constexpr int DK_ELTS_PER_CTA_ROW_P = DK_ELTS_PER_WARP_ROW_P * Cta_tile_dk::WARPS_K;  // 8192*1 = 8192
    static constexpr int DK_ELTS_PER_CTA_COL_P = Mma_tile_dk::M_PER_MMA * Cta_tile_dk::WARPS_M;  // 16*8=128

    static constexpr int DK_ELTS_PER_ROW_Q = Cta_tile_dk::N;  // 64
    static constexpr int DK_ELTS_PER_WARP_ROW_Q = DK_ELTS_PER_ROW_Q * Mma_tile_dk::K_PER_MMA;  // 64*16 = 1024
    static constexpr int DK_ELTS_PER_CTA_ROW_Q = DK_ELTS_PER_WARP_ROW_Q * Cta_tile_dk::WARPS_K;  // 1024*1 = 1024
    static constexpr int DK_ELTS_PER_CTA_COL_Q = Mma_tile_dk::N_PER_MMA * Cta_tile_dk::WARPS_N;  // 16*1=16

    static constexpr int DK_ELTS_PER_ROW_K = Cta_tile_dk::N;  // 64
    static constexpr int DK_ELTS_PER_WARP_ROW_K = DK_ELTS_PER_ROW_K * Mma_tile_dk::M_PER_MMA;  // 64*16 = 1024
    static constexpr int DK_ELTS_PER_CTA_ROW_K = DK_ELTS_PER_WARP_ROW_K * Cta_tile_dk::WARPS_M;  // 1024*8 = 8192
    static constexpr int DK_ELTS_PER_CTA_COL_K = Mma_tile_dk::N_PER_MMA * Cta_tile_dk::WARPS_N;  // 16*1=16

    ///////////////////////////////////////////////////

    __device__ inline L2Dist_dQ_dK(char *smem_q_ptr, char *smem_k_ptr, char *smem_p_ptr, char *smem_dp_ptr, const int tidx) {
        smem_q_ptr_ = (elem_type *)smem_q_ptr;
        smem_k_ptr_ = (elem_type *)smem_k_ptr;
        smem_p_ptr_ = (elem_type *)smem_p_ptr;
        smem_dp_ptr_ = (elem_type *)smem_dp_ptr;

// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
//         if (std::is_same<elem_type, __nv_bfloat16>::value) {
//             to_float_func_ = [](const __nv_bfloat16& a) { return __bfloat162float(a); };
//         } else {
//             to_float_func_ = [](const __half& a) { return __half2float(a); };
//         }
// #else
//         assert((std::is_same<elem_type, __half>::value));
//         to_float_func_ = [](const __half& a) { return __half2float(a); };
// #endif
        assert((std::is_same<elem_type, __half>::value));
        to_float_func_ = [](const __half& a) { return __half2float(a); };
    }

    template<typename Acc, int M, int N>
    __device__ inline void cal_dq(Acc (&acc_dq)[M][N], int tidx, int loop_step_idx, int l) {
        // Do this part of dQ = (K-Q)*dP/P.
        int warp = tidx / WARP_SIZE;
        int lane = tidx % WARP_SIZE;

        #pragma unroll
        for( int ki = 0; ki < Mma_tile_dq::MMAS_K; ++ki ) {  // K=4
            #pragma unroll
            for (int mi = 0; mi < M; ++mi) {  // M=1
                // elem_type *this_warp_p = smem_p_ptr_ + mi * DK_ELTS_PER_CTA_COL_P/*128*/
                //                         + ki * (Cta_tile_dq::WARPS_K/*8*/ * Mma_tile_dq::K_PER_MMA/*16*/)
                //                         + warp * Mma_tile_dq::K_PER_MMA/*16*/;
                elem_type *this_warp_dp = smem_dp_ptr_ + mi * DQ_ELTS_PER_CTA_ROW_P/*4096*/
                                        + ki * (Cta_tile_dq::WARPS_K/*8*/ * Mma_tile_dq::K_PER_MMA/*16*/)
                                        + warp * Mma_tile_dq::K_PER_MMA/*16*/;
                #pragma unroll
                for (int ni = 0; ni < N; ++ni) {  // N=4
                    elem_type *this_warp_q = smem_q_ptr_ + mi * DQ_ELTS_PER_CTA_ROW_Q/*1024*/
                                            + ni * (Cta_tile_dq::WARPS_N/*1*/ * Mma_tile_dq::N_PER_MMA/*16*/);
                    elem_type *this_warp_k = smem_k_ptr_ + ki * DQ_ELTS_PER_CTA_ROW_K/*8192*/
                                            + ni * (Cta_tile_dq::WARPS_N/*1*/ * Mma_tile_dq::N_PER_MMA/*16*/)
                                            + warp * DQ_ELTS_PER_WARP_ROW_K/*1024*/;
                    /**
                     * cal dQ:
                     * for Cta_tile_dq:
                     * dQ_{it} = \sum_{j=0}^{255} \frac{K_{jt}-Q_{it}}{P_{ij}}*dP_{ij}, i \in [0,16), t \in [0,64), j \in [0,256)
                     * 
                     * 
                     * ref: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
                     * for 0 <= di < 8:
                     * ri=0: row = lane/4,      col = lane%4*2
                     * ri=1: row = lane/4,      col = lane%4*2 + 1
                     * ri=2: row = lane/4 + 8,  col = lane%4*2
                     * ri=3: row = lane/4 + 8,  col = lane%4*2 + 1
                     * ri=4: row = lane/4,      col = lane%4*2 + 8
                     * ri=5: row = lane/4,      col = lane%4*2 + 8 + 1
                     * ri=6: row = lane/4 + 8,  col = lane%4*2 + 8
                     * ri=7: row = lane/4 + 8,  col = lane%4*2 + 8 + 1
                    */
                    #pragma unroll
                    for (int ri = 0; ri < 8; ++ri) {
                        int row_in_warp = (lane >> 2) + 4 * (ri & 0x2);  // i
                        int col_in_warp = lane % 4 * 2 + 2 * (ri & 0x4) + (ri & 0x1);  // t
                        // const elem_type *p_row = this_warp_p + row_in_warp * DQ_ELTS_PER_ROW_P/*256*/;
                        const elem_type *dp_row = this_warp_dp + row_in_warp * DQ_ELTS_PER_ROW_P/*256*/;
                        // Attention: we use K^T to calculate, so the column `col_in_warp` is needed
                        const elem_type *k_col = this_warp_k + col_in_warp;
                        const elem_type &q_it = *(this_warp_q + row_in_warp * DQ_ELTS_PER_ROW_Q/*64*/ + col_in_warp);
                        float *dQ_it = &acc_dq[mi][ni].elt(ri);
                        cal_dQ_it(dQ_it, /*p_row*/nullptr, dp_row, k_col, q_it, loop_step_idx, l);
                        // if (blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && loop_step_idx==0 && l==0 && ki==0 && ni==1) {
                        //     printf("*** tidx= %d warp= %d lane= %d ki= %d mi= %d ni= %d ri=%d,i=%d,t=%d, acc_dq[%d][%d].elt(%d)=%f, q_it=%f\n",
                        //         tidx,warp,lane,ki,mi,ni,ri,row_in_warp,col_in_warp,mi,ni,ri,*dQ_it,q_it_f32
                        //     );
                        //     printf("*** tidx= %d lane=%d ki=%d mi=%d ni=%d ri=%d,i=%d,t=%d, "\
                        //         "p_row[%d]={%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f}\n",
                        //         tidx,lane,ki,mi,ni,ri,row_in_warp,col_in_warp,row_in_warp,
                        //         __half2float(p_row[0]),__half2float(p_row[1]),__half2float(p_row[2]),__half2float(p_row[3]),
                        //         __half2float(p_row[4]),__half2float(p_row[5]),__half2float(p_row[6]),__half2float(p_row[7]),
                        //         __half2float(p_row[8]),__half2float(p_row[9]),__half2float(p_row[10]),__half2float(p_row[11]),
                        //         __half2float(p_row[12]),__half2float(p_row[13]),__half2float(p_row[14]),__half2float(p_row[15])
                        //     );
                        //     printf("*** tidx= %d lane=%d ki=%d mi=%d ni=%d ri=%d,i=%d,t=%d, "\
                        //         "dp_row[%d]={%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f}\n",
                        //         tidx,lane,ki,mi,ni,ri,row_in_warp,col_in_warp,row_in_warp,
                        //         __half2float(dp_row[0]),__half2float(dp_row[1]),__half2float(dp_row[2]),__half2float(dp_row[3]),
                        //         __half2float(dp_row[4]),__half2float(dp_row[5]),__half2float(dp_row[6]),__half2float(dp_row[7]),
                        //         __half2float(dp_row[8]),__half2float(dp_row[9]),__half2float(dp_row[10]),__half2float(dp_row[11]),
                        //         __half2float(dp_row[12]),__half2float(dp_row[13]),__half2float(dp_row[14]),__half2float(dp_row[15])
                        //     );
                        //     printf("*** tidx= %d lane=%d ki=%d mi=%d ni=%d ri=%d,i=%d,t=%d, "\
                        //         "k_col[%d]={%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f}\n",
                        //         tidx,lane,ki,mi,ni,ri,row_in_warp,col_in_warp,col_in_warp,
                        //         __half2float(k_col[0*DQ_ELTS_PER_ROW_K]),__half2float(k_col[1*DQ_ELTS_PER_ROW_K]),__half2float(k_col[2*DQ_ELTS_PER_ROW_K]),__half2float(k_col[3*DQ_ELTS_PER_ROW_K]),
                        //         __half2float(k_col[4*DQ_ELTS_PER_ROW_K]),__half2float(k_col[5*DQ_ELTS_PER_ROW_K]),__half2float(k_col[6*DQ_ELTS_PER_ROW_K]),__half2float(k_col[7*DQ_ELTS_PER_ROW_K]),
                        //         __half2float(k_col[8*DQ_ELTS_PER_ROW_K]),__half2float(k_col[9*DQ_ELTS_PER_ROW_K]),__half2float(k_col[10*DQ_ELTS_PER_ROW_K]),__half2float(k_col[11*DQ_ELTS_PER_ROW_K]),
                        //         __half2float(k_col[12*DQ_ELTS_PER_ROW_K]),__half2float(k_col[13*DQ_ELTS_PER_ROW_K]),__half2float(k_col[14*DQ_ELTS_PER_ROW_K]),__half2float(k_col[15*DQ_ELTS_PER_ROW_K])
                        //     );
                        // }
                    }
                }
            }
        }
    }

    template<typename Acc, int M, int N>
    __device__ inline void cal_dk(Acc (&acc_dk)[M][N], int tidx, int loop_step_idx, int l) {
        // Do this part of dK = (Q-K)*dP/P.
        int warp = tidx / WARP_SIZE;
        int lane = tidx % WARP_SIZE;

        #pragma unroll
        for( int ki = 0; ki < Mma_tile_dk::MMAS_K; ++ki ) {  // K=1
            #pragma unroll
            for (int mi = 0; mi < M; ++mi) {  // M=4
                // elem_type *this_warp_p = smem_p_ptr_ + ki * DK_ELTS_PER_CTA_ROW_P/*8192*/
                //                         + mi * DK_ELTS_PER_CTA_COL_P/*128*/
                //                         + warp * Mma_tile_dk::M_PER_MMA/*16*/;
                elem_type *this_warp_dp = smem_dp_ptr_ + ki * DK_ELTS_PER_CTA_ROW_P/*8192*/
                                        + mi * DK_ELTS_PER_CTA_COL_P/*128*/
                                        + warp * Mma_tile_dk::M_PER_MMA/*16*/;
                #pragma unroll
                for (int ni = 0; ni < N; ++ni) {  // N=4
                    elem_type *this_warp_q = smem_q_ptr_ + ki * DK_ELTS_PER_CTA_ROW_Q/*1024*/
                                            + ni * DK_ELTS_PER_CTA_COL_Q/*16*/;
                    elem_type *this_warp_k = smem_k_ptr_ + mi * DK_ELTS_PER_CTA_ROW_K/*8192*/
                                            + ni * DK_ELTS_PER_CTA_COL_K/*16*/
                                            + warp * DK_ELTS_PER_WARP_ROW_K/*1024*/;
                    /**
                     * cal dK:
                     * for Cta_tile_dk:
                     * dK_{jt} = \sum_{i=0}^{15} \frac{Q_{it}-K_{jt}}{P_{ij}}*dP_{ij}, i \in [0,16), t \in [0,64), j \in [0,256)
                     * 
                     * ref: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
                     * for 0 <= di < 8:
                     * ri=0: row = lane/4,      col = lane%4*2
                     * ri=1: row = lane/4,      col = lane%4*2 + 1
                     * ri=2: row = lane/4 + 8,  col = lane%4*2
                     * ri=3: row = lane/4 + 8,  col = lane%4*2 + 1
                     * ri=4: row = lane/4,      col = lane%4*2 + 8
                     * ri=5: row = lane/4,      col = lane%4*2 + 8 + 1
                     * ri=6: row = lane/4 + 8,  col = lane%4*2 + 8
                     * ri=7: row = lane/4 + 8,  col = lane%4*2 + 8 + 1
                    */
                    #pragma unroll
                    for (int ri = 0; ri < 8; ++ri) {
                        int row_in_warp = (lane >> 2) + 4 * (ri & 0x2);  // j
                        int col_in_warp = lane % 4 * 2 + 2 * (ri & 0x4) + (ri & 0x1);  // t
                        // const elem_type *p_col = this_warp_p + row_in_warp;
                        const elem_type *dp_col = this_warp_dp + row_in_warp;
                        const elem_type *q_col = this_warp_q + col_in_warp;
                        const elem_type &k_jt = *(this_warp_k + row_in_warp * DK_ELTS_PER_ROW_K/*64*/ + col_in_warp);
                        float *dK_jt = &acc_dk[mi][ni].elt(ri);
                        cal_dK_jt(dK_jt, /*p_col*/nullptr, dp_col, q_col, k_jt, loop_step_idx, l);
                        // if (blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && loop_step_idx==0 && l==0 && mi==0 && ni==0) {
                        //     printf("*** tidx= %d warp= %d lane= %d ki= %d mi= %d ni= %d ri=%d,j=%d,t=%d, acc_dk[%d][%d].elt(%d)=%f, k_jt=%f\n",
                        //         tidx,warp,lane,ki,mi,ni,ri,row_in_warp,col_in_warp,mi,ni,ri,*dK_jt,__half2float(k_jt)
                        //     );
                        //     printf("*** tidx= %d lane=%d ki=%d mi=%d ni=%d ri=%d,j=%d,t=%d, "\
                        //         "p_col[%d]={%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f}\n",
                        //         tidx,lane,ki,mi,ni,ri,row_in_warp,col_in_warp,row_in_warp,
                        //         __half2float(p_col[0*DK_ELTS_PER_ROW_P]),__half2float(p_col[1*DK_ELTS_PER_ROW_P]),__half2float(p_col[2*DK_ELTS_PER_ROW_P]),__half2float(p_col[3*DK_ELTS_PER_ROW_P]),
                        //         __half2float(p_col[4*DK_ELTS_PER_ROW_P]),__half2float(p_col[5*DK_ELTS_PER_ROW_P]),__half2float(p_col[6*DK_ELTS_PER_ROW_P]),__half2float(p_col[7*DK_ELTS_PER_ROW_P]),
                        //         __half2float(p_col[8*DK_ELTS_PER_ROW_P]),__half2float(p_col[9*DK_ELTS_PER_ROW_P]),__half2float(p_col[10*DK_ELTS_PER_ROW_P]),__half2float(p_col[11*DK_ELTS_PER_ROW_P]),
                        //         __half2float(p_col[12*DK_ELTS_PER_ROW_P]),__half2float(p_col[13*DK_ELTS_PER_ROW_P]),__half2float(p_col[14*DK_ELTS_PER_ROW_P]),__half2float(p_col[15*DK_ELTS_PER_ROW_P])
                        //     );
                        //     printf("*** tidx= %d lane=%d ki=%d mi=%d ni=%d ri=%d,j=%d,t=%d, "\
                        //         "dp_col[%d]={%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f}\n",
                        //         tidx,lane,ki,mi,ni,ri,row_in_warp,col_in_warp,row_in_warp,
                        //         __half2float(dp_col[0*DK_ELTS_PER_ROW_P]),__half2float(dp_col[1*DK_ELTS_PER_ROW_P]),__half2float(dp_col[2*DK_ELTS_PER_ROW_P]),__half2float(dp_col[3*DK_ELTS_PER_ROW_P]),
                        //         __half2float(dp_col[4*DK_ELTS_PER_ROW_P]),__half2float(dp_col[5*DK_ELTS_PER_ROW_P]),__half2float(dp_col[6*DK_ELTS_PER_ROW_P]),__half2float(dp_col[7*DK_ELTS_PER_ROW_P]),
                        //         __half2float(dp_col[8*DK_ELTS_PER_ROW_P]),__half2float(dp_col[9*DK_ELTS_PER_ROW_P]),__half2float(dp_col[10*DK_ELTS_PER_ROW_P]),__half2float(dp_col[11*DK_ELTS_PER_ROW_P]),
                        //         __half2float(dp_col[12*DK_ELTS_PER_ROW_P]),__half2float(dp_col[13*DK_ELTS_PER_ROW_P]),__half2float(dp_col[14*DK_ELTS_PER_ROW_P]),__half2float(dp_col[15*DK_ELTS_PER_ROW_P])
                        //     );
                        //     printf("*** tidx= %d lane=%d ki=%d mi=%d ni=%d ri=%d,j=%d,t=%d, "\
                        //         "q_col[%d]={%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f}\n",
                        //         tidx,lane,ki,mi,ni,ri,row_in_warp,col_in_warp,col_in_warp,
                        //         __half2float(q_col[0*DK_ELTS_PER_ROW_Q]),__half2float(q_col[1*DK_ELTS_PER_ROW_Q]),__half2float(q_col[2*DK_ELTS_PER_ROW_Q]),__half2float(q_col[3*DK_ELTS_PER_ROW_Q]),
                        //         __half2float(q_col[4*DK_ELTS_PER_ROW_Q]),__half2float(q_col[5*DK_ELTS_PER_ROW_Q]),__half2float(q_col[6*DK_ELTS_PER_ROW_Q]),__half2float(q_col[7*DK_ELTS_PER_ROW_Q]),
                        //         __half2float(q_col[8*DK_ELTS_PER_ROW_Q]),__half2float(q_col[9*DK_ELTS_PER_ROW_Q]),__half2float(q_col[10*DK_ELTS_PER_ROW_Q]),__half2float(q_col[11*DK_ELTS_PER_ROW_Q]),
                        //         __half2float(q_col[12*DK_ELTS_PER_ROW_Q]),__half2float(q_col[13*DK_ELTS_PER_ROW_Q]),__half2float(q_col[14*DK_ELTS_PER_ROW_Q]),__half2float(q_col[15*DK_ELTS_PER_ROW_Q])
                        //     );
                        // }
                    }
                }
            }
        }
    }

    elem_type *smem_q_ptr_, *smem_k_ptr_;
    elem_type *smem_p_ptr_, *smem_dp_ptr_;

private:
    __device__ inline void cal_dQ_it(float *dQ_it, const elem_type *p_row, const elem_type *dp_row, 
                                                        const elem_type *k_col, const elem_type &q_it, int loop_step_idx, int l) {
        float q_it_f32 = to_float_func_(q_it);
        #pragma unroll
        for (int j = 0; j < Mma_tile_dq::K_PER_MMA/*16*/; ++j) {
            // TODO: don't use float
            // float p_ij = to_float_func_(p_row[j]);
            float dp_ij = to_float_func_(dp_row[j]);
            float k_jt = to_float_func_(k_col[j * DQ_ELTS_PER_ROW_K]);
            *dQ_it += (q_it_f32 - k_jt) * dp_ij;// / p_ij;  // TODO: assert(p_ij != 0)
        }
    }

    __device__ inline void cal_dK_jt(float *dK_jt, const elem_type *p_col, const elem_type *dp_col, 
                                                        const elem_type *q_col, const elem_type &k_jt, int loop_step_idx, int l) {
        float k_jt_f32 = to_float_func_(k_jt);
        #pragma unroll
        for (int i = 0; i < Mma_tile_dk::K_PER_MMA/*16*/; ++i) {
            // TODO: don't use float
            // float p_ij = to_float_func_(p_col[i * DK_ELTS_PER_ROW_P]);
            float dp_ij = to_float_func_(dp_col[i * DK_ELTS_PER_ROW_P]);
            float q_it = to_float_func_(q_col[i * DK_ELTS_PER_ROW_Q]);
            *dK_jt += (k_jt_f32 - q_it) * dp_ij; // / p_ij;  // TODO: assert(p_ij != 0)
        }
    }

    float(* to_float_func_)(const elem_type&);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha

