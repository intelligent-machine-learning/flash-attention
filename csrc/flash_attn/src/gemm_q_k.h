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
#include <fmha/gemm.h>
#include <fmha/utils.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits>
struct Gemm_Q_K_base {
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;
    using Fragment_q = typename Smem_tile_q::Fragment;
    using Fragment_k = typename Smem_tile_k::Fragment;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;

    static constexpr int SMEM_BYTES_SOFTMAX = Cta_tile_p::M * Cta_tile_p::WARPS_N * sizeof(float) * 2;

    __device__ inline Gemm_Q_K_base(char * smem_ptr_q, char * smem_ptr_k, const int tidx) 
        : smem_q(smem_ptr_q, tidx)
        , smem_k(smem_ptr_k, tidx) {

    }

    __device__ inline void load_q() {
        smem_q.load(frag_q[0], 0);
    }

    __device__ inline void reload_q() {
        smem_q.load(frag_q[0], 0);
    }

    Fragment_q frag_q[2][Mma_tile_p::MMAS_M];
    Smem_tile_q smem_q;
    Smem_tile_k smem_k;
};

template<typename Kernel_traits, bool K_in_regs, typename elem_type_=__half>
struct Gemm_Q_K : public Gemm_Q_K_base<Kernel_traits> {

    using Base = Gemm_Q_K_base<Kernel_traits>;
    using Smem_tile_o = typename Base::Smem_tile_o;
    using Smem_tile_q = typename Base::Smem_tile_q;
    using Smem_tile_k = typename Base::Smem_tile_k;
    using Fragment_k = typename Base::Fragment_k;
    using Mma_tile_p = typename Base::Mma_tile_p;
    using elem_type = elem_type_;

    static constexpr bool SHARE_SMEM_FOR_K_AND_V = Kernel_traits::SHARE_SMEM_FOR_K_AND_V;
    // If V is stored in shared memory, we can't load K using the same shared memory.
    static_assert(Kernel_traits::V_IN_REGS);

    static constexpr int SMEM_OFFSET_O = Smem_tile_q::BYTES_PER_TILE;
    static constexpr int SMEM_OFFSET_SOFTMAX = SMEM_OFFSET_O + Smem_tile_o::BYTES_PER_TILE;
    static constexpr int SMEM_OFFSET_V = Smem_tile_q::BYTES_PER_TILE + (SHARE_SMEM_FOR_K_AND_V ? 0 : Smem_tile_k::BYTES_PER_TILE);

    // Q | K / V
    //   | O | SOFTMAX
    static constexpr int SMEM_BYTES = Smem_tile_q::BYTES_PER_TILE 
                                    + std::max((SHARE_SMEM_FOR_K_AND_V ? 1 : 2) * Smem_tile_k::BYTES_PER_TILE,
                                               Smem_tile_o::BYTES_PER_TILE + Base::SMEM_BYTES_SOFTMAX);

    __device__ inline Gemm_Q_K(char * smem_, const int tidx) 
        : Base(smem_, smem_ + Smem_tile_q::BYTES_PER_TILE, tidx) {
    }

    __device__ inline void load_k(){
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) {
            Base::smem_k.load(frag_k[ki], ki);
        }
    }

    template<typename Acc, int M, int N>
    __device__ inline void operator()(Acc (&acc_p)[M][N]){
        // Do this part of P^T = (Q * K^T)^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            Base::smem_q.load(Base::frag_q[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
        }
        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
        }
    }

    __device__ inline void reload_k(){
        // Noop.
    }

    Fragment_k frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];
};


template<typename Kernel_traits, typename elem_type_>
struct Gemm_Q_K<Kernel_traits, false, elem_type_> : public Gemm_Q_K_base<Kernel_traits> {
    using Base = Gemm_Q_K_base<Kernel_traits>;
    using Smem_tile_o = typename Base::Smem_tile_o;
    using Smem_tile_q = typename Base::Smem_tile_q;
    using Smem_tile_k = typename Base::Smem_tile_k;
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;
    using Fragment_k = typename Base::Fragment_k;
    using Mma_tile_p = typename Base::Mma_tile_p;
    using elem_type = elem_type_;
    Fragment_k frag_k[2][Mma_tile_p::MMAS_N];

    static constexpr bool SHARE_SMEM_FOR_K_AND_V = Kernel_traits::SHARE_SMEM_FOR_K_AND_V;
    static constexpr bool V_IN_REGS = Kernel_traits::V_IN_REGS;
    static_assert(V_IN_REGS || !SHARE_SMEM_FOR_K_AND_V);

    static constexpr int SMEM_OFFSET_V = Smem_tile_q::BYTES_PER_TILE + (SHARE_SMEM_FOR_K_AND_V ? 0 : Smem_tile_k::BYTES_PER_TILE);
    static_assert(Smem_tile_v::BYTES_PER_TILE == (int) Smem_tile_k::BYTES_PER_TILE);
    static constexpr int SMEM_OFFSET_O = SMEM_OFFSET_V + Smem_tile_v::BYTES_PER_TILE;
    static constexpr int SMEM_OFFSET_SOFTMAX = SMEM_OFFSET_O + Smem_tile_o::BYTES_PER_TILE;

    // If V_IN_REGS and SHARE_SMEM_FOR_K_AND_V:      Q | K/V | O | SOFTMAX
    // If !V_IN_REGS (then !SHARE_SMEM_FOR_K_AND_V): Q | K   | V | O | SOFTMAX
    static constexpr int SMEM_BYTES = Smem_tile_q::BYTES_PER_TILE
                                    + (SHARE_SMEM_FOR_K_AND_V ? 1 : 2) * Smem_tile_k::BYTES_PER_TILE 
                                    + Smem_tile_o::BYTES_PER_TILE + Base::SMEM_BYTES_SOFTMAX;

    __device__ inline Gemm_Q_K(char * smem_, const int tidx) 
      : Base(smem_, smem_ + Smem_tile_q::BYTES_PER_TILE, tidx) {
    }

    __device__ inline void load_k(){
        Base::smem_k.load(frag_k[0], 0);
    }

    template<typename Acc, int M, int N>
    __device__ inline void operator()(Acc (&acc_p)[M][N]){
        // Do this part of P^T = (Q * K^T)^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            Base::smem_q.load(Base::frag_q[ki & 1], ki);
            Base::smem_k.load(frag_k[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1) & 1]);
        }
        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1) & 1]);
        }
    }

    __device__ inline void reload_k(){
        Base::smem_k.load(frag_k[0], 0);
    }
};

template<typename Kernel_traits>
constexpr size_t get_dynamic_smem_size(){
    return Gemm_Q_K<Kernel_traits, Kernel_traits::K_IN_REGS>::SMEM_BYTES;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha

