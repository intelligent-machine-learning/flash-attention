// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "l2attn_bwd_launch_template.h"

void run_l2attn_bwd_hdim128(FMHA_dgrad_params &params, cudaStream_t stream, const bool configure) {
    FP16_SWITCH(params.is_bf16, ({
        using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 8, 0x100u, elem_type>;
        run_l2attn_bwd_loop<Kernel_traits>(params, stream, configure);
    }));
}