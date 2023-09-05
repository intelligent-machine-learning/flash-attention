# for developing

cmake_minimum_required(VERSION 3.10)
project(flash_attn LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    # set(CMAKE_BUILD_TYPE RELEASE)
    set(CMAKE_BUILD_TYPE DEBUG)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
    # set(CMAKE_CUDA_ARCHITECTURES 75 80)
endif()

execute_process(COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)"
    TIMEOUT 5
    OUTPUT_VARIABLE torch_cmake_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(COMMAND python -c "import pybind11;print(pybind11.get_cmake_dir())"
    TIMEOUT 5
    OUTPUT_VARIABLE pybind11_cmake_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${torch_cmake_path} ${pybind11_cmake_path})

find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)

message(STATUS "*** torch cmake path: ${TORCH_CAMKE_PATH}")
message(STATUS "*** TORCH_INCLUDE_DIRS=" ${TORCH_INCLUDE_DIRS})
message(STATUS "*** TORCH_LIBRARIES=" ${TORCH_LIBRARIES})
message(STATUS "*** Python3_INCLUDE_DIRS=" ${Python3_INCLUDE_DIRS})
message(STATUS "*** Python3_LIBRARIES=" ${Python3_LIBRARIES})
message(STATUS "*** Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")
message(STATUS "*** Found pybind11 v${pybind11_VERSION}: ${pybind11_LIBRARIES}")
message(STATUS "*** CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "*** CUDA_LIBRARIES: ${CUDA_LIBRARIES}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

execute_process(COMMAND python -c "import torch;print(torch.__path__[0])"
    TIMEOUT 5
    OUTPUT_VARIABLE torch_install_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(torch_lib_dir "${torch_install_path}/lib")
message(STATUS "*** torch_lib_dir: ${torch_lib_dir}")

link_directories(
    ${torch_lib_dir}
)

include_directories(
    src
    cutlass/include
)

# set(CUDA_SEPARABLE_COMPILATION ON)
# set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# set_target_properties(${PROJECT_NAME}
#    PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON
#        CUDA_RESOLVE_DEVICE_SYMBOLS ON
# )

# add_library(gemm src/fmha/gemm.cu)
# target_compile_options(gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#     -g -G
#     # -O0
#     -U__CUDA_NO_HALF_OPERATORS__
#     -U__CUDA_NO_HALF_CONVERSIONS__
#     --expt-relaxed-constexpr
#     --expt-extended-lambda
#     --use_fast_math
#     --ptxas-options=-v
#     # --ptxas-options=-O0
#     -lineinfo
#     # -dlink
#     -rdc=true
#     >)

# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
pybind11_add_module(flash_attn_cuda
    fmha_api.cpp
    src/fmha_fwd_hdim32.cu
    src/fmha_fwd_hdim64.cu
    src/fmha_fwd_hdim128.cu
    src/fmha_bwd_hdim32.cu
    src/fmha_bwd_hdim64.cu
    src/fmha_bwd_hdim128.cu
    src/fmha_block_fprop_fp16_kernel.sm80.cu
    src/fmha_block_dgrad_fp16_kernel_loop.sm80.cu
    src/l2attn_fwd_hdim32.cu
    src/l2attn_fwd_hdim64.cu
    src/l2attn_fwd_hdim128.cu
    src/l2attn_bwd_hdim32.cu
    src/l2attn_bwd_hdim64.cu
    src/l2attn_bwd_hdim128.cu
)
target_link_libraries(flash_attn_cuda PRIVATE
    # ${TORCH_LIBRARIES}
    -lc10
    -lc10_cuda
    -ltorch
    -ltorch_cpu
    -ltorch_cuda
    -ltorch_python
    ${CUDA_LIBRARIES}
    # gemm
)
target_compile_options(flash_attn_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -g -G
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --use_fast_math
    --ptxas-options=-v
    -lineinfo
    # -dlink
    # -rdc=true
    >)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math --ptxas-options=-v -lineinfo")
