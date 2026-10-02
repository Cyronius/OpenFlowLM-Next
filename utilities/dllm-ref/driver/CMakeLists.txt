# dllm-ref: runs the open_dllm decode policies (src/open_dllm) on llama.cpp -- Phase 1 of
# .claude/plans/diffusion-lm-support.md. See main.cpp's header for what it does.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
#   cmake --build build --target dllm-ref
#
# build.cmd does exactly that from a VS 2022 developer environment.
#
# llama.cpp comes in as a subdirectory, from LLAMA_CPP_DIR: the pinned tree (bb4caa75) with
# the two local dllm patches applied. Only the targets dllm-ref needs are built
# (EXCLUDE_FROM_ALL), and llama.cpp's own tools, tests, examples and server stay off, as
# they do whenever llama.cpp is not the top-level project.
cmake_minimum_required(VERSION 3.16)
project(dllm_ref C CXX)

# A single-config generator (Ninja) with no build type would build ggml unoptimised.
get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Default: $DLLM_ROOT/llama.cpp-dllm, where PATCHES.md puts the patched tree.
set(_llama_default "")
if(DEFINED ENV{DLLM_ROOT})
    set(_llama_default "$ENV{DLLM_ROOT}/llama.cpp-dllm")
endif()
set(LLAMA_CPP_DIR "${_llama_default}" CACHE PATH
    "llama.cpp source tree: bb4caa75 with the dllm patches (LLAMA_ALLOW_NONCONSECUTIVE_POS, LLAMA_BLOCK_ROUTING)")
file(TO_CMAKE_PATH "${LLAMA_CPP_DIR}" _llama_dir)
if(NOT EXISTS "${_llama_dir}/include/llama.h")
    message(FATAL_ERROR "LLAMA_CPP_DIR='${LLAMA_CPP_DIR}' is not a llama.cpp tree (no include/llama.h)")
endif()

# ggml's own option, declared here too so -DGGML_VULKAN=ON is documented where it is used.
# The Vulkan backend needs the LunarG Vulkan SDK (glslc); OFF builds CPU-only, which gives
# the same results more slowly.
option(GGML_VULKAN "ggml: build the Vulkan backend" OFF)

# Static llama/ggml: dllm-ref.exe then runs with no DLLs beside it. Overridable; a shared
# build also works, because every runtime artifact lands in build/bin (below).
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build llama.cpp and ggml as shared libraries")

# llama.cpp puts its DLLs in ${CMAKE_BINARY_DIR}/bin; the exe goes there too.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(${_llama_dir} llama EXCLUDE_FROM_ALL)

set(OFLM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../../src)

add_executable(dllm-ref
    main.cpp
    ${OFLM_SRC}/open_dllm/sampling.cpp
    ${OFLM_SRC}/open_dllm/policy_wedlm.cpp
    ${OFLM_SRC}/open_dllm/policy_sdar.cpp
    ${OFLM_SRC}/open_dllm/policy_llada2.cpp
)
# ../../../src for open_dllm/policy.hpp, ../../../src/include for nlohmann/json.hpp.
target_include_directories(dllm-ref PRIVATE ${OFLM_SRC} ${OFLM_SRC}/include)
target_link_libraries(dllm-ref PRIVATE llama)
set_target_properties(dllm-ref PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
if(MSVC)
    # /utf-8: json.hpp is UTF-8; /bigobj: json-heavy translation unit; /Zc:__cplusplus as the
    # policy tests build them (src/open_dllm/build.cmd).
    target_compile_options(dllm-ref PRIVATE /W4 /utf-8 /bigobj /Zc:__cplusplus /EHsc)
    target_compile_definitions(dllm-ref PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()
