#include <type_traits>

#include "marketforge/cuda/attention.hpp"
#include "marketforge/cuda/cublas_handle.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/decoder_layer.hpp"
#include "marketforge/cuda/device_buffer.hpp"
#include "marketforge/cuda/elementwise.hpp"
#include "marketforge/cuda/embedding.hpp"
#include "marketforge/cuda/greedy.hpp"
#include "marketforge/cuda/kv_cache.hpp"
#include "marketforge/cuda/lifecycle_probe.hpp"
#include "marketforge/cuda/linear.hpp"
#include "marketforge/cuda/rms_norm.hpp"
#include "marketforge/cuda/rope.hpp"
#include "marketforge/cuda/smollm2.hpp"
#include "marketforge/cuda/swiglu.hpp"

static_assert(!std::is_copy_constructible_v<marketforge::cuda::CublasHandle>);
static_assert(!std::is_copy_assignable_v<marketforge::cuda::CublasHandle>);
static_assert(
    std::is_nothrow_move_constructible_v<marketforge::cuda::CublasHandle>);
static_assert(
    std::is_nothrow_move_assignable_v<marketforge::cuda::CublasHandle>);
static_assert(std::is_nothrow_destructible_v<marketforge::cuda::CublasHandle>);
static_assert(!std::is_copy_constructible_v<marketforge::cuda::CudaStream>);
static_assert(!std::is_copy_assignable_v<marketforge::cuda::CudaStream>);
static_assert(
    std::is_nothrow_move_constructible_v<marketforge::cuda::CudaStream>);
static_assert(std::is_nothrow_move_assignable_v<marketforge::cuda::CudaStream>);
static_assert(std::is_nothrow_destructible_v<marketforge::cuda::CudaStream>);
static_assert(!std::is_copy_constructible_v<marketforge::cuda::DeviceBuffer>);
static_assert(!std::is_copy_assignable_v<marketforge::cuda::DeviceBuffer>);
static_assert(
    std::is_nothrow_move_constructible_v<marketforge::cuda::DeviceBuffer>);
static_assert(
    std::is_nothrow_move_assignable_v<marketforge::cuda::DeviceBuffer>);
static_assert(std::is_nothrow_destructible_v<marketforge::cuda::DeviceBuffer>);

int main() { return 0; }
