#include <type_traits>

#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"
#include "marketforge/cuda/lifecycle_probe.hpp"

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
