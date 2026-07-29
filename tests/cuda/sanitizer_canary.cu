#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdio>
#include <cstring>

namespace {

constexpr std::size_t kLeakBytes = 256;

__global__ void invalid_global_write_kernel(int* output) {
  output[1] = 0x5A5A5A5A;
}

int invalid_global_write() {
  void* allocation = nullptr;
  if (cudaMalloc(&allocation, sizeof(int)) != cudaSuccess) {
    return 3;
  }

  std::puts("MARKETFORGE_SANITIZER_CANARY mode=invalid-global-write bytes=4");
  std::fflush(stdout);
  invalid_global_write_kernel<<<1, 1>>>(static_cast<int*>(allocation));
  (void)cudaDeviceSynchronize();
  (void)cudaFree(allocation);
  return 0;
}

int device_leak() {
  void* allocation = nullptr;
  if (cudaMalloc(&allocation, kLeakBytes) != cudaSuccess) {
    return 3;
  }
  if (cudaMemset(allocation, 0xA5, kLeakBytes) != cudaSuccess) {
    (void)cudaFree(allocation);
    return 4;
  }

  std::puts("MARKETFORGE_SANITIZER_CANARY mode=device-leak bytes=256");
  std::fflush(stdout);
  (void)cudaDeviceSynchronize();
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 3 || std::strcmp(argv[1], "--mode") != 0) {
    return 2;
  }
  if (std::strcmp(argv[2], "invalid-global-write") == 0) {
    return invalid_global_write();
  }
  if (std::strcmp(argv[2], "device-leak") == 0) {
    return device_leak();
  }
  return 2;
}
