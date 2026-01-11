#include <cuda_runtime.h>

namespace gpism {
namespace {

__global__ void gpism_noop_kernel(int* data) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    data[0] = 1;
  }
}

}  // namespace

bool cuda_smoke_test() {
  int* device_data = nullptr;
  if (cudaMalloc(&device_data, sizeof(int)) != cudaSuccess) {
    return false;
  }

  gpism_noop_kernel<<<1, 1>>>(device_data);
  if (cudaGetLastError() != cudaSuccess) {
    cudaFree(device_data);
    return false;
  }

  int host_data = 0;
  if (cudaMemcpy(&host_data, device_data, sizeof(int), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    cudaFree(device_data);
    return false;
  }

  cudaFree(device_data);
  return host_data == 1;
}

}  // namespace gpism
