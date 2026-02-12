#pragma once

#include <vector>

#include "gpism/field2d.h"
#include "gpism/field_stag2d.h"

#include <cuda_runtime.h>

namespace gpism {

enum class ReductionBackend { CUB, CUBLAS };

class DeviceScalarBuffer {
public:
  DeviceScalarBuffer() = default;
  ~DeviceScalarBuffer();

  DeviceScalarBuffer(const DeviceScalarBuffer&) = delete;
  DeviceScalarBuffer& operator=(const DeviceScalarBuffer&) = delete;

  DeviceScalarBuffer(DeviceScalarBuffer&& other) noexcept;
  DeviceScalarBuffer& operator=(DeviceScalarBuffer&& other) noexcept;

  void ensure(int count);
  int capacity() const { return capacity_; }

  double* device_data() { return device_data_; }
  const double* device_data() const { return device_data_; }
  double* host_data() { return host_data_; }
  const double* host_data() const { return host_data_; }

private:
  double* device_data_ = nullptr;
  double* host_data_ = nullptr;
  int capacity_ = 0;
};

void set_reduction_backend(ReductionBackend backend);
ReductionBackend reduction_backend();

void dot_device_async(const FieldStag2D<double>& a, const FieldStag2D<double>& b,
                      DeviceScalarBuffer& out, int out_index = 0,
                      cudaStream_t stream = nullptr);

void dot_batch_device_async(
    const FieldStag2D<double>& w,
    const std::vector<const FieldStag2D<double>*>& vectors,
    DeviceScalarBuffer& out, int count, cudaStream_t stream = nullptr);

void norm1_device_async(const Field2D<double>& a, DeviceScalarBuffer& out,
                        int out_index = 0, cudaStream_t stream = nullptr);

void diff_norm1_device_async(const Field2D<double>& a, const Field2D<double>& b,
                             DeviceScalarBuffer& out, int out_index = 0,
                             cudaStream_t stream = nullptr);

void copy_device_scalars_to_host(DeviceScalarBuffer& out, int count);

void copy_device_scalars_to_host_async(DeviceScalarBuffer& out, int count,
                                       cudaStream_t stream = nullptr);

void synchronize_cuda_stream(cudaStream_t stream = nullptr);

void normalize_stag_from_norm2_device(
    const FieldStag2D<double>& src, FieldStag2D<double>& dst,
    const DeviceScalarBuffer& norm2_buffer, int norm2_index,
    cudaStream_t stream = nullptr);

void normalize_stag_from_norm2_pointer_device(
    const FieldStag2D<double>& src, FieldStag2D<double>& dst,
    const double* norm2_device_ptr, cudaStream_t stream = nullptr);

void gmres_apply_givens_device(DeviceScalarBuffer& hessenberg,
                               DeviceScalarBuffer& cs,
                               DeviceScalarBuffer& sn,
                               DeviceScalarBuffer& g,
                               int restart,
                               int j,
                               cudaStream_t stream = nullptr);

void gmres_backsolve_device(const DeviceScalarBuffer& hessenberg,
                            const DeviceScalarBuffer& g,
                            DeviceScalarBuffer& y,
                            int restart,
                            int k,
                            cudaStream_t stream = nullptr);

void gmres_update_solution_device(const DeviceScalarBuffer& y, int k,
                                  const FieldStag2D<double>& basis,
                                  const double** basis_u_dev,
                                  const double** basis_v_dev,
                                  FieldStag2D<double>& x,
                                  cudaStream_t stream = nullptr);

double norm1(const Field2D<double>& a);
double diff_norm1(const Field2D<double>& a, const Field2D<double>& b);

void axpy(double alpha, const FieldStag2D<double>& x, FieldStag2D<double>& y);
void scal(double alpha, FieldStag2D<double>& x);
void copy(const FieldStag2D<double>& x, FieldStag2D<double>& y);
void set(double value, FieldStag2D<double>& x);

double dot(const FieldStag2D<double>& a, const FieldStag2D<double>& b);

double norm2(const FieldStag2D<double>& a);

double norm1(const FieldStag2D<double>& a);

double diff_norm1(const FieldStag2D<double>& a, const FieldStag2D<double>& b);

}  // namespace gpism
