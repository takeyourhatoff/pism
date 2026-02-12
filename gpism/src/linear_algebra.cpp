#include "gpism/linear_algebra.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpism {

void axpy_stag_cuda(int mx, int my, int gw, int stride_x_u, int stride_x_v,
                    int stride_y_u, int stride_y_v, const double* x_u,
                    const double* x_v, double* y_u, double* y_v, double alpha);
void scal_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                    double* x_u, double* x_v, double alpha);
void copy_stag_cuda(int mx, int my, int gw, int stride_x_u, int stride_x_v,
                    int stride_y_u, int stride_y_v, const double* x_u,
                    const double* x_v, double* y_u, double* y_v);
void set_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                   double* x_u, double* x_v, double value);
double dot_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                     const double* a_u, const double* a_v,
                     const double* b_u, const double* b_v);
double norm1_cuda(int mx, int my, int gw, int stride, const double* a);
double norm1_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                       const double* a_u, const double* a_v);
double diff_norm1_cuda(int mx, int my, int gw, int stride_a, int stride_b,
                       const double* a, const double* b);
double diff_norm1_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                            const double* a_u, const double* a_v,
                            const double* b_u, const double* b_v);
void dot_stag_cuda_async(int mx, int my, int gw, int stride_u, int stride_v,
                         const double* a_u, const double* a_v,
                         const double* b_u, const double* b_v, double* out,
                         int out_index, cudaStream_t stream);
void dot_stag_batch_cuda_async(int mx, int my, int gw, int stride_u,
                               int stride_v, const double* w_u,
                               const double* w_v, const double** V_u,
                               const double** V_v, int count, double* out,
                               cudaStream_t stream);
void norm1_cuda_async(int mx, int my, int gw, int stride, const double* a,
                      double* out, int out_index, cudaStream_t stream);
void diff_norm1_cuda_async(int mx, int my, int gw, int stride_a, int stride_b,
                           const double* a, const double* b, double* out,
                           int out_index, cudaStream_t stream);
void normalize_stag_from_norm2_cuda(int mx, int my, int gw, int stride_u,
                                    int stride_v, const double* src_u,
                                    const double* src_v, double* dst_u,
                                    double* dst_v, const double* norm2,
                                    cudaStream_t stream);
void gmres_apply_givens_cuda(double* hessenberg, int restart, int j,
                             double* cs, double* sn, double* g,
                             cudaStream_t stream);
void gmres_backsolve_cuda(const double* hessenberg, int restart, int k,
                          const double* g, double* y, cudaStream_t stream);
void gmres_update_solution_stag_cuda(
    int mx, int my, int gw, int stride_u, int stride_v, double* x_u,
    double* x_v, const double** V_u, const double** V_v, const double* y, int k,
    cudaStream_t stream);
void device_scalars_copy_to_host(const double* d_src, double* h_dst, int count);
void device_scalars_copy_to_host_async(const double* d_src, double* h_dst,
                                       int count, cudaStream_t stream);
void cuda_stream_synchronize(cudaStream_t stream);

namespace {

ReductionBackend& backend_state() {
  static ReductionBackend backend = ReductionBackend::CUB;
  return backend;
}

void require_device(const Field2D<double>& field, const char* where) {
  if (!field.has_device_data()) {
    throw std::runtime_error(std::string(where) +
                             " requires device-resident Field2D");
  }
}

void require_device(const FieldStag2D<double>& field, const char* where) {
  if (!field.component(0).has_device_data() ||
      !field.component(1).has_device_data()) {
    throw std::runtime_error(std::string(where) +
                             " requires device-resident FieldStag2D");
  }
}

void require_capacity(const DeviceScalarBuffer& out, int count,
                      const char* where) {
  if (count < 0 || out.capacity() < count) {
    throw std::runtime_error(std::string(where) +
                             " requires DeviceScalarBuffer capacity >= count");
  }
}

class DotBatchPointerCache {
public:
  ~DotBatchPointerCache() {
    if (v_u_dev_ != nullptr) {
      cudaFree(v_u_dev_);
      v_u_dev_ = nullptr;
    }
    if (v_v_dev_ != nullptr) {
      cudaFree(v_v_dev_);
      v_v_dev_ = nullptr;
    }
  }

  void ensure(int count) {
    if (count <= capacity_) {
      return;
    }
    if (v_u_dev_ != nullptr) {
      cudaFree(v_u_dev_);
      v_u_dev_ = nullptr;
    }
    if (v_v_dev_ != nullptr) {
      cudaFree(v_v_dev_);
      v_v_dev_ = nullptr;
    }
    cudaMalloc(reinterpret_cast<void**>(&v_u_dev_),
               static_cast<std::size_t>(count) * sizeof(double*));
    cudaMalloc(reinterpret_cast<void**>(&v_v_dev_),
               static_cast<std::size_t>(count) * sizeof(double*));
    v_u_host_.resize(static_cast<std::size_t>(count));
    v_v_host_.resize(static_cast<std::size_t>(count));
    capacity_ = count;
  }

  void update(const std::vector<const FieldStag2D<double>*>& vectors, int count) {
    ensure(count);
    for (int i = 0; i < count; ++i) {
      const FieldStag2D<double>* v = vectors[static_cast<std::size_t>(i)];
      v_u_host_[static_cast<std::size_t>(i)] = v->component(0).device_data();
      v_v_host_[static_cast<std::size_t>(i)] = v->component(1).device_data();
    }
    cudaMemcpy(v_u_dev_, v_u_host_.data(),
               static_cast<std::size_t>(count) * sizeof(double*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(v_v_dev_, v_v_host_.data(),
               static_cast<std::size_t>(count) * sizeof(double*),
               cudaMemcpyHostToDevice);
  }

  const double** v_u_dev() const { return v_u_dev_; }
  const double** v_v_dev() const { return v_v_dev_; }

private:
  int capacity_ = 0;
  const double** v_u_dev_ = nullptr;
  const double** v_v_dev_ = nullptr;
  std::vector<const double*> v_u_host_;
  std::vector<const double*> v_v_host_;
};

DotBatchPointerCache& dot_batch_cache() {
  static DotBatchPointerCache cache;
  return cache;
}

}  // namespace

DeviceScalarBuffer::~DeviceScalarBuffer() {
  if (device_data_ != nullptr) {
    cudaFree(device_data_);
    device_data_ = nullptr;
  }
  if (host_data_ != nullptr) {
    cudaFreeHost(host_data_);
    host_data_ = nullptr;
  }
  capacity_ = 0;
}

DeviceScalarBuffer::DeviceScalarBuffer(DeviceScalarBuffer&& other) noexcept
    : device_data_(other.device_data_),
      host_data_(other.host_data_),
      capacity_(other.capacity_) {
  other.device_data_ = nullptr;
  other.host_data_ = nullptr;
  other.capacity_ = 0;
}

DeviceScalarBuffer& DeviceScalarBuffer::operator=(
    DeviceScalarBuffer&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (device_data_ != nullptr) {
    cudaFree(device_data_);
  }
  if (host_data_ != nullptr) {
    cudaFreeHost(host_data_);
  }
  device_data_ = other.device_data_;
  host_data_ = other.host_data_;
  capacity_ = other.capacity_;
  other.device_data_ = nullptr;
  other.host_data_ = nullptr;
  other.capacity_ = 0;
  return *this;
}

void DeviceScalarBuffer::ensure(int count) {
  if (count <= capacity_) {
    return;
  }
  if (device_data_ != nullptr) {
    cudaFree(device_data_);
    device_data_ = nullptr;
  }
  if (host_data_ != nullptr) {
    cudaFreeHost(host_data_);
    host_data_ = nullptr;
  }
  cudaMalloc(reinterpret_cast<void**>(&device_data_),
             static_cast<std::size_t>(count) * sizeof(double));
  cudaHostAlloc(reinterpret_cast<void**>(&host_data_),
                static_cast<std::size_t>(count) * sizeof(double),
                cudaHostAllocDefault);
  capacity_ = count;
}

void set_reduction_backend(ReductionBackend backend) {
  backend_state() = backend;
}

ReductionBackend reduction_backend() {
  return backend_state();
}

void dot_device_async(const FieldStag2D<double>& a, const FieldStag2D<double>& b,
                      DeviceScalarBuffer& out, int out_index,
                      cudaStream_t stream) {
  require_device(a, "dot_device_async");
  require_device(b, "dot_device_async");
  require_capacity(out, out_index + 1, "dot_device_async");
  dot_stag_cuda_async(a.local_mx(), a.local_my(), a.ghost_width(),
                      a.component(0).stride(), a.component(1).stride(),
                      a.component(0).device_data(), a.component(1).device_data(),
                      b.component(0).device_data(), b.component(1).device_data(),
                      out.device_data(), out_index, stream);
}

void dot_batch_device_async(
    const FieldStag2D<double>& w,
    const std::vector<const FieldStag2D<double>*>& vectors,
    DeviceScalarBuffer& out, int count, cudaStream_t stream) {
  require_device(w, "dot_batch_device_async");
  if (count < 0 || count > static_cast<int>(vectors.size())) {
    throw std::runtime_error("dot_batch_device_async received invalid count");
  }
  require_capacity(out, count, "dot_batch_device_async");
  for (int i = 0; i < count; ++i) {
    if (vectors[static_cast<std::size_t>(i)] == nullptr) {
      throw std::runtime_error("dot_batch_device_async vector pointer is null");
    }
    require_device(*vectors[static_cast<std::size_t>(i)],
                   "dot_batch_device_async");
  }
  auto& cache = dot_batch_cache();
  cache.update(vectors, count);
  dot_stag_batch_cuda_async(
      w.local_mx(), w.local_my(), w.ghost_width(), w.component(0).stride(),
      w.component(1).stride(), w.component(0).device_data(),
      w.component(1).device_data(), cache.v_u_dev(), cache.v_v_dev(), count,
      out.device_data(), stream);
}

void norm1_device_async(const Field2D<double>& a, DeviceScalarBuffer& out,
                        int out_index, cudaStream_t stream) {
  require_device(a, "norm1_device_async");
  require_capacity(out, out_index + 1, "norm1_device_async");
  norm1_cuda_async(a.local_mx(), a.local_my(), a.ghost_width(), a.stride(),
                   a.device_data(), out.device_data(), out_index, stream);
}

void diff_norm1_device_async(const Field2D<double>& a, const Field2D<double>& b,
                             DeviceScalarBuffer& out, int out_index,
                             cudaStream_t stream) {
  require_device(a, "diff_norm1_device_async");
  require_device(b, "diff_norm1_device_async");
  require_capacity(out, out_index + 1, "diff_norm1_device_async");
  diff_norm1_cuda_async(a.local_mx(), a.local_my(), a.ghost_width(),
                        a.stride(), b.stride(), a.device_data(),
                        b.device_data(), out.device_data(), out_index, stream);
}

void copy_device_scalars_to_host(DeviceScalarBuffer& out, int count) {
  require_capacity(out, count, "copy_device_scalars_to_host");
  device_scalars_copy_to_host(out.device_data(), out.host_data(), count);
}

void copy_device_scalars_to_host_async(DeviceScalarBuffer& out, int count,
                                       cudaStream_t stream) {
  require_capacity(out, count, "copy_device_scalars_to_host_async");
  device_scalars_copy_to_host_async(out.device_data(), out.host_data(), count,
                                    stream);
}

void synchronize_cuda_stream(cudaStream_t stream) {
  cuda_stream_synchronize(stream);
}

void normalize_stag_from_norm2_device(
    const FieldStag2D<double>& src, FieldStag2D<double>& dst,
    const DeviceScalarBuffer& norm2_buffer, int norm2_index,
    cudaStream_t stream) {
  require_device(src, "normalize_stag_from_norm2_device");
  require_device(dst, "normalize_stag_from_norm2_device");
  require_capacity(norm2_buffer, norm2_index + 1,
                   "normalize_stag_from_norm2_device");
  normalize_stag_from_norm2_cuda(
      src.local_mx(), src.local_my(), src.ghost_width(),
      src.component(0).stride(), src.component(1).stride(),
      src.component(0).device_data(), src.component(1).device_data(),
      dst.component(0).device_data(), dst.component(1).device_data(),
      norm2_buffer.device_data() + norm2_index, stream);
}

void normalize_stag_from_norm2_pointer_device(
    const FieldStag2D<double>& src, FieldStag2D<double>& dst,
    const double* norm2_device_ptr, cudaStream_t stream) {
  require_device(src, "normalize_stag_from_norm2_pointer_device");
  require_device(dst, "normalize_stag_from_norm2_pointer_device");
  if (norm2_device_ptr == nullptr) {
    throw std::runtime_error(
        "normalize_stag_from_norm2_pointer_device requires non-null norm2 ptr");
  }
  normalize_stag_from_norm2_cuda(
      src.local_mx(), src.local_my(), src.ghost_width(),
      src.component(0).stride(), src.component(1).stride(),
      src.component(0).device_data(), src.component(1).device_data(),
      dst.component(0).device_data(), dst.component(1).device_data(),
      norm2_device_ptr, stream);
}

void gmres_apply_givens_device(DeviceScalarBuffer& hessenberg,
                               DeviceScalarBuffer& cs,
                               DeviceScalarBuffer& sn,
                               DeviceScalarBuffer& g,
                               int restart, int j, cudaStream_t stream) {
  const int hess_needed = (restart + 1) * restart;
  require_capacity(hessenberg, hess_needed, "gmres_apply_givens_device");
  require_capacity(cs, restart, "gmres_apply_givens_device");
  require_capacity(sn, restart, "gmres_apply_givens_device");
  require_capacity(g, restart + 1, "gmres_apply_givens_device");
  gmres_apply_givens_cuda(hessenberg.device_data(), restart, j, cs.device_data(),
                          sn.device_data(), g.device_data(), stream);
}

void gmres_backsolve_device(const DeviceScalarBuffer& hessenberg,
                            const DeviceScalarBuffer& g,
                            DeviceScalarBuffer& y,
                            int restart, int k,
                            cudaStream_t stream) {
  const int hess_needed = (restart + 1) * restart;
  require_capacity(hessenberg, hess_needed, "gmres_backsolve_device");
  require_capacity(g, restart + 1, "gmres_backsolve_device");
  require_capacity(y, k, "gmres_backsolve_device");
  gmres_backsolve_cuda(hessenberg.device_data(), restart, k, g.device_data(),
                       y.device_data(), stream);
}

void gmres_update_solution_device(const DeviceScalarBuffer& y, int k,
                                  const FieldStag2D<double>& basis,
                                  const double** basis_u_dev,
                                  const double** basis_v_dev,
                                  FieldStag2D<double>& x,
                                  cudaStream_t stream) {
  require_device(basis, "gmres_update_solution_device");
  require_device(x, "gmres_update_solution_device");
  require_capacity(y, k, "gmres_update_solution_device");
  gmres_update_solution_stag_cuda(
      basis.local_mx(), basis.local_my(), basis.ghost_width(),
      basis.component(0).stride(), basis.component(1).stride(),
      x.component(0).device_data(), x.component(1).device_data(), basis_u_dev,
      basis_v_dev, y.device_data(), k, stream);
}

double norm1(const Field2D<double>& a) {
  require_device(a, "norm1");
  return norm1_cuda(a.local_mx(), a.local_my(), a.ghost_width(), a.stride(),
                    a.device_data());
}

double diff_norm1(const Field2D<double>& a, const Field2D<double>& b) {
  require_device(a, "diff_norm1");
  require_device(b, "diff_norm1");
  return diff_norm1_cuda(a.local_mx(), a.local_my(), a.ghost_width(),
                         a.stride(), b.stride(), a.device_data(),
                         b.device_data());
}

void axpy(double alpha, const FieldStag2D<double>& x, FieldStag2D<double>& y) {
  require_device(x, "axpy");
  require_device(y, "axpy");
  axpy_stag_cuda(x.local_mx(), x.local_my(), x.ghost_width(),
                 x.component(0).stride(), x.component(1).stride(),
                 y.component(0).stride(), y.component(1).stride(),
                 x.component(0).device_data(), x.component(1).device_data(),
                 y.component(0).device_data(), y.component(1).device_data(),
                 alpha);
}

void scal(double alpha, FieldStag2D<double>& x) {
  require_device(x, "scal");
  scal_stag_cuda(x.local_mx(), x.local_my(), x.ghost_width(),
                 x.component(0).stride(), x.component(1).stride(),
                 x.component(0).device_data(), x.component(1).device_data(),
                 alpha);
}

void copy(const FieldStag2D<double>& x, FieldStag2D<double>& y) {
  require_device(x, "copy");
  require_device(y, "copy");
  copy_stag_cuda(x.local_mx(), x.local_my(), x.ghost_width(),
                 x.component(0).stride(), x.component(1).stride(),
                 y.component(0).stride(), y.component(1).stride(),
                 x.component(0).device_data(), x.component(1).device_data(),
                 y.component(0).device_data(), y.component(1).device_data());
}

void set(double value, FieldStag2D<double>& x) {
  require_device(x, "set");
  set_stag_cuda(x.local_mx(), x.local_my(), x.ghost_width(),
                x.component(0).stride(), x.component(1).stride(),
                x.component(0).device_data(), x.component(1).device_data(),
                value);
}

double dot(const FieldStag2D<double>& a, const FieldStag2D<double>& b) {
  require_device(a, "dot");
  require_device(b, "dot");
  return dot_stag_cuda(a.local_mx(), a.local_my(), a.ghost_width(),
                       a.component(0).stride(), a.component(1).stride(),
                       a.component(0).device_data(), a.component(1).device_data(),
                       b.component(0).device_data(), b.component(1).device_data());
}

double norm2(const FieldStag2D<double>& a) { return std::sqrt(dot(a, a)); }

double norm1(const FieldStag2D<double>& a) {
  require_device(a, "norm1");
  return norm1_stag_cuda(a.local_mx(), a.local_my(), a.ghost_width(),
                         a.component(0).stride(), a.component(1).stride(),
                         a.component(0).device_data(),
                         a.component(1).device_data());
}

double diff_norm1(const FieldStag2D<double>& a, const FieldStag2D<double>& b) {
  require_device(a, "diff_norm1");
  require_device(b, "diff_norm1");
  return diff_norm1_stag_cuda(a.local_mx(), a.local_my(), a.ghost_width(),
                              a.component(0).stride(), a.component(1).stride(),
                              a.component(0).device_data(),
                              a.component(1).device_data(),
                              b.component(0).device_data(),
                              b.component(1).device_data());
}

}  // namespace gpism
