#pragma once

namespace gpism {

class Context {
public:
  Context(int* argc, char*** argv);
  ~Context();

  int rank() const;
  int size() const;
  bool mpi_enabled() const;
  int device_id() const;
  int device_count() const;

private:
  bool owns_mpi_;
  int rank_;
  int size_;
  int device_id_;
  int device_count_;
};

}  // namespace gpism
