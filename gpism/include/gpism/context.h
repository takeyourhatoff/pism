#pragma once

namespace gpism {

class Context {
public:
  Context(int* argc, char*** argv);
  ~Context();

  int rank() const;
  int size() const;
  bool mpi_enabled() const;

private:
  bool owns_mpi_;
  int rank_;
  int size_;
};

}  // namespace gpism
