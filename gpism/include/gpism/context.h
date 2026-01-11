#pragma once

namespace gpism {

class Context {
public:
  struct NeighborRanks {
    int west;
    int east;
    int south;
    int north;
    int coord_x;
    int coord_y;
    int dims_x;
    int dims_y;
  };

  Context(int* argc, char*** argv);
  ~Context();

  int rank() const;
  int size() const;
  bool mpi_enabled() const;
  int device_id() const;
  int device_count() const;
  NeighborRanks neighbors_2d(int dims_x, int dims_y) const;

private:
  bool owns_mpi_;
  int rank_;
  int size_;
  int device_id_;
  int device_count_;
};

}  // namespace gpism
