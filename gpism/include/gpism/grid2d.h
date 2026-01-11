#pragma once

#include <cstddef>

namespace gpism {

class Grid2D {
public:
  Grid2D(int global_mx, int global_my, double dx, double dy, int ghost_width,
         int rank, int size);

  int global_mx() const;
  int global_my() const;
  double dx() const;
  double dy() const;
  int ghost_width() const;

  int local_mx() const;
  int local_my() const;
  int xs() const;
  int ys() const;

  int dims_x() const;
  int dims_y() const;
  int coord_x() const;
  int coord_y() const;

  int neighbor_west() const;
  int neighbor_east() const;
  int neighbor_south() const;
  int neighbor_north() const;

private:
  int global_mx_;
  int global_my_;
  double dx_;
  double dy_;
  int ghost_width_;

  int dims_x_;
  int dims_y_;
  int coord_x_;
  int coord_y_;

  int xs_;
  int ys_;
  int local_mx_;
  int local_my_;

  int neighbor_west_;
  int neighbor_east_;
  int neighbor_south_;
  int neighbor_north_;
};

}  // namespace gpism
