#include "gpism/grid2d.h"

#include <algorithm>
#include <cmath>

namespace gpism {
namespace {

void compute_dims(int size, int* dims_x, int* dims_y) {
  int px = static_cast<int>(std::sqrt(static_cast<double>(size)));
  while (px > 1 && size % px != 0) {
    --px;
  }
  int py = size / px;
  *dims_x = px;
  *dims_y = py;
}

void compute_local_extent(int global, int dims, int coord, int* start, int* count) {
  int base = global / dims;
  int extra = global % dims;
  int local = base + (coord < extra ? 1 : 0);
  int offset = coord * base + std::min(coord, extra);
  *start = offset;
  *count = local;
}

}  // namespace

Grid2D::Grid2D(int global_mx, int global_my, double dx, double dy, int ghost_width,
               int rank, int size)
    : global_mx_(global_mx),
      global_my_(global_my),
      dx_(dx),
      dy_(dy),
      ghost_width_(ghost_width),
      dims_x_(1),
      dims_y_(1),
      coord_x_(0),
      coord_y_(0),
      xs_(0),
      ys_(0),
      local_mx_(global_mx),
      local_my_(global_my),
      neighbor_west_(-1),
      neighbor_east_(-1),
      neighbor_south_(-1),
      neighbor_north_(-1) {
  compute_dims(size, &dims_x_, &dims_y_);
  coord_x_ = rank % dims_x_;
  coord_y_ = rank / dims_x_;

  compute_local_extent(global_mx_, dims_x_, coord_x_, &xs_, &local_mx_);
  compute_local_extent(global_my_, dims_y_, coord_y_, &ys_, &local_my_);

  neighbor_west_ = (coord_x_ > 0) ? rank - 1 : -1;
  neighbor_east_ = (coord_x_ + 1 < dims_x_) ? rank + 1 : -1;
  neighbor_south_ = (coord_y_ > 0) ? rank - dims_x_ : -1;
  neighbor_north_ = (coord_y_ + 1 < dims_y_) ? rank + dims_x_ : -1;
}

int Grid2D::global_mx() const { return global_mx_; }
int Grid2D::global_my() const { return global_my_; }
double Grid2D::dx() const { return dx_; }
double Grid2D::dy() const { return dy_; }
int Grid2D::ghost_width() const { return ghost_width_; }

int Grid2D::local_mx() const { return local_mx_; }
int Grid2D::local_my() const { return local_my_; }
int Grid2D::xs() const { return xs_; }
int Grid2D::ys() const { return ys_; }

int Grid2D::dims_x() const { return dims_x_; }
int Grid2D::dims_y() const { return dims_y_; }
int Grid2D::coord_x() const { return coord_x_; }
int Grid2D::coord_y() const { return coord_y_; }

int Grid2D::neighbor_west() const { return neighbor_west_; }
int Grid2D::neighbor_east() const { return neighbor_east_; }
int Grid2D::neighbor_south() const { return neighbor_south_; }
int Grid2D::neighbor_north() const { return neighbor_north_; }

}  // namespace gpism
