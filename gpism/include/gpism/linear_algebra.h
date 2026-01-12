#pragma once

#include "gpism/field_stag2d.h"

namespace gpism {

void axpy(double alpha, const FieldStag2D<double>& x, FieldStag2D<double>& y);
void scal(double alpha, FieldStag2D<double>& x);
void copy(const FieldStag2D<double>& x, FieldStag2D<double>& y);
void set(double value, FieldStag2D<double>& x);

double dot(const FieldStag2D<double>& a, const FieldStag2D<double>& b);

double norm2(const FieldStag2D<double>& a);

double norm1(const FieldStag2D<double>& a);

double diff_norm1(const FieldStag2D<double>& a, const FieldStag2D<double>& b);

}  // namespace gpism
