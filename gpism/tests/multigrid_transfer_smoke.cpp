#include "gpism/multigrid.h"
#include "gpism/field_sync.h"

#include <cmath>
#include <iostream>

namespace {

bool nearly_equal(double a, double b, double tol = 1e-12) {
  return std::abs(a - b) <= tol;
}

bool check_restrict(const gpism::FieldStag2D<double>& coarse,
                    const gpism::FieldStag2D<double>& fine) {
  for (int j = 0; j < coarse.local_my(); ++j) {
    for (int i = 0; i < coarse.local_mx(); ++i) {
      const int fi = 2 * i;
      const int fj = 2 * j;
      for (int comp = 0; comp < 2; ++comp) {
        const double expected =
            0.25 * (fine(fi, fj, comp) + fine(fi + 1, fj, comp) +
                    fine(fi, fj + 1, comp) + fine(fi + 1, fj + 1, comp));
        if (!nearly_equal(coarse(i, j, comp), expected)) {
          std::cerr << "restrict mismatch at " << i << "," << j << "," << comp
                    << "\n";
          return false;
        }
      }
    }
  }
  return true;
}

bool check_prolong(const gpism::FieldStag2D<double>& fine,
                   const gpism::FieldStag2D<double>& coarse) {
  const int coarse_mx = coarse.local_mx();
  const int coarse_my = coarse.local_my();
  const int fine_mx = fine.local_mx();
  const int fine_my = fine.local_my();

  auto sample = [&](int i, int j, int comp) {
    const int ii = std::min(std::max(i, 0), coarse_mx - 1);
    const int jj = std::min(std::max(j, 0), coarse_my - 1);
    return coarse(ii, jj, comp);
  };

  for (int j = 0; j < fine_my; ++j) {
    for (int i = 0; i < fine_mx; ++i) {
      const int ic = i / 2;
      const int jc = j / 2;
      const int di = i % 2;
      const int dj = j % 2;
      for (int comp = 0; comp < 2; ++comp) {
        double expected = 0.0;
        if (di == 0 && dj == 0) {
          expected = sample(ic, jc, comp);
        } else if (di == 1 && dj == 0) {
          expected = 0.5 * (sample(ic, jc, comp) + sample(ic + 1, jc, comp));
        } else if (di == 0 && dj == 1) {
          expected = 0.5 * (sample(ic, jc, comp) + sample(ic, jc + 1, comp));
        } else {
          expected = 0.25 * (sample(ic, jc, comp) + sample(ic + 1, jc, comp) +
                             sample(ic, jc + 1, comp) +
                             sample(ic + 1, jc + 1, comp));
        }
        if (!nearly_equal(fine(i, j, comp), expected)) {
          std::cerr << "prolong mismatch at " << i << "," << j << "," << comp
                    << "\n";
          return false;
        }
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  const int fine_mx = 4;
  const int fine_my = 4;
  const int coarse_mx = 2;
  const int coarse_my = 2;
  const int gw = 1;

  gpism::FieldStag2D<double> fine(fine_mx, fine_my, gw);
  gpism::FieldStag2D<double> coarse(coarse_mx, coarse_my, gw);
  gpism::FieldStag2D<double> fine_from_coarse(fine_mx, fine_my, gw);

  for (int j = 0; j < fine_my; ++j) {
    for (int i = 0; i < fine_mx; ++i) {
      fine(i, j, 0) = static_cast<double>(i + 10 * j);
      fine(i, j, 1) = static_cast<double>(100 + i + 10 * j);
    }
  }
  gpism::sync_host_to_device(fine);

  gpism::restrict_stag(fine, coarse);
  gpism::sync_device_to_host(coarse);
  if (!check_restrict(coarse, fine)) {
    return 1;
  }

  gpism::prolong_stag(coarse, fine_from_coarse);
  gpism::sync_device_to_host(fine_from_coarse);
  if (!check_prolong(fine_from_coarse, coarse)) {
    return 1;
  }

  std::cout << "multigrid_transfer_smoke passed\n";
  return 0;
}
