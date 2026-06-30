#include "colmap_eigen.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

using RowMajorMatrix9 =
    Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>;
using RowMajorMatrix10 =
    Eigen::Matrix<double, 10, 10, Eigen::RowMajor>;
using RowMajorMatrix3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

bool IsFinite(const double* values, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    if (!std::isfinite(values[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace

int rustsfm_eigen_right_nullspace_9(
    const double* row_major,
    size_t rows,
    size_t nullity,
    double* output) {
  if (row_major == nullptr || output == nullptr || rows == 0 ||
      nullity == 0 || rows + nullity != 9 || !IsFinite(row_major, rows * 9)) {
    return 0;
  }

  const Eigen::Map<const RowMajorMatrix9> a(row_major, rows, 9);
  const Eigen::Matrix<double, 9, Eigen::Dynamic> at = a.transpose();
  Eigen::HouseholderQR<Eigen::Matrix<double, 9, Eigen::Dynamic>> qr(at);
  const Eigen::Matrix<double, 9, 9> q = qr.householderQ();

  const size_t first = 9 - nullity;
  for (size_t basis_idx = 0; basis_idx < nullity; ++basis_idx) {
    const size_t q_col = first + basis_idx;
    for (size_t row = 0; row < 9; ++row) {
      output[basis_idx * 9 + row] = q(static_cast<Eigen::Index>(row),
                                      static_cast<Eigen::Index>(q_col));
    }
  }
  return IsFinite(output, nullity * 9) ? 1 : 0;
}

int rustsfm_eigen_full_piv_right_nullspace_9(
    const double* row_major,
    size_t rows,
    size_t nullity,
    double* output) {
  if (row_major == nullptr || output == nullptr || rows == 0 ||
      nullity == 0 || rows + nullity != 9 || !IsFinite(row_major, rows * 9)) {
    return 0;
  }

  const Eigen::Map<const RowMajorMatrix9> a(row_major, rows, 9);
  const Eigen::Matrix<double, 9, Eigen::Dynamic> at = a.transpose();
  const Eigen::Matrix<double, 9, 9> q = at.fullPivHouseholderQr().matrixQ();

  const size_t first = 9 - nullity;
  for (size_t basis_idx = 0; basis_idx < nullity; ++basis_idx) {
    const size_t q_col = first + basis_idx;
    for (size_t row = 0; row < 9; ++row) {
      output[basis_idx * 9 + row] = q(static_cast<Eigen::Index>(row),
                                      static_cast<Eigen::Index>(q_col));
    }
  }
  return IsFinite(output, nullity * 9) ? 1 : 0;
}

int rustsfm_eigen_jacobi_svd_vt_9(
    const double* row_major,
    size_t rows,
    double* output_vt,
    double* output_singular_values,
    size_t* output_num_singular_values) {
  if (row_major == nullptr || output_vt == nullptr ||
      output_singular_values == nullptr ||
      output_num_singular_values == nullptr || rows == 0 ||
      !IsFinite(row_major, rows * 9)) {
    return 0;
  }

  const Eigen::Map<const RowMajorMatrix9> a(row_major, rows, 9);
  const Eigen::JacobiSVD<RowMajorMatrix9> svd(a, Eigen::ComputeFullV);
  const Eigen::Matrix<double, 9, 9> vt = svd.matrixV().transpose();

  for (size_t row = 0; row < 9; ++row) {
    for (size_t col = 0; col < 9; ++col) {
      output_vt[row * 9 + col] = vt(static_cast<Eigen::Index>(row),
                                    static_cast<Eigen::Index>(col));
    }
  }

  std::fill(output_singular_values, output_singular_values + 9, 0.0);
  const size_t count = std::min<size_t>(rows, 9);
  for (size_t idx = 0; idx < count; ++idx) {
    output_singular_values[idx] =
        svd.singularValues()(static_cast<Eigen::Index>(idx));
  }
  *output_num_singular_values = count;

  return IsFinite(output_vt, 81) && IsFinite(output_singular_values, count) ? 1
                                                                            : 0;
}

int rustsfm_eigen_companion_roots(
    const double* coeffs,
    size_t len,
    double* output_interleaved_complex,
    size_t* output_num_roots) {
  if (coeffs == nullptr || output_interleaved_complex == nullptr ||
      output_num_roots == nullptr || len < 3 || !IsFinite(coeffs, len)) {
    return 0;
  }

  const size_t degree = len - 1;
  const double lead = coeffs[0];
  if (!std::isfinite(lead) || std::abs(lead) < 1.0e-15) {
    return 0;
  }

  Eigen::MatrixXd companion =
      Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(degree),
                            static_cast<Eigen::Index>(degree));
  for (size_t row = 1; row < degree; ++row) {
    companion(static_cast<Eigen::Index>(row),
              static_cast<Eigen::Index>(row - 1)) = 1.0;
  }
  for (size_t col = 0; col < degree; ++col) {
    companion(0, static_cast<Eigen::Index>(col)) =
        -coeffs[col + 1] / lead;
  }

  Eigen::EigenSolver<Eigen::MatrixXd> solver(companion, false);
  if (solver.info() != Eigen::Success) {
    return 0;
  }
  const Eigen::VectorXcd roots = solver.eigenvalues();
  for (size_t idx = 0; idx < degree; ++idx) {
    const auto root = roots(static_cast<Eigen::Index>(idx));
    output_interleaved_complex[2 * idx] = root.real();
    output_interleaved_complex[2 * idx + 1] = root.imag();
  }
  *output_num_roots = degree;

  return IsFinite(output_interleaved_complex, degree * 2) ? 1 : 0;
}

int rustsfm_eigen_fundamental_seven_point(
    const double* points1_xy,
    const double* points2_xy,
    double* output_row_major_models,
    size_t* output_num_models) {
  if (points1_xy == nullptr || points2_xy == nullptr ||
      output_row_major_models == nullptr || output_num_models == nullptr ||
      !IsFinite(points1_xy, 14) || !IsFinite(points2_xy, 14)) {
    return 0;
  }

  Eigen::Matrix<double, 9, 7> A;
  for (size_t i = 0; i < 7; ++i) {
    const double x1 = points1_xy[2 * i];
    const double y1 = points1_xy[2 * i + 1];
    const double x2 = points2_xy[2 * i];
    const double y2 = points2_xy[2 * i + 1];
    const Eigen::Vector3d point2_h(x2, y2, 1.0);
    A.col(static_cast<Eigen::Index>(i)) << x1 * point2_h, y1 * point2_h,
        point2_h;
  }

  const Eigen::Matrix<double, 9, 9> Q = A.fullPivHouseholderQr().matrixQ();
  Eigen::Matrix<double, 9, 1> f1 = Q.col(7);
  const Eigen::Matrix<double, 9, 1> f2 = Q.col(8);
  f1 -= f2;

  const double t0 = f1(4) * f1(8) - f1(5) * f1(7);
  const double t1 = f1(3) * f1(8) - f1(5) * f1(6);
  const double t2 = f1(3) * f1(7) - f1(4) * f1(6);
  const double t3 = f2(4) * f2(8) - f2(5) * f2(7);
  const double t4 = f2(3) * f2(8) - f2(5) * f2(6);
  const double t5 = f2(3) * f2(7) - f2(4) * f2(6);

  Eigen::Vector4d coeffs;
  coeffs(0) = f1(0) * t0 - f1(1) * t1 + f1(2) * t2;
  if (std::abs(coeffs(0)) < 1e-16) {
    *output_num_models = 0;
    return 1;
  }

  coeffs(1) = f2(0) * t0 - f2(1) * t1 + f2(2) * t2 -
              f2(3) * (f1(1) * f1(8) - f1(2) * f1(7)) +
              f2(4) * (f1(0) * f1(8) - f1(2) * f1(6)) -
              f2(5) * (f1(0) * f1(7) - f1(1) * f1(6)) +
              f2(6) * (f1(1) * f1(5) - f1(2) * f1(4)) -
              f2(7) * (f1(0) * f1(5) - f1(2) * f1(3)) +
              f2(8) * (f1(0) * f1(4) - f1(1) * f1(3));
  coeffs(2) = f1(0) * t3 - f1(1) * t4 + f1(2) * t5 -
              f1(3) * (f2(1) * f2(8) - f2(2) * f2(7)) +
              f1(4) * (f2(0) * f2(8) - f2(2) * f2(6)) -
              f1(5) * (f2(0) * f2(7) - f2(1) * f2(6)) +
              f1(6) * (f2(1) * f2(5) - f2(2) * f2(4)) -
              f1(7) * (f2(0) * f2(5) - f2(2) * f2(3)) +
              f1(8) * (f2(0) * f2(4) - f2(1) * f2(3));
  coeffs(3) = f2(0) * t3 - f2(1) * t4 + f2(2) * t5;

  coeffs.tail<3>() /= coeffs(0);

  // Mirror COLMAP 3.13's FindCubicPolynomialRoots for seven-point parity.
  constexpr double k2PiOver3 = 2.09439510239319526263557236234192;
  constexpr double k4PiOver3 = 4.18879020478639052527114472468384;
  const double c2 = coeffs(1);
  const double c1 = coeffs(2);
  const double c0 = coeffs(3);
  const double c2_over_3 = c2 / 3.0;
  const double a = c1 - c2 * c2_over_3;
  double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
  double c = b * b / 4.0 + a * a * a / 27.0;
  Eigen::Vector3d roots = Eigen::Vector3d::Zero();
  int num_roots = 0;
  if (c > 0) {
    c = std::sqrt(c);
    b *= -0.5;
    roots[0] = std::cbrt(b + c) + std::cbrt(b - c) - c2_over_3;
    num_roots = 1;
  } else {
    c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
    const double d = 2.0 * std::sqrt(-a / 3.0);
    const double acos_over_3 = std::acos(c) / 3.0;
    roots[0] = d * std::cos(acos_over_3) - c2_over_3;
    roots[1] = d * std::cos(acos_over_3 - k2PiOver3) - c2_over_3;
    roots[2] = d * std::cos(acos_over_3 - k4PiOver3) - c2_over_3;
    num_roots = 3;
  }

  for (int idx = 0; idx < num_roots; ++idx) {
    const double x = roots[idx];
    const double x2 = x * x;
    const double x3 = x * x2;
    const double dx =
        -(x3 + c2 * x2 + c1 * x + c0) / (3 * x2 + 2 * c2 * x + c1);
    roots[idx] += dx;
  }

  size_t num_models = 0;
  for (int idx = 0; idx < num_roots; ++idx) {
    const double root = roots[idx];
    if (!std::isfinite(root)) {
      continue;
    }
    const Eigen::Matrix<double, 9, 1> F =
        (f1 * root + f2).normalized();
    if (!F.allFinite()) {
      continue;
    }
    const Eigen::Map<const Eigen::Matrix3d> model(F.data());
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        output_row_major_models[num_models * 9 + row * 3 + col] =
            model(row, col);
      }
    }
    num_models += 1;
  }
  *output_num_models = num_models;
  return IsFinite(output_row_major_models, num_models * 9) ? 1 : 0;
}

int rustsfm_eigen_partial_piv_lu_solve_10x10(
    const double* lhs_row_major,
    const double* rhs_row_major,
    double* output_row_major) {
  if (lhs_row_major == nullptr || rhs_row_major == nullptr ||
      output_row_major == nullptr || !IsFinite(lhs_row_major, 100) ||
      !IsFinite(rhs_row_major, 100)) {
    return 0;
  }

  const Eigen::Map<const RowMajorMatrix10> lhs(lhs_row_major);
  const Eigen::Map<const RowMajorMatrix10> rhs(rhs_row_major);
  const Eigen::Matrix<double, 10, 10> solution =
      lhs.partialPivLu().solve(rhs);
  if (!solution.allFinite()) {
    return 0;
  }
  for (int row = 0; row < 10; ++row) {
    for (int col = 0; col < 10; ++col) {
      output_row_major[row * 10 + col] = solution(row, col);
    }
  }
  return IsFinite(output_row_major, 100) ? 1 : 0;
}

int rustsfm_eigen_jacobi_svd_right_null_vector_3(
    const double* matrix_row_major,
    double* output_vector) {
  if (matrix_row_major == nullptr || output_vector == nullptr ||
      !IsFinite(matrix_row_major, 9)) {
    return 0;
  }

  const Eigen::Map<const RowMajorMatrix3> matrix(matrix_row_major);
  const Eigen::JacobiSVD<RowMajorMatrix3> svd(matrix, Eigen::ComputeFullV);
  const Eigen::Vector3d vector = svd.matrixV().rightCols<1>();
  for (int idx = 0; idx < 3; ++idx) {
    output_vector[idx] = vector(idx);
  }
  return IsFinite(output_vector, 3) ? 1 : 0;
}
