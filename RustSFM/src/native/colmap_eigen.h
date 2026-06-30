#ifndef RUSTSFM_COLMAP_EIGEN_H
#define RUSTSFM_COLMAP_EIGEN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int rustsfm_eigen_right_nullspace_9(
    const double* row_major,
    size_t rows,
    size_t nullity,
    double* output);

int rustsfm_eigen_full_piv_right_nullspace_9(
    const double* row_major,
    size_t rows,
    size_t nullity,
    double* output);

int rustsfm_eigen_jacobi_svd_vt_9(
    const double* row_major,
    size_t rows,
    double* output_vt,
    double* output_singular_values,
    size_t* output_num_singular_values);

int rustsfm_eigen_companion_roots(
    const double* coeffs,
    size_t len,
    double* output_interleaved_complex,
    size_t* output_num_roots);

int rustsfm_eigen_fundamental_seven_point(
    const double* points1_xy,
    const double* points2_xy,
    double* output_row_major_models,
    size_t* output_num_models);

int rustsfm_eigen_partial_piv_lu_solve_10x10(
    const double* lhs_row_major,
    const double* rhs_row_major,
    double* output_row_major);

int rustsfm_eigen_jacobi_svd_right_null_vector_3(
    const double* matrix_row_major,
    double* output_vector);

#ifdef __cplusplus
}
#endif

#endif
