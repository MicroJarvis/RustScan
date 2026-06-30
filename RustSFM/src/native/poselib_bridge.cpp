#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <PoseLib/camera_pose.h>
#include <PoseLib/solvers/gen_relpose_6pt.h>
#include <PoseLib/solvers/gp3p.h>
#include <PoseLib/solvers/p3p.h>

struct RustSfmPoseLibPose {
  double qvec[4];
  double tvec[3];
};

namespace {

struct GrnpObservation {
  Eigen::Quaterniond cam_from_rig_q;
  Eigen::Vector3d cam_from_rig_t;
  Eigen::Vector3d ray_in_cam;
};

Eigen::Matrix3d Skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d skew;
  skew << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return skew;
}

Eigen::Matrix3d EssentialMatrixFromPose(const Eigen::Quaterniond& q,
                                        const Eigen::Vector3d& t) {
  const double norm = t.norm();
  if (norm <= 1.0e-12 || !std::isfinite(norm)) {
    return Eigen::Matrix3d::Zero();
  }
  return Skew(t / norm) * q.normalized().toRotationMatrix();
}

void CopyPose(const poselib::CameraPose& pose, RustSfmPoseLibPose* output) {
  for (int j = 0; j < 4; ++j) {
    output->qvec[j] = pose.q(j);
  }
  for (int j = 0; j < 3; ++j) {
    output->tvec[j] = pose.t(j);
  }
}

void CopyPose(const Eigen::Quaterniond& q,
              const Eigen::Vector3d& t,
              RustSfmPoseLibPose* output) {
  const Eigen::Quaterniond normalized = q.normalized();
  output->qvec[0] = normalized.w();
  output->qvec[1] = normalized.x();
  output->qvec[2] = normalized.y();
  output->qvec[3] = normalized.z();
  output->tvec[0] = t.x();
  output->tvec[1] = t.y();
  output->tvec[2] = t.z();
}

bool ReadObservations(const double* origins1,
                      const double* rays1,
                      const double* origins2,
                      const double* rays2,
                      const double* cam_qvecs1,
                      const double* cam_tvecs1,
                      const double* cam_qvecs2,
                      const double* cam_tvecs2,
                      size_t num_points,
                      std::vector<GrnpObservation>* points1,
                      std::vector<GrnpObservation>* points2) {
  if (origins1 == nullptr || rays1 == nullptr || origins2 == nullptr ||
      rays2 == nullptr) {
    return false;
  }
  points1->resize(num_points);
  points2->resize(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    (*points1)[i].cam_from_rig_t = Eigen::Vector3d(origins1[3 * i + 0],
                                                   origins1[3 * i + 1],
                                                   origins1[3 * i + 2]);
    (*points2)[i].cam_from_rig_t = Eigen::Vector3d(origins2[3 * i + 0],
                                                   origins2[3 * i + 1],
                                                   origins2[3 * i + 2]);
    (*points1)[i].ray_in_cam = Eigen::Vector3d(rays1[3 * i + 0],
                                               rays1[3 * i + 1],
                                               rays1[3 * i + 2]);
    (*points2)[i].ray_in_cam = Eigen::Vector3d(rays2[3 * i + 0],
                                               rays2[3 * i + 1],
                                               rays2[3 * i + 2]);
    if (cam_qvecs1 != nullptr && cam_tvecs1 != nullptr &&
        cam_qvecs2 != nullptr && cam_tvecs2 != nullptr) {
      (*points1)[i].cam_from_rig_q = Eigen::Quaterniond(cam_qvecs1[4 * i + 0],
                                                        cam_qvecs1[4 * i + 1],
                                                        cam_qvecs1[4 * i + 2],
                                                        cam_qvecs1[4 * i + 3])
                                         .normalized();
      (*points2)[i].cam_from_rig_q = Eigen::Quaterniond(cam_qvecs2[4 * i + 0],
                                                        cam_qvecs2[4 * i + 1],
                                                        cam_qvecs2[4 * i + 2],
                                                        cam_qvecs2[4 * i + 3])
                                         .normalized();
      (*points1)[i].cam_from_rig_t = Eigen::Vector3d(cam_tvecs1[3 * i + 0],
                                                     cam_tvecs1[3 * i + 1],
                                                     cam_tvecs1[3 * i + 2]);
      (*points2)[i].cam_from_rig_t = Eigen::Vector3d(cam_tvecs2[3 * i + 0],
                                                     cam_tvecs2[3 * i + 1],
                                                     cam_tvecs2[3 * i + 2]);
    } else {
      (*points1)[i].cam_from_rig_q = Eigen::Quaterniond::Identity();
      (*points2)[i].cam_from_rig_q = Eigen::Quaterniond::Identity();
    }
  }
  return true;
}

void ComposePlueckerData(const Eigen::Quaterniond& rig_from_cam_q,
                         const Eigen::Vector3d& rig_from_cam_t,
                         const Eigen::Vector3d& ray_in_cam,
                         Eigen::Vector3d* origin_in_rig,
                         Eigen::Matrix<double, 6, 1>* pluecker) {
  const Eigen::Vector3d ray_in_rig =
      (rig_from_cam_q.normalized() * ray_in_cam).normalized();
  *origin_in_rig = rig_from_cam_t;
  *pluecker << ray_in_rig, rig_from_cam_t.cross(ray_in_rig);
}

Eigen::Matrix3d CayleyToRotationMatrix(const Eigen::Vector3d& cayley) {
  const double cayley0_sqr = cayley[0] * cayley[0];
  const double cayley1_sqr = cayley[1] * cayley[1];
  const double cayley2_sqr = cayley[2] * cayley[2];
  const double cayley01 = cayley[0] * cayley[1];
  const double cayley12 = cayley[1] * cayley[2];
  const double cayley02 = cayley[0] * cayley[2];

  const double scale = 1 + cayley0_sqr + cayley1_sqr + cayley2_sqr;
  const double inv_scale = 1.0 / scale;

  Eigen::Matrix3d R;
  R(0, 0) = inv_scale * (1 + cayley0_sqr - cayley1_sqr - cayley2_sqr);
  R(0, 1) = inv_scale * (2 * (cayley01 - cayley[2]));
  R(0, 2) = inv_scale * (2 * (cayley02 + cayley[1]));
  R(1, 0) = inv_scale * (2 * (cayley01 + cayley[2]));
  R(1, 1) = inv_scale * (1 - cayley0_sqr + cayley1_sqr - cayley2_sqr);
  R(1, 2) = inv_scale * (2 * (cayley12 - cayley[0]));
  R(2, 0) = inv_scale * (2 * (cayley02 - cayley[1]));
  R(2, 1) = inv_scale * (2 * (cayley12 + cayley[0]));
  R(2, 2) = inv_scale * (1 - cayley0_sqr - cayley1_sqr + cayley2_sqr);
  return R;
}

Eigen::Vector3d RotationMatrixToCayley(const Eigen::Matrix3d& R) {
  const Eigen::Matrix3d C = (R - Eigen::Matrix3d::Identity()) *
                            (R + Eigen::Matrix3d::Identity()).inverse();
  return Eigen::Vector3d(-C(1, 2), C(0, 2), -C(0, 1));
}

Eigen::Vector3d ComputeRotationBetweenPoints(
    const std::vector<Eigen::Matrix<double, 6, 1>>& plueckers1,
    const std::vector<Eigen::Matrix<double, 6, 1>>& plueckers2) {
  Eigen::Vector3d points_center1 = Eigen::Vector3d::Zero();
  Eigen::Vector3d points_center2 = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < plueckers1.size(); i++) {
    points_center1 += plueckers1[i].head<3>();
    points_center2 += plueckers2[i].head<3>();
  }
  points_center1 /= plueckers1.size();
  points_center2 /= plueckers1.size();

  Eigen::Matrix3d Hcross = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < plueckers1.size(); i++) {
    const Eigen::Vector3d f1 = plueckers1[i].head<3>() - points_center1;
    const Eigen::Vector3d f2 = plueckers2[i].head<3>() - points_center2;
    Hcross += f2 * f1.transpose();
  }

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      Hcross, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Matrix3d& V = svd.matrixV();
  const Eigen::Matrix3d& U = svd.matrixU();

  Eigen::Matrix3d R = V * U.transpose();
  if (R.determinant() < 0) {
    Eigen::Matrix3d V_prime;
    V_prime.col(0) = V.col(0);
    V_prime.col(1) = V.col(1);
    V_prime.col(2) = -V.col(2);
    R = V_prime * U.transpose();
  }
  return RotationMatrixToCayley(R);
}

Eigen::Matrix4d ComposeG(const Eigen::Matrix3d& xxF,
                         const Eigen::Matrix3d& yyF,
                         const Eigen::Matrix3d& zzF,
                         const Eigen::Matrix3d& xyF,
                         const Eigen::Matrix3d& yzF,
                         const Eigen::Matrix3d& zxF,
                         const Eigen::Matrix<double, 3, 9>& x1P,
                         const Eigen::Matrix<double, 3, 9>& y1P,
                         const Eigen::Matrix<double, 3, 9>& z1P,
                         const Eigen::Matrix<double, 3, 9>& x2P,
                         const Eigen::Matrix<double, 3, 9>& y2P,
                         const Eigen::Matrix<double, 3, 9>& z2P,
                         const Eigen::Matrix<double, 9, 9>& m11P,
                         const Eigen::Matrix<double, 9, 9>& m12P,
                         const Eigen::Matrix<double, 9, 9>& m22P,
                         const Eigen::Vector3d& rotation) {
  const Eigen::Matrix3d R = CayleyToRotationMatrix(rotation);
  Eigen::Matrix<double, 1, 9> R_rows;
  R_rows << R.row(0), R.row(1), R.row(2);
  Eigen::Matrix<double, 9, 1> R_cols;
  R_cols << R.col(0), R.col(1), R.col(2);

  const Eigen::Vector3d xxFr1t = xxF * R.row(1).transpose();
  const Eigen::Vector3d yyFr0t = yyF * R.row(0).transpose();
  const Eigen::Vector3d zzFr0t = zzF * R.row(0).transpose();
  const Eigen::Vector3d yzFr0t = yzF * R.row(0).transpose();
  const Eigen::Vector3d xyFr1t = xyF * R.row(1).transpose();
  const Eigen::Vector3d xyFr2t = xyF * R.row(2).transpose();
  const Eigen::Vector3d zxFr1t = zxF * R.row(1).transpose();
  const Eigen::Vector3d zxFr2t = zxF * R.row(2).transpose();

  const Eigen::Vector3d x1PC = x1P * R_cols;
  const Eigen::Vector3d y1PC = y1P * R_cols;
  const Eigen::Vector3d z1PC = z1P * R_cols;
  const Eigen::Vector3d x2PR = x2P * R_rows.transpose();
  const Eigen::Vector3d y2PR = y2P * R_rows.transpose();
  const Eigen::Vector3d z2PR = z2P * R_rows.transpose();

  Eigen::Matrix4d G;
  G(0, 0) = R.row(2) * yyF * R.row(2).transpose();
  G(0, 0) += -2.0 * R.row(2) * yzF * R.row(1).transpose();
  G(0, 0) += R.row(1) * zzF * R.row(1).transpose();
  G(0, 1) = R.row(2) * yzFr0t;
  G(0, 1) += -1.0 * R.row(2) * xyFr2t;
  G(0, 1) += -1.0 * R.row(1) * zzFr0t;
  G(0, 1) += R.row(1) * zxFr2t;
  G(0, 2) = R.row(2) * xyFr1t;
  G(0, 2) += -1.0 * R.row(2) * yyFr0t;
  G(0, 2) += -1.0 * R.row(1) * zxFr1t;
  G(0, 2) += R.row(1) * yzFr0t;
  G(1, 1) = R.row(0) * zzFr0t;
  G(1, 1) += -2.0 * R.row(0) * zxFr2t;
  G(1, 1) += R.row(2) * xxF * R.row(2).transpose();
  G(1, 2) = R.row(0) * zxFr1t;
  G(1, 2) += -1.0 * R.row(0) * yzFr0t;
  G(1, 2) += -1.0 * R.row(2) * xxFr1t;
  G(1, 2) += R.row(0) * xyFr2t;
  G(2, 2) = R.row(1) * xxFr1t;
  G(2, 2) += -2.0 * R.row(0) * xyFr1t;
  G(2, 2) += R.row(0) * yyFr0t;
  G(1, 0) = G(0, 1);
  G(2, 0) = G(0, 2);
  G(2, 1) = G(1, 2);
  G(0, 3) = R.row(2) * y1PC;
  G(0, 3) += R.row(2) * y2PR;
  G(0, 3) += -1.0 * R.row(1) * z1PC;
  G(0, 3) += -1.0 * R.row(1) * z2PR;
  G(1, 3) = R.row(0) * z1PC;
  G(1, 3) += R.row(0) * z2PR;
  G(1, 3) += -1.0 * R.row(2) * x1PC;
  G(1, 3) += -1.0 * R.row(2) * x2PR;
  G(2, 3) = R.row(1) * x1PC;
  G(2, 3) += R.row(1) * x2PR;
  G(2, 3) += -1.0 * R.row(0) * y1PC;
  G(2, 3) += -1.0 * R.row(0) * y2PR;
  G(3, 3) = -1.0 * R_cols.transpose() * m11P * R_cols;
  G(3, 3) += -1.0 * R_rows * m22P * R_rows.transpose();
  G(3, 3) += -2.0 * R_rows * m12P * R_cols;
  G(3, 0) = G(0, 3);
  G(3, 1) = G(1, 3);
  G(3, 2) = G(2, 3);
  return G;
}

Eigen::Vector4d ComputeEigenValue(const Eigen::Matrix3d& xxF,
                                  const Eigen::Matrix3d& yyF,
                                  const Eigen::Matrix3d& zzF,
                                  const Eigen::Matrix3d& xyF,
                                  const Eigen::Matrix3d& yzF,
                                  const Eigen::Matrix3d& zxF,
                                  const Eigen::Matrix<double, 3, 9>& x1P,
                                  const Eigen::Matrix<double, 3, 9>& y1P,
                                  const Eigen::Matrix<double, 3, 9>& z1P,
                                  const Eigen::Matrix<double, 3, 9>& x2P,
                                  const Eigen::Matrix<double, 3, 9>& y2P,
                                  const Eigen::Matrix<double, 3, 9>& z2P,
                                  const Eigen::Matrix<double, 9, 9>& m11P,
                                  const Eigen::Matrix<double, 9, 9>& m12P,
                                  const Eigen::Matrix<double, 9, 9>& m22P,
                                  const Eigen::Vector3d& rotation) {
  const Eigen::Matrix4d G = ComposeG(xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P,
                                     z1P, x2P, y2P, z2P, m11P, m12P, m22P,
                                     rotation);
  const double B = -G(3, 3) - G(2, 2) - G(1, 1) - G(0, 0);
  const double C = -G(2, 3) * G(2, 3) + G(2, 2) * G(3, 3) -
                   G(1, 3) * G(1, 3) - G(1, 2) * G(1, 2) +
                   G(1, 1) * G(3, 3) + G(1, 1) * G(2, 2) -
                   G(0, 3) * G(0, 3) - G(0, 2) * G(0, 2) -
                   G(0, 1) * G(0, 1) + G(0, 0) * G(3, 3) +
                   G(0, 0) * G(2, 2) + G(0, 0) * G(1, 1);
  const double D = G(1, 3) * G(1, 3) * G(2, 2) -
                   2.0 * G(1, 2) * G(1, 3) * G(2, 3) +
                   G(1, 2) * G(1, 2) * G(3, 3) +
                   G(1, 1) * G(2, 3) * G(2, 3) -
                   G(1, 1) * G(2, 2) * G(3, 3) +
                   G(0, 3) * G(0, 3) * G(2, 2) +
                   G(0, 3) * G(0, 3) * G(1, 1) -
                   2.0 * G(0, 2) * G(0, 3) * G(2, 3) +
                   G(0, 2) * G(0, 2) * G(3, 3) +
                   G(0, 2) * G(0, 2) * G(1, 1) -
                   2.0 * G(0, 1) * G(0, 3) * G(1, 3) -
                   2.0 * G(0, 1) * G(0, 2) * G(1, 2) +
                   G(0, 1) * G(0, 1) * G(3, 3) +
                   G(0, 1) * G(0, 1) * G(2, 2) +
                   G(0, 0) * G(2, 3) * G(2, 3) -
                   G(0, 0) * G(2, 2) * G(3, 3) +
                   G(0, 0) * G(1, 3) * G(1, 3) +
                   G(0, 0) * G(1, 2) * G(1, 2) -
                   G(0, 0) * G(1, 1) * G(3, 3) -
                   G(0, 0) * G(1, 1) * G(2, 2);
  const double E =
      G(0, 3) * G(0, 3) * G(1, 2) * G(1, 2) -
      G(0, 3) * G(0, 3) * G(1, 1) * G(2, 2) -
      2.0 * G(0, 2) * G(0, 3) * G(1, 2) * G(1, 3) +
      2.0 * G(0, 2) * G(0, 3) * G(1, 1) * G(2, 3) +
      G(0, 2) * G(0, 2) * G(1, 3) * G(1, 3) -
      G(0, 2) * G(0, 2) * G(1, 1) * G(3, 3) +
      2.0 * G(0, 1) * G(0, 3) * G(1, 3) * G(2, 2) -
      2.0 * G(0, 1) * G(0, 3) * G(1, 2) * G(2, 3) -
      2.0 * G(0, 1) * G(0, 2) * G(1, 3) * G(2, 3) +
      2.0 * G(0, 1) * G(0, 2) * G(1, 2) * G(3, 3) +
      G(0, 1) * G(0, 1) * G(2, 3) * G(2, 3) -
      G(0, 1) * G(0, 1) * G(2, 2) * G(3, 3) -
      G(0, 0) * G(1, 3) * G(1, 3) * G(2, 2) +
      2.0 * G(0, 0) * G(1, 2) * G(1, 3) * G(2, 3) -
      G(0, 0) * G(1, 2) * G(1, 2) * G(3, 3) -
      G(0, 0) * G(1, 1) * G(2, 3) * G(2, 3) +
      G(0, 0) * G(1, 1) * G(2, 2) * G(3, 3);

  const double alpha = -0.375 * B * B + C;
  const double beta = B * B * B / 8.0 - B * C / 2.0 + D;
  const double gamma = -0.01171875 * B * B * B * B + B * B * C / 16.0 -
                       B * D / 4.0 + E;
  const double p = -alpha * alpha / 12.0 - gamma;
  const double q = -alpha * alpha * alpha / 108.0 + alpha * gamma / 3.0 -
                   beta * beta / 8.0;
  const double helper1 = -p * p * p / 27.0;
  const double theta2 = std::pow(helper1, 1.0 / 3.0);
  const double theta1 =
      std::sqrt(theta2) *
      std::cos((1.0 / 3.0) * std::acos((-q / 2.0) / std::sqrt(helper1)));
  const double y = -(5.0 / 6.0) * alpha -
                   ((1.0 / 3.0) * p * theta1 - theta1 * theta2) / theta2;
  const double w = std::sqrt(alpha + 2.0 * y);

  Eigen::Vector4d roots;
  roots(0) = -B / 4.0 + 0.5 * w +
             0.5 * std::sqrt(-3.0 * alpha - 2.0 * y - 2.0 * beta / w);
  roots(1) = -B / 4.0 + 0.5 * w -
             0.5 * std::sqrt(-3.0 * alpha - 2.0 * y - 2.0 * beta / w);
  roots(2) = -B / 4.0 - 0.5 * w +
             0.5 * std::sqrt(-3.0 * alpha - 2.0 * y + 2.0 * beta / w);
  roots(3) = -B / 4.0 - 0.5 * w -
             0.5 * std::sqrt(-3.0 * alpha - 2.0 * y + 2.0 * beta / w);
  return roots;
}

double ComputeCost(const Eigen::Matrix3d& xxF,
                   const Eigen::Matrix3d& yyF,
                   const Eigen::Matrix3d& zzF,
                   const Eigen::Matrix3d& xyF,
                   const Eigen::Matrix3d& yzF,
                   const Eigen::Matrix3d& zxF,
                   const Eigen::Matrix<double, 3, 9>& x1P,
                   const Eigen::Matrix<double, 3, 9>& y1P,
                   const Eigen::Matrix<double, 3, 9>& z1P,
                   const Eigen::Matrix<double, 3, 9>& x2P,
                   const Eigen::Matrix<double, 3, 9>& y2P,
                   const Eigen::Matrix<double, 3, 9>& z2P,
                   const Eigen::Matrix<double, 9, 9>& m11P,
                   const Eigen::Matrix<double, 9, 9>& m12P,
                   const Eigen::Matrix<double, 9, 9>& m22P,
                   const Eigen::Vector3d& rotation,
                   const int step) {
  const Eigen::Vector4d roots = ComputeEigenValue(
      xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P, m11P, m12P,
      m22P, rotation);
  return step == 0 ? roots[2] : roots[3];
}

Eigen::Vector3d ComputeJacobian(const Eigen::Matrix3d& xxF,
                                const Eigen::Matrix3d& yyF,
                                const Eigen::Matrix3d& zzF,
                                const Eigen::Matrix3d& xyF,
                                const Eigen::Matrix3d& yzF,
                                const Eigen::Matrix3d& zxF,
                                const Eigen::Matrix<double, 3, 9>& x1P,
                                const Eigen::Matrix<double, 3, 9>& y1P,
                                const Eigen::Matrix<double, 3, 9>& z1P,
                                const Eigen::Matrix<double, 3, 9>& x2P,
                                const Eigen::Matrix<double, 3, 9>& y2P,
                                const Eigen::Matrix<double, 3, 9>& z2P,
                                const Eigen::Matrix<double, 9, 9>& m11P,
                                const Eigen::Matrix<double, 9, 9>& m12P,
                                const Eigen::Matrix<double, 9, 9>& m22P,
                                const Eigen::Vector3d& rotation,
                                const double current_cost,
                                const int step) {
  (void)current_cost;
  Eigen::Vector3d jacobian;
  constexpr double kStepSize = 1e-8;
  for (int j = 0; j < 3; j++) {
    Eigen::Vector3d cayley_j = rotation;
    cayley_j[j] += kStepSize;
    jacobian(j) = ComputeCost(xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P,
                              x2P, y2P, z2P, m11P, m12P, m22P, cayley_j,
                              step) -
                  current_cost;
  }
  return jacobian;
}

std::vector<RustSfmPoseLibPose> EstimateGr8P(
    const std::vector<GrnpObservation>& points1,
    const std::vector<GrnpObservation>& points2,
    uint32_t random_seed) {
  std::vector<RustSfmPoseLibPose> output;
  if (points1.size() < 8 || points1.size() != points2.size()) {
    return output;
  }

  std::vector<Eigen::Vector3d> origins_in_rig1(points1.size());
  std::vector<Eigen::Vector3d> origins_in_rig2(points1.size());
  std::vector<Eigen::Matrix<double, 6, 1>> plueckers1(points1.size());
  std::vector<Eigen::Matrix<double, 6, 1>> plueckers2(points1.size());
  for (size_t i = 0; i < points1.size(); ++i) {
    const Eigen::Quaterniond rig_from_cam1_q =
        points1[i].cam_from_rig_q.conjugate();
    const Eigen::Vector3d rig_from_cam1_t =
        -(rig_from_cam1_q * points1[i].cam_from_rig_t);
    const Eigen::Quaterniond rig_from_cam2_q =
        points2[i].cam_from_rig_q.conjugate();
    const Eigen::Vector3d rig_from_cam2_t =
        -(rig_from_cam2_q * points2[i].cam_from_rig_t);
    ComposePlueckerData(rig_from_cam1_q, rig_from_cam1_t,
                        points1[i].ray_in_cam, &origins_in_rig1[i],
                        &plueckers1[i]);
    ComposePlueckerData(rig_from_cam2_q, rig_from_cam2_t,
                        points2[i].ray_in_cam, &origins_in_rig2[i],
                        &plueckers2[i]);
  }

  Eigen::Matrix3d xxF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d yyF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d zzF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d xyF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d yzF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d zxF = Eigen::Matrix3d::Zero();
  Eigen::Matrix<double, 3, 9> x1P = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> y1P = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> z1P = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> x2P = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> y2P = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> z2P = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 9, 9> m11P = Eigen::Matrix<double, 9, 9>::Zero();
  Eigen::Matrix<double, 9, 9> m12P = Eigen::Matrix<double, 9, 9>::Zero();
  Eigen::Matrix<double, 9, 9> m22P = Eigen::Matrix<double, 9, 9>::Zero();

  for (size_t i = 0; i < points1.size(); ++i) {
    const Eigen::Vector3d f1 = plueckers1[i].head<3>();
    const Eigen::Vector3d f2 = plueckers2[i].head<3>();
    const Eigen::Vector3d& t1 = origins_in_rig1[i];
    const Eigen::Vector3d& t2 = origins_in_rig2[i];
    const Eigen::Matrix3d F = f2 * f2.transpose();
    xxF += f1[0] * f1[0] * F;
    yyF += f1[1] * f1[1] * F;
    zzF += f1[2] * f1[2] * F;
    xyF += f1[0] * f1[1] * F;
    yzF += f1[1] * f1[2] * F;
    zxF += f1[2] * f1[0] * F;

    Eigen::Matrix<double, 9, 1> ff1;
    ff1 << f1[0] * (f2[1] * t2[2] - f2[2] * t2[1]),
        f1[1] * (f2[1] * t2[2] - f2[2] * t2[1]),
        f1[2] * (f2[1] * t2[2] - f2[2] * t2[1]),
        f1[0] * (f2[2] * t2[0] - f2[0] * t2[2]),
        f1[1] * (f2[2] * t2[0] - f2[0] * t2[2]),
        f1[2] * (f2[2] * t2[0] - f2[0] * t2[2]),
        f1[0] * (f2[0] * t2[1] - f2[1] * t2[0]),
        f1[1] * (f2[0] * t2[1] - f2[1] * t2[0]),
        f1[2] * (f2[0] * t2[1] - f2[1] * t2[0]);
    x1P += f1[0] * f2 * ff1.transpose();
    y1P += f1[1] * f2 * ff1.transpose();
    z1P += f1[2] * f2 * ff1.transpose();

    Eigen::Matrix<double, 9, 1> ff2;
    ff2 << f2[0] * (f1[1] * t1[2] - f1[2] * t1[1]),
        f2[1] * (f1[1] * t1[2] - f1[2] * t1[1]),
        f2[2] * (f1[1] * t1[2] - f1[2] * t1[1]),
        f2[0] * (f1[2] * t1[0] - f1[0] * t1[2]),
        f2[1] * (f1[2] * t1[0] - f1[0] * t1[2]),
        f2[2] * (f1[2] * t1[0] - f1[0] * t1[2]),
        f2[0] * (f1[0] * t1[1] - f1[1] * t1[0]),
        f2[1] * (f1[0] * t1[1] - f1[1] * t1[0]),
        f2[2] * (f1[0] * t1[1] - f1[1] * t1[0]);
    x2P += f1[0] * f2 * ff2.transpose();
    y2P += f1[1] * f2 * ff2.transpose();
    z2P += f1[2] * f2 * ff2.transpose();
    m11P -= ff1 * ff1.transpose();
    m22P -= ff2 * ff2.transpose();
    m12P -= ff2 * ff1.transpose();
  }

  const Eigen::Vector3d initial_rotation =
      ComputeRotationBetweenPoints(plueckers1, plueckers2);
  std::mt19937 rng(random_seed);
  std::uniform_real_distribution<double> perturb_small(-0.3, 0.3);
  std::uniform_real_distribution<double> perturb_large(-0.6, 0.6);

  const double kMinLambda = 0.00001;
  const double kMaxLambda = 0.08;
  const double kLambdaModifier = 2.0;
  const int kMaxNumIterations = 50;
  const bool kDisableIncrements = true;

  Eigen::Vector3d rotation;
  int num_random_trials = 0;
  double perturbation_amplitude = 0.3;
  while (num_random_trials < 5) {
    if (num_random_trials > 2) {
      perturbation_amplitude = 0.6;
    }
    if (num_random_trials == 0) {
      rotation = initial_rotation;
    } else {
      auto& distribution =
          perturbation_amplitude > 0.3 ? perturb_large : perturb_small;
      rotation = initial_rotation +
                 Eigen::Vector3d(distribution(rng), distribution(rng),
                                 distribution(rng));
    }

    double lambda = 0.01;
    int num_iterations = 0;
    double smallest_eigen_value = ComputeCost(
        xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P, m11P,
        m12P, m22P, rotation, 1);
    for (int iter = 0; iter < kMaxNumIterations; ++iter) {
      const Eigen::Vector3d jacobian =
          ComputeJacobian(xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P,
                          y2P, z2P, m11P, m12P, m22P, rotation,
                          smallest_eigen_value, 1);
      const double jacobian_norm = jacobian.norm();
      if (jacobian_norm <= 1.0e-15 || !std::isfinite(jacobian_norm)) {
        break;
      }
      const Eigen::Vector3d normalized_jacobian = jacobian / jacobian_norm;
      Eigen::Vector3d sampling_point = rotation - lambda * normalized_jacobian;
      double sampling_eigen_value = ComputeCost(
          xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P, m11P,
          m12P, m22P, sampling_point, 1);
      if (num_iterations == 0 || !kDisableIncrements) {
        while (sampling_eigen_value < smallest_eigen_value) {
          smallest_eigen_value = sampling_eigen_value;
          if (lambda * kLambdaModifier > kMaxLambda) {
            break;
          }
          lambda *= kLambdaModifier;
          sampling_point = rotation - lambda * normalized_jacobian;
          sampling_eigen_value = ComputeCost(
              xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P,
              m11P, m12P, m22P, sampling_point, 1);
        }
      }
      while (sampling_eigen_value > smallest_eigen_value) {
        lambda /= kLambdaModifier;
        sampling_point = rotation - lambda * normalized_jacobian;
        sampling_eigen_value = ComputeCost(
            xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P,
            m11P, m12P, m22P, sampling_point, 1);
      }
      rotation = sampling_point;
      smallest_eigen_value = sampling_eigen_value;
      if (lambda < kMinLambda) {
        break;
      }
    }

    if (rotation.norm() < 0.01) {
      const double eigen_value2 = ComputeCost(
          xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P, m11P,
          m12P, m22P, rotation, 0);
      if (eigen_value2 > 0.001) {
        num_random_trials += 1;
      } else {
        break;
      }
    } else {
      break;
    }
  }

  const Eigen::Matrix3d R = CayleyToRotationMatrix(rotation).transpose();
  const Eigen::Matrix4d G =
      ComposeG(xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P,
               m11P, m12P, m22P, rotation);
  const Eigen::EigenSolver<Eigen::Matrix4d> eigen_solver_G(G, true);
  const Eigen::Matrix4cd V = eigen_solver_G.eigenvectors();
  const Eigen::Matrix<double, 3, 4> VV = V.real().colwise().hnormalized();

  output.resize(4);
  for (int i = 0; i < 4; ++i) {
    CopyPose(Eigen::Quaterniond(R), -R * VV.col(i), &output[i]);
  }
  return output;
}

}  // namespace

extern "C" {

int rustsfm_poselib_gen_relpose_6pt(const double* origins1,
                                    const double* rays1,
                                    const double* origins2,
                                    const double* rays2,
                                    size_t num_points,
                                    RustSfmPoseLibPose* output,
                                    size_t max_output,
                                    size_t* num_output) {
  if (num_output == nullptr) {
    return -1;
  }
  *num_output = 0;
  if (origins1 == nullptr || rays1 == nullptr || origins2 == nullptr ||
      rays2 == nullptr || output == nullptr || num_points != 6) {
    return -2;
  }

  try {
    std::vector<Eigen::Vector3d> p1(num_points);
    std::vector<Eigen::Vector3d> x1(num_points);
    std::vector<Eigen::Vector3d> p2(num_points);
    std::vector<Eigen::Vector3d> x2(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      p1[i] = Eigen::Vector3d(origins1[3 * i + 0],
                              origins1[3 * i + 1],
                              origins1[3 * i + 2]);
      x1[i] = Eigen::Vector3d(rays1[3 * i + 0],
                              rays1[3 * i + 1],
                              rays1[3 * i + 2]);
      p2[i] = Eigen::Vector3d(origins2[3 * i + 0],
                              origins2[3 * i + 1],
                              origins2[3 * i + 2]);
      x2[i] = Eigen::Vector3d(rays2[3 * i + 0],
                              rays2[3 * i + 1],
                              rays2[3 * i + 2]);
    }

    std::vector<poselib::CameraPose> poses;
    const int result = poselib::gen_relpose_6pt(p1, x1, p2, x2, &poses);
    if (result < 0) {
      return result;
    }

    const size_t count = std::min(max_output, poses.size());
    for (size_t i = 0; i < count; ++i) {
      CopyPose(poses[i], &output[i]);
    }
    *num_output = count;
    return 0;
  } catch (const std::exception&) {
    return -3;
  } catch (...) {
    return -4;
  }
}

int rustsfm_poselib_gen_relpose_8pt(const double* origins1,
                                    const double* rays1,
                                    const double* origins2,
                                    const double* rays2,
                                    const double* cam_qvecs1,
                                    const double* cam_tvecs1,
                                    const double* cam_qvecs2,
                                    const double* cam_tvecs2,
                                    size_t num_points,
                                    uint32_t random_seed,
                                    RustSfmPoseLibPose* output,
                                    size_t max_output,
                                    size_t* num_output) {
  if (num_output == nullptr) {
    return -1;
  }
  *num_output = 0;
  if (output == nullptr || num_points < 8) {
    return -2;
  }

  try {
    std::vector<GrnpObservation> points1;
    std::vector<GrnpObservation> points2;
    if (!ReadObservations(origins1,
                          rays1,
                          origins2,
                          rays2,
                          cam_qvecs1,
                          cam_tvecs1,
                          cam_qvecs2,
                          cam_tvecs2,
                          num_points,
                          &points1,
                          &points2)) {
      return -2;
    }
    const std::vector<RustSfmPoseLibPose> poses =
        EstimateGr8P(points1, points2, random_seed);
    const size_t count = std::min(max_output, poses.size());
    for (size_t i = 0; i < count; ++i) {
      output[i] = poses[i];
    }
    *num_output = count;
    return 0;
  } catch (const std::exception&) {
    return -3;
  } catch (...) {
    return -4;
  }
}

int rustsfm_poselib_gp3p(const double* origins,
                         const double* rays,
                         const double* points3d,
                         size_t num_points,
                         RustSfmPoseLibPose* output,
                         size_t max_output,
                         size_t* num_output) {
  if (num_output == nullptr) {
    return -1;
  }
  *num_output = 0;
  if (origins == nullptr || rays == nullptr || points3d == nullptr ||
      output == nullptr || num_points != 3) {
    return -2;
  }

  try {
    std::vector<Eigen::Vector3d> p(num_points);
    std::vector<Eigen::Vector3d> x(num_points);
    std::vector<Eigen::Vector3d> X(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      p[i] = Eigen::Vector3d(origins[3 * i + 0],
                             origins[3 * i + 1],
                             origins[3 * i + 2]);
      x[i] = Eigen::Vector3d(rays[3 * i + 0],
                             rays[3 * i + 1],
                             rays[3 * i + 2]);
      X[i] = Eigen::Vector3d(points3d[3 * i + 0],
                             points3d[3 * i + 1],
                             points3d[3 * i + 2]);
    }

    std::vector<poselib::CameraPose> poses;
    if (p[0].isApprox(p[1], 1.0e-6) && p[0].isApprox(p[2], 1.0e-6)) {
      const int result = poselib::p3p(x, X, &poses);
      if (result < 0) {
        return result;
      }
      for (poselib::CameraPose& pose : poses) {
        pose.t += p[0];
      }
    } else {
      const int result = poselib::gp3p(p, x, X, &poses);
      if (result < 0) {
        return result;
      }
    }

    const size_t count = std::min(max_output, poses.size());
    for (size_t i = 0; i < count; ++i) {
      CopyPose(poses[i], &output[i]);
    }
    *num_output = count;
    return 0;
  } catch (const std::exception&) {
    return -3;
  } catch (...) {
    return -4;
  }
}

}  // extern "C"
