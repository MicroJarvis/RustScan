#include "colmap/estimators/essential_matrix.h"
#include "colmap/estimators/fundamental_matrix.h"
#include "colmap/estimators/homography_matrix.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/types.h"
#include "colmap/optim/loransac.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/models.h"
#include "colmap/util/threading.h"
#include "colmap/util/types.h"

#include <sqlite3.h>

#include <Eigen/Core>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Args {
  std::string database;
  int num_threads = 4;
  int batch_size = 1000;
  int random_seed = -1;
  double max_error_px = 4.0;
  int max_trials = 10000;
  int min_trials = 100;
  int min_inliers = 15;
  bool include_inlier_matches = false;
  bool include_models = false;
};

struct ImageRow {
  colmap::image_t image_id = colmap::kInvalidImageId;
  std::string name;
  colmap::camera_t camera_id = colmap::kInvalidCameraId;
};

struct PairJob {
  colmap::image_t image_id1 = colmap::kInvalidImageId;
  colmap::image_t image_id2 = colmap::kInvalidImageId;
  size_t left_index = 0;
  size_t right_index = 0;
  std::string left_image;
  std::string right_image;
  colmap::Camera camera1;
  colmap::Camera camera2;
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  colmap::FeatureMatches matches;
};

struct TraceEvent {
  size_t worker_id = 0;
  size_t dequeue_order = 0;
  size_t complete_order = 0;
  size_t left_index = 0;
  size_t right_index = 0;
  std::string left_image;
  std::string right_image;
  size_t num_matches = 0;
  size_t num_inliers = 0;
  int two_view_config = 0;
  colmap::FeatureMatches inlier_matches;
  bool has_model_details = false;
  bool e_success = false;
  bool f_success = false;
  bool h_success = false;
  size_t e_inliers = 0;
  size_t f_inliers = 0;
  size_t h_inliers = 0;
  std::string selected_source;
  bool has_e_model = false;
  bool has_f_model = false;
  bool has_h_model = false;
  Eigen::Matrix3d e_model = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d f_model = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d h_model = Eigen::Matrix3d::Zero();
};

struct WorkerOutput {
  TraceEvent event;
};

struct DetailedTwoViewGeometry {
  colmap::TwoViewGeometry geometry;
  bool has_model_details = false;
  bool e_success = false;
  bool f_success = false;
  bool h_success = false;
  size_t e_inliers = 0;
  size_t f_inliers = 0;
  size_t h_inliers = 0;
  std::string selected_source;
  bool has_e_model = false;
  bool has_f_model = false;
  bool has_h_model = false;
  Eigen::Matrix3d e_model = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d f_model = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d h_model = Eigen::Matrix3d::Zero();
};

class SqliteDb {
 public:
  explicit SqliteDb(const std::string& path) {
    if (sqlite3_open_v2(
            path.c_str(), &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX, nullptr) !=
        SQLITE_OK) {
      throw std::runtime_error("failed to open database: " + path);
    }
    char* error = nullptr;
    if (sqlite3_exec(db_, "PRAGMA query_only=ON;", nullptr, nullptr, &error) !=
        SQLITE_OK) {
      const std::string message = error == nullptr ? sqlite3_errmsg(db_) : error;
      sqlite3_free(error);
      throw std::runtime_error("failed to enable query_only: " + message);
    }
  }

  ~SqliteDb() {
    if (db_ != nullptr) {
      sqlite3_close(db_);
    }
  }

  SqliteDb(const SqliteDb&) = delete;
  SqliteDb& operator=(const SqliteDb&) = delete;

  sqlite3* get() const { return db_; }

 private:
  sqlite3* db_ = nullptr;
};

class Statement {
 public:
  Statement(sqlite3* db, const std::string& sql) : db_(db) {
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt_, nullptr) !=
        SQLITE_OK) {
      throw std::runtime_error("prepare failed: " + sql + ": " +
                               sqlite3_errmsg(db_));
    }
  }

  ~Statement() {
    if (stmt_ != nullptr) {
      sqlite3_finalize(stmt_);
    }
  }

  Statement(const Statement&) = delete;
  Statement& operator=(const Statement&) = delete;

  sqlite3_stmt* get() const { return stmt_; }

 private:
  sqlite3* db_ = nullptr;
  sqlite3_stmt* stmt_ = nullptr;
};

void BindInt64(sqlite3_stmt* stmt, const int index, const int64_t value) {
  if (sqlite3_bind_int64(stmt, index, value) != SQLITE_OK) {
    throw std::runtime_error("sqlite bind int failed");
  }
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream out;
  for (const char ch : value) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

bool MatrixLooksValid(const Eigen::Matrix3d& matrix) {
  double squared_norm = 0.0;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      const double value = matrix(row, col);
      if (!std::isfinite(value)) {
        return false;
      }
      squared_norm += value * value;
    }
  }
  return squared_norm > 0.0;
}

void WriteMatrixJsonField(const char* name, const Eigen::Matrix3d& matrix) {
  std::cout << ",\"" << name << "\":[";
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      if (row != 0 || col != 0) {
        std::cout << ",";
      }
      std::cout << matrix(row, col);
    }
  }
  std::cout << "]";
}

std::vector<ImageRow> ReadAllImages(sqlite3* db) {
  Statement stmt(db, "SELECT image_id, name, camera_id FROM images");
  std::vector<ImageRow> images;
  while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
    ImageRow row;
    row.image_id =
        static_cast<colmap::image_t>(sqlite3_column_int64(stmt.get(), 0));
    row.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt.get(), 1));
    row.camera_id =
        static_cast<colmap::camera_t>(sqlite3_column_int64(stmt.get(), 2));
    images.push_back(row);
  }
  return images;
}

std::vector<ImageRow> SortedImagesByName(std::vector<ImageRow> images) {
  std::sort(images.begin(), images.end(), [](const auto& left, const auto& right) {
    return left.name < right.name;
  });
  return images;
}

std::unordered_map<colmap::image_t, ImageRow> ImageRowsById(
    const std::vector<ImageRow>& images) {
  std::unordered_map<colmap::image_t, ImageRow> by_id;
  by_id.reserve(images.size());
  for (const ImageRow& image : images) {
    by_id.emplace(image.image_id, image);
  }
  return by_id;
}

std::unordered_map<colmap::image_t, size_t> ImageIndicesById(
    const std::vector<ImageRow>& sorted_images) {
  std::unordered_map<colmap::image_t, size_t> by_id;
  by_id.reserve(sorted_images.size());
  for (size_t idx = 0; idx < sorted_images.size(); ++idx) {
    by_id.emplace(sorted_images[idx].image_id, idx);
  }
  return by_id;
}

std::unordered_map<colmap::camera_t, colmap::Camera> ReadAllCameras(sqlite3* db) {
  Statement stmt(
      db,
      "SELECT camera_id, model, width, height, params, prior_focal_length "
      "FROM cameras");
  std::unordered_map<colmap::camera_t, colmap::Camera> cameras;
  while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
    colmap::Camera camera;
    camera.camera_id =
        static_cast<colmap::camera_t>(sqlite3_column_int64(stmt.get(), 0));
    camera.model_id =
        static_cast<colmap::CameraModelId>(sqlite3_column_int(stmt.get(), 1));
    camera.width = static_cast<size_t>(sqlite3_column_int64(stmt.get(), 2));
    camera.height = static_cast<size_t>(sqlite3_column_int64(stmt.get(), 3));
    const void* data = sqlite3_column_blob(stmt.get(), 4);
    const int bytes = sqlite3_column_bytes(stmt.get(), 4);
    if (data == nullptr || bytes <= 0 ||
        bytes % static_cast<int>(sizeof(double)) != 0) {
      throw std::runtime_error("invalid camera params blob");
    }
    const size_t count = static_cast<size_t>(bytes) / sizeof(double);
    camera.params.resize(count);
    std::memcpy(camera.params.data(), data, static_cast<size_t>(bytes));
    camera.has_prior_focal_length = sqlite3_column_int(stmt.get(), 5) != 0;
    if (!camera.VerifyParams()) {
      throw std::runtime_error("camera params failed COLMAP verification");
    }
    cameras.emplace(camera.camera_id, std::move(camera));
  }
  return cameras;
}

std::vector<Eigen::Vector2d> ReadKeypointPoints(sqlite3* db,
                                                const colmap::image_t image_id) {
  Statement stmt(db,
                 "SELECT rows, cols, data FROM keypoints WHERE image_id = ?1");
  BindInt64(stmt.get(), 1, static_cast<int64_t>(image_id));
  if (sqlite3_step(stmt.get()) != SQLITE_ROW) {
    throw std::runtime_error("missing keypoints for image_id=" +
                             std::to_string(image_id));
  }
  const int rows = sqlite3_column_int(stmt.get(), 0);
  const int cols = sqlite3_column_int(stmt.get(), 1);
  const void* data = sqlite3_column_blob(stmt.get(), 2);
  const int bytes = sqlite3_column_bytes(stmt.get(), 2);
  if (rows < 0 || cols < 2 ||
      bytes != rows * cols * static_cast<int>(sizeof(float))) {
    throw std::runtime_error("invalid keypoints blob");
  }
  const auto* values = static_cast<const float*>(data);
  std::vector<Eigen::Vector2d> points;
  points.reserve(static_cast<size_t>(rows));
  for (int row = 0; row < rows; ++row) {
    points.emplace_back(values[row * cols + 0], values[row * cols + 1]);
  }
  return points;
}

std::unordered_map<colmap::image_t, std::vector<Eigen::Vector2d>>
ReadAllKeypointPoints(sqlite3* db, const std::vector<ImageRow>& images) {
  std::unordered_map<colmap::image_t, std::vector<Eigen::Vector2d>> keypoints;
  keypoints.reserve(images.size());
  for (const ImageRow& image : images) {
    keypoints.emplace(image.image_id, ReadKeypointPoints(db, image.image_id));
  }
  return keypoints;
}

std::vector<std::pair<colmap::image_pair_t, int>> ReadNumMatches(sqlite3* db) {
  Statement stmt(db, "SELECT pair_id, rows FROM matches WHERE rows > 0;");
  std::vector<std::pair<colmap::image_pair_t, int>> pairs;
  while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
    pairs.emplace_back(
        static_cast<colmap::image_pair_t>(sqlite3_column_int64(stmt.get(), 0)),
        static_cast<int>(sqlite3_column_int64(stmt.get(), 1)));
  }
  return pairs;
}

colmap::FeatureMatches ReadMatches(sqlite3* db,
                                   const colmap::image_t image_id1,
                                   const colmap::image_t image_id2) {
  Statement stmt(db, "SELECT rows, cols, data FROM matches WHERE pair_id = ?1");
  BindInt64(stmt.get(),
            1,
            static_cast<int64_t>(
                colmap::ImagePairToPairId(image_id1, image_id2)));
  if (sqlite3_step(stmt.get()) != SQLITE_ROW) {
    throw std::runtime_error("missing matches for image pair");
  }
  const int rows = sqlite3_column_int(stmt.get(), 0);
  const int cols = sqlite3_column_int(stmt.get(), 1);
  const void* data = sqlite3_column_blob(stmt.get(), 2);
  const int bytes = sqlite3_column_bytes(stmt.get(), 2);
  if (rows < 0 || cols != 2 ||
      bytes != rows * cols * static_cast<int>(sizeof(colmap::point2D_t))) {
    throw std::runtime_error("invalid matches blob");
  }
  const auto* values = static_cast<const colmap::point2D_t*>(data);
  const bool swapped = image_id1 > image_id2;
  colmap::FeatureMatches matches;
  matches.reserve(static_cast<size_t>(rows));
  for (int row = 0; row < rows; ++row) {
    if (swapped) {
      matches.emplace_back(values[row * 2 + 1], values[row * 2 + 0]);
    } else {
      matches.emplace_back(values[row * 2 + 0], values[row * 2 + 1]);
    }
  }
  return matches;
}

std::vector<PairJob> LoadPairJobs(sqlite3* db,
                                  const Args& args,
                                  const std::vector<ImageRow>& images,
                                  const std::vector<ImageRow>& sorted_images) {
  const auto image_by_id = ImageRowsById(images);
  const auto index_by_id = ImageIndicesById(sorted_images);
  const auto cameras = ReadAllCameras(db);
  const auto keypoints = ReadAllKeypointPoints(db, images);
  const auto num_matches = ReadNumMatches(db);

  std::vector<PairJob> jobs;
  jobs.reserve(num_matches.size());
  for (const auto& [pair_id, _] : num_matches) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const auto image1 = image_by_id.find(image_id1);
    const auto image2 = image_by_id.find(image_id2);
    if (image1 == image_by_id.end() || image2 == image_by_id.end()) {
      continue;
    }
    PairJob job;
    job.image_id1 = image_id1;
    job.image_id2 = image_id2;
    job.left_index = index_by_id.at(image_id1);
    job.right_index = index_by_id.at(image_id2);
    job.left_image = image1->second.name;
    job.right_image = image2->second.name;
    job.camera1 = cameras.at(image1->second.camera_id);
    job.camera2 = cameras.at(image2->second.camera_id);
    job.points1 = keypoints.at(image_id1);
    job.points2 = keypoints.at(image_id2);
    job.matches = ReadMatches(db, image_id1, image_id2);
    jobs.push_back(std::move(job));
  }
  return jobs;
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int idx = 1; idx < argc; ++idx) {
    const std::string key = argv[idx];
    auto next = [&]() -> std::string {
      if (idx + 1 >= argc) {
        throw std::runtime_error("missing value for " + key);
      }
      return argv[++idx];
    };
    if (key == "--database") {
      args.database = next();
    } else if (key == "--num-threads") {
      args.num_threads = std::stoi(next());
    } else if (key == "--batch-size") {
      args.batch_size = std::stoi(next());
    } else if (key == "--random-seed") {
      args.random_seed = std::stoi(next());
    } else if (key == "--max-error-px") {
      args.max_error_px = std::stod(next());
    } else if (key == "--max-trials") {
      args.max_trials = std::stoi(next());
    } else if (key == "--min-trials") {
      args.min_trials = std::stoi(next());
    } else if (key == "--min-inliers") {
      args.min_inliers = std::stoi(next());
    } else if (key == "--include-inlier-matches") {
      args.include_inlier_matches = true;
    } else if (key == "--include-models") {
      args.include_models = true;
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (args.database.empty()) {
    throw std::runtime_error("usage: colmap_verifier_trace --database DB");
  }
  if (args.num_threads <= 0) {
    throw std::runtime_error("--num-threads must be > 0");
  }
  if (args.batch_size <= 1) {
    throw std::runtime_error("--batch-size must be > 1");
  }
  return args;
}

colmap::TwoViewGeometryOptions TwoViewOptionsFromArgs(const Args& args) {
  colmap::TwoViewGeometryOptions options;
  options.min_num_inliers = args.min_inliers;
  options.ransac_options.max_error = args.max_error_px;
  options.ransac_options.confidence = 0.999;
  options.ransac_options.min_num_trials = args.min_trials;
  options.ransac_options.max_num_trials = args.max_trials;
  options.ransac_options.min_inlier_ratio = 0.25;
  options.ransac_options.random_seed = args.random_seed;
  options.Check();
  return options;
}

colmap::FeatureMatches ExtractInlierMatches(
    const colmap::FeatureMatches& matches,
    const size_t num_inliers,
    const std::vector<char>& inlier_mask) {
  colmap::FeatureMatches inlier_matches(num_inliers);
  size_t out_idx = 0;
  for (size_t idx = 0; idx < matches.size(); ++idx) {
    if (inlier_mask[idx]) {
      inlier_matches[out_idx] = matches[idx];
      out_idx += 1;
    }
  }
  return inlier_matches;
}

DetailedTwoViewGeometry EstimateCalibratedTwoViewGeometryDetailed(
    const PairJob& job,
    const colmap::TwoViewGeometryOptions& options) {
  DetailedTwoViewGeometry detail;
  detail.has_model_details = true;

  const size_t min_num_inliers = static_cast<size_t>(options.min_num_inliers);
  if (job.matches.size() < min_num_inliers) {
    detail.geometry.config =
        colmap::TwoViewGeometry::ConfigurationType::DEGENERATE;
    return detail;
  }

  std::vector<Eigen::Vector2d> matched_img_points1(job.matches.size());
  std::vector<Eigen::Vector2d> matched_img_points2(job.matches.size());
  std::vector<Eigen::Vector3d> matched_cam_rays1(job.matches.size());
  std::vector<Eigen::Vector3d> matched_cam_rays2(job.matches.size());
  for (size_t idx = 0; idx < job.matches.size(); ++idx) {
    const colmap::point2D_t point_idx1 = job.matches[idx].point2D_idx1;
    const colmap::point2D_t point_idx2 = job.matches[idx].point2D_idx2;
    matched_img_points1[idx] = job.points1[point_idx1];
    matched_img_points2[idx] = job.points2[point_idx2];
    if (const std::optional<Eigen::Vector2d> cam_point1 =
            job.camera1.CamFromImg(job.points1[point_idx1]);
        cam_point1) {
      matched_cam_rays1[idx] = cam_point1->homogeneous().normalized();
    } else {
      matched_cam_rays1[idx].setZero();
    }
    if (const std::optional<Eigen::Vector2d> cam_point2 =
            job.camera2.CamFromImg(job.points2[point_idx2]);
        cam_point2) {
      matched_cam_rays2[idx] = cam_point2->homogeneous().normalized();
    } else {
      matched_cam_rays2[idx].setZero();
    }
  }

  auto E_ransac_options = options.ransac_options;
  E_ransac_options.max_error =
      (job.camera1.CamFromImgThreshold(options.ransac_options.max_error) +
       job.camera2.CamFromImgThreshold(options.ransac_options.max_error)) /
      2;

  colmap::LORANSAC<colmap::EssentialMatrixFivePointEstimator,
                   colmap::EssentialMatrixFivePointEstimator>
      E_ransac(E_ransac_options);
  const auto E_report =
      E_ransac.Estimate(matched_cam_rays1, matched_cam_rays2);
  detail.geometry.E = E_report.model;
  detail.e_success = E_report.success;
  detail.e_inliers = E_report.support.num_inliers;
  detail.has_e_model = MatrixLooksValid(E_report.model);
  if (detail.has_e_model) {
    detail.e_model = E_report.model;
  }

  colmap::LORANSAC<colmap::FundamentalMatrixSevenPointEstimator,
                   colmap::FundamentalMatrixEightPointEstimator>
      F_ransac(options.ransac_options);
  const auto F_report =
      F_ransac.Estimate(matched_img_points1, matched_img_points2);
  detail.geometry.F = F_report.model;
  detail.f_success = F_report.success;
  detail.f_inliers = F_report.support.num_inliers;
  detail.has_f_model = MatrixLooksValid(F_report.model);
  if (detail.has_f_model) {
    detail.f_model = F_report.model;
  }

  colmap::LORANSAC<colmap::HomographyMatrixEstimator,
                   colmap::HomographyMatrixEstimator>
      H_ransac(options.ransac_options);
  const auto H_report =
      H_ransac.Estimate(matched_img_points1, matched_img_points2);
  detail.geometry.H = H_report.model;
  detail.h_success = H_report.success;
  detail.h_inliers = H_report.support.num_inliers;
  detail.has_h_model = MatrixLooksValid(H_report.model);
  if (detail.has_h_model) {
    detail.h_model = H_report.model;
  }

  if ((!E_report.success && !F_report.success && !H_report.success) ||
      (E_report.support.num_inliers < min_num_inliers &&
       F_report.support.num_inliers < min_num_inliers &&
       H_report.support.num_inliers < min_num_inliers)) {
    detail.geometry.config =
        colmap::TwoViewGeometry::ConfigurationType::DEGENERATE;
    return detail;
  }

  const double E_F_inlier_ratio =
      static_cast<double>(E_report.support.num_inliers) /
      F_report.support.num_inliers;
  const double H_F_inlier_ratio =
      static_cast<double>(H_report.support.num_inliers) /
      F_report.support.num_inliers;
  const double H_E_inlier_ratio =
      static_cast<double>(H_report.support.num_inliers) /
      E_report.support.num_inliers;

  const std::vector<char>* best_inlier_mask = nullptr;
  size_t num_inliers = 0;

  if (E_report.success && E_F_inlier_ratio > options.min_E_F_inlier_ratio &&
      E_report.support.num_inliers >= min_num_inliers) {
    if (E_report.support.num_inliers >= F_report.support.num_inliers) {
      num_inliers = E_report.support.num_inliers;
      best_inlier_mask = &E_report.inlier_mask;
      detail.selected_source = "essential";
    } else {
      num_inliers = F_report.support.num_inliers;
      best_inlier_mask = &F_report.inlier_mask;
      detail.selected_source = "fundamental";
    }

    if (H_E_inlier_ratio > options.max_H_inlier_ratio) {
      detail.geometry.config =
          colmap::TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
      if (H_report.support.num_inliers > num_inliers) {
        num_inliers = H_report.support.num_inliers;
        best_inlier_mask = &H_report.inlier_mask;
        detail.selected_source = "homography";
      }
    } else {
      detail.geometry.config =
          colmap::TwoViewGeometry::ConfigurationType::CALIBRATED;
    }
  } else if (F_report.success &&
             F_report.support.num_inliers >= min_num_inliers) {
    num_inliers = F_report.support.num_inliers;
    best_inlier_mask = &F_report.inlier_mask;
    detail.selected_source = "fundamental";

    if (H_F_inlier_ratio > options.max_H_inlier_ratio) {
      detail.geometry.config =
          colmap::TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
      if (H_report.support.num_inliers > num_inliers) {
        num_inliers = H_report.support.num_inliers;
        best_inlier_mask = &H_report.inlier_mask;
        detail.selected_source = "homography";
      }
    } else {
      detail.geometry.config =
          colmap::TwoViewGeometry::ConfigurationType::UNCALIBRATED;
    }
  } else if (H_report.success &&
             H_report.support.num_inliers >= min_num_inliers) {
    num_inliers = H_report.support.num_inliers;
    best_inlier_mask = &H_report.inlier_mask;
    detail.selected_source = "homography";
    detail.geometry.config =
        colmap::TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
  } else {
    detail.geometry.config =
        colmap::TwoViewGeometry::ConfigurationType::DEGENERATE;
    return detail;
  }

  if (best_inlier_mask != nullptr) {
    detail.geometry.inlier_matches =
        ExtractInlierMatches(job.matches, num_inliers, *best_inlier_mask);
    if (options.detect_watermark &&
        colmap::DetectWatermarkMatches(job.camera1,
                                       matched_img_points1,
                                       job.camera2,
                                       matched_img_points2,
                                       num_inliers,
                                       *best_inlier_mask,
                                       options)) {
      detail.geometry.config =
          colmap::TwoViewGeometry::ConfigurationType::WATERMARK;
      detail.selected_source = "watermark";
    }
  }

  return detail;
}

DetailedTwoViewGeometry EstimateTwoViewGeometryDetailed(
    const PairJob& job,
    const colmap::TwoViewGeometryOptions& options) {
  if (job.camera1.has_prior_focal_length && job.camera2.has_prior_focal_length &&
      !options.force_H_use && !options.multiple_models &&
      !options.filter_stationary_matches) {
    return EstimateCalibratedTwoViewGeometryDetailed(job, options);
  }

  DetailedTwoViewGeometry detail;
  detail.geometry = colmap::EstimateTwoViewGeometry(job.camera1,
                                                    job.points1,
                                                    job.camera2,
                                                    job.points2,
                                                    job.matches,
                                                    options);
  return detail;
}

TraceEvent EventFromGeometry(const size_t worker_id,
                             const size_t dequeue_order,
                             const PairJob& job,
                             const DetailedTwoViewGeometry& detail,
                             const bool include_inlier_matches) {
  const colmap::TwoViewGeometry& geometry = detail.geometry;
  TraceEvent event;
  event.worker_id = worker_id;
  event.dequeue_order = dequeue_order;
  event.left_index = job.left_index;
  event.right_index = job.right_index;
  event.left_image = job.left_image;
  event.right_image = job.right_image;
  event.num_matches = job.matches.size();
  event.num_inliers = geometry.inlier_matches.size();
  event.two_view_config = geometry.config;
  event.has_model_details = detail.has_model_details;
  event.e_success = detail.e_success;
  event.f_success = detail.f_success;
  event.h_success = detail.h_success;
  event.e_inliers = detail.e_inliers;
  event.f_inliers = detail.f_inliers;
  event.h_inliers = detail.h_inliers;
  event.selected_source = detail.selected_source;
  event.has_e_model = detail.has_e_model;
  event.has_f_model = detail.has_f_model;
  event.has_h_model = detail.has_h_model;
  event.e_model = detail.e_model;
  event.f_model = detail.f_model;
  event.h_model = detail.h_model;
  if (include_inlier_matches) {
    event.inlier_matches = geometry.inlier_matches;
  }
  return event;
}

std::vector<TraceEvent> RunVerifierTrace(const Args& args,
                                         const std::vector<PairJob>& jobs) {
  colmap::JobQueue<PairJob> input_queue;
  colmap::JobQueue<WorkerOutput> output_queue;
  const colmap::TwoViewGeometryOptions options = TwoViewOptionsFromArgs(args);
  std::atomic<size_t> next_dequeue_order{0};

  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(args.num_threads));
  for (int worker_id = 0; worker_id < args.num_threads; ++worker_id) {
    workers.emplace_back([&, worker_id]() {
      while (true) {
        auto input_job = input_queue.Pop();
        if (!input_job.IsValid()) {
          return;
        }
        PairJob& job = input_job.Data();
        const size_t dequeue_order = next_dequeue_order.fetch_add(1);
        DetailedTwoViewGeometry detail;
        if (job.matches.size() >= static_cast<size_t>(options.min_num_inliers)) {
          detail = EstimateTwoViewGeometryDetailed(job, options);
        }
        if (detail.geometry.inlier_matches.size() <
            static_cast<size_t>(options.min_num_inliers)) {
          detail.geometry = colmap::TwoViewGeometry();
        }
        WorkerOutput output;
        output.event = EventFromGeometry(static_cast<size_t>(worker_id),
                                         dequeue_order,
                                         job,
                                         detail,
                                         args.include_inlier_matches);
        if (!output_queue.Push(std::move(output))) {
          return;
        }
      }
    });
  }

  std::vector<TraceEvent> events;
  events.reserve(jobs.size());
  for (size_t batch_begin = 0; batch_begin < jobs.size();
       batch_begin += static_cast<size_t>(args.batch_size)) {
    const size_t batch_end =
        std::min(jobs.size(), batch_begin + static_cast<size_t>(args.batch_size));
    for (size_t idx = batch_begin; idx < batch_end; ++idx) {
      if (!input_queue.Push(jobs[idx])) {
        throw std::runtime_error("failed to push verifier job");
      }
    }
    for (size_t idx = batch_begin; idx < batch_end; ++idx) {
      auto output_job = output_queue.Pop();
      if (!output_job.IsValid()) {
        throw std::runtime_error("invalid verifier output");
      }
      TraceEvent event = std::move(output_job.Data().event);
      event.complete_order = events.size();
      events.push_back(std::move(event));
    }
  }

  input_queue.Stop();
  output_queue.Stop();
  for (std::thread& worker : workers) {
    worker.join();
  }
  return events;
}

void WriteJson(const Args& args,
               const size_t pair_count,
               const std::vector<TraceEvent>& events) {
  size_t verified_pairs = 0;
  size_t total_matches = 0;
  size_t total_inliers = 0;
  for (const TraceEvent& event : events) {
    total_matches += event.num_matches;
    total_inliers += event.num_inliers;
    if (event.num_inliers >= static_cast<size_t>(args.min_inliers)) {
      verified_pairs += 1;
    }
  }

  std::cout << std::setprecision(17);
  std::cout << "{\n";
  std::cout << "  \"database\":\"" << JsonEscape(args.database) << "\",\n";
  std::cout << "  \"mode\":\"colmap_verifier_trace\",\n";
  std::cout << "  \"worker_count\":" << args.num_threads << ",\n";
  std::cout << "  \"batch_size\":" << args.batch_size << ",\n";
  std::cout << "  \"pair_count\":" << pair_count << ",\n";
  std::cout << "  \"matched_pairs\":" << events.size() << ",\n";
  std::cout << "  \"verified_pairs\":" << verified_pairs << ",\n";
  std::cout << "  \"total_matches\":" << total_matches << ",\n";
  std::cout << "  \"total_inliers\":" << total_inliers << ",\n";
  std::cout << "  \"events\":[";
  for (size_t idx = 0; idx < events.size(); ++idx) {
    const TraceEvent& event = events[idx];
    if (idx > 0) {
      std::cout << ",";
    }
    std::cout << "\n    {"
              << "\"worker_id\":" << event.worker_id << ","
              << "\"dequeue_order\":" << event.dequeue_order << ","
              << "\"complete_order\":" << event.complete_order << ","
              << "\"left_index\":" << event.left_index << ","
              << "\"right_index\":" << event.right_index << ","
              << "\"left_image\":\"" << JsonEscape(event.left_image) << "\","
              << "\"right_image\":\"" << JsonEscape(event.right_image) << "\","
              << "\"num_matches\":" << event.num_matches << ","
              << "\"num_inliers\":" << event.num_inliers << ","
              << "\"two_view_config\":" << event.two_view_config << ","
              << "\"has_model_details\":"
              << (event.has_model_details ? "true" : "false") << ","
              << "\"e_success\":" << (event.e_success ? "true" : "false")
              << ","
              << "\"f_success\":" << (event.f_success ? "true" : "false")
              << ","
              << "\"h_success\":" << (event.h_success ? "true" : "false")
              << ","
              << "\"e_inliers\":" << event.e_inliers << ","
              << "\"f_inliers\":" << event.f_inliers << ","
              << "\"h_inliers\":" << event.h_inliers << ","
              << "\"selected_source\":\"" << JsonEscape(event.selected_source)
              << "\"";
    if (args.include_models) {
      std::cout << ",\"has_e_model\":"
                << (event.has_e_model ? "true" : "false")
                << ",\"has_f_model\":"
                << (event.has_f_model ? "true" : "false")
                << ",\"has_h_model\":"
                << (event.has_h_model ? "true" : "false");
      if (event.has_e_model) {
        WriteMatrixJsonField("e_matrix", event.e_model);
      }
      if (event.has_f_model) {
        WriteMatrixJsonField("f_matrix", event.f_model);
      }
      if (event.has_h_model) {
        WriteMatrixJsonField("h_matrix", event.h_model);
      }
    }
    if (args.include_inlier_matches) {
      std::cout << ",\"inlier_matches\":[";
      for (size_t match_idx = 0; match_idx < event.inlier_matches.size();
           ++match_idx) {
        const colmap::FeatureMatch& match = event.inlier_matches[match_idx];
        if (match_idx > 0) {
          std::cout << ",";
        }
        std::cout << "[" << match.point2D_idx1 << "," << match.point2D_idx2
                  << "]";
      }
      std::cout << "]";
    }
    std::cout << "}";
  }
  if (!events.empty()) {
    std::cout << "\n  ";
  }
  std::cout << "]\n";
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    SqliteDb db(args.database);
    const std::vector<ImageRow> images = ReadAllImages(db.get());
    const std::vector<ImageRow> sorted_images = SortedImagesByName(images);
    const std::vector<PairJob> jobs =
        LoadPairJobs(db.get(), args, images, sorted_images);
    const std::vector<TraceEvent> events = RunVerifierTrace(args, jobs);
    WriteJson(args, jobs.size(), events);
  } catch (const std::exception& e) {
    std::cerr << "colmap_verifier_trace: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
