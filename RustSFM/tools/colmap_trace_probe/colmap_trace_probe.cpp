#include "colmap/estimators/essential_matrix.h"
#include "colmap/estimators/fundamental_matrix.h"
#include "colmap/estimators/homography_matrix.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/math/random.h"
#include "colmap/optim/random_sampler.h"
#include "colmap/optim/ransac.h"
#include "colmap/optim/support_measurement.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/sensor/models.h"
#include "colmap/util/types.h"

#include <sqlite3.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr size_t kMaxTraceBestUpdates = 128;
constexpr size_t kMaxBoundaryResiduals = 16;

struct Args {
  std::string database;
  std::string image1;
  std::string image2;
  int random_seed = -1;
  double max_error_px = 4.0;
  int max_trials = 10000;
  int min_trials = 100;
  int min_inliers = 15;
  int init_min_num_inliers = 100;
  double init_min_tri_angle_deg = 16.0;
  double init_max_forward_motion = 0.95;
};

struct ImageRow {
  uint32_t image_id = 0;
  std::string name;
  uint32_t camera_id = 0;
};

struct CameraRow {
  colmap::Camera camera;
};

struct Keypoint {
  double x = 0.0;
  double y = 0.0;
};

struct MatchRow {
  uint32_t point2D_idx1 = 0;
  uint32_t point2D_idx2 = 0;
};

struct PreparedPair {
  std::vector<Eigen::Vector2d> img1;
  std::vector<Eigen::Vector2d> img2;
  std::vector<Eigen::Vector3d> rays1;
  std::vector<Eigen::Vector3d> rays2;
};

struct InitialPairGateReport {
  bool has_stored_inliers = false;
  bool pose_success = false;
  bool accepted = false;
  int config = 0;
  size_t input_matches = 0;
  size_t estimate_inliers = 0;
  double tri_angle_deg = -1.0;
  double translation_z = 0.0;
  double abs_translation_z = 0.0;
};

struct Support {
  size_t num_inliers = 0;
  double residual_sum = std::numeric_limits<double>::max();
};

struct LocalUpdate {
  size_t local_trial = 0;
  size_t local_model_index = 0;
  size_t local_models_in_trial = 0;
  size_t inlier_sample_size = 0;
  size_t inliers = 0;
  double residual_sum = 0.0;
};

struct BestUpdate {
  size_t trial = 0;
  size_t model_index = 0;
  size_t models_in_sample = 0;
  std::vector<size_t> sample;
  size_t raw_inliers = 0;
  double raw_residual_sum = 0.0;
  size_t lo_inliers = 0;
  double lo_residual_sum = 0.0;
  bool lo_improved = false;
  std::vector<LocalUpdate> local_updates;
  size_t dynamic_max_trials = 0;
};

struct BoundaryResidual {
  size_t index = 0;
  double residual = 0.0;
  double squared_threshold = 0.0;
  double margin = 0.0;
  bool inlier = false;
};

struct TraceReportBase {
  std::string model_name;
  size_t sample_size = 0;
  size_t min_trials = 0;
  size_t max_trials = 0;
  size_t executed_trials = 0;
  size_t final_dynamic_max_trials = 0;
  std::string termination_reason = "max_trials";
  bool success = false;
  size_t final_inliers = 0;
  double final_residual_sum = std::numeric_limits<double>::max();
  std::vector<char> inlier_mask;
  std::vector<BestUpdate> best_updates;
  std::vector<BoundaryResidual> boundary_residuals;
};

template <typename Model>
struct TraceReport : public TraceReportBase {
  Model model = Model::Zero();
  bool best_model_is_local = false;
};

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

std::string BoolJson(const bool value) { return value ? "true" : "false"; }

double FiniteOrZero(const double value) {
  return std::isfinite(value) ? value : 0.0;
}

bool IsLeftBetter(const Support& left, const Support& right) {
  return left.num_inliers > right.num_inliers ||
         (left.num_inliers == right.num_inliers &&
          left.residual_sum < right.residual_sum);
}

Support EvaluateSupport(const std::vector<double>& residuals,
                        const double max_residual) {
  Support support;
  support.num_inliers = 0;
  support.residual_sum = 0.0;
  for (const double residual : residuals) {
    if (residual <= max_residual) {
      support.num_inliers += 1;
      support.residual_sum += residual;
    }
  }
  return support;
}

std::vector<BoundaryResidual> BoundaryResiduals(
    const std::vector<double>& residuals,
    const double max_residual) {
  std::vector<BoundaryResidual> values;
  values.reserve(residuals.size());
  for (size_t idx = 0; idx < residuals.size(); ++idx) {
    const double residual = residuals[idx];
    if (!std::isfinite(residual)) {
      continue;
    }
    BoundaryResidual boundary;
    boundary.index = idx;
    boundary.residual = residual;
    boundary.squared_threshold = max_residual;
    boundary.margin = residual - max_residual;
    boundary.inlier = residual <= max_residual;
    values.push_back(boundary);
  }
  std::sort(values.begin(), values.end(), [](const auto& left, const auto& right) {
    const double l = std::abs(left.margin);
    const double r = std::abs(right.margin);
    if (l == r) {
      return left.index < right.index;
    }
    return l < r;
  });
  if (values.size() > kMaxBoundaryResiduals) {
    values.resize(kMaxBoundaryResiduals);
  }
  return values;
}

std::vector<size_t> InlierIndices(const std::vector<double>& residuals,
                                  const double max_residual) {
  std::vector<size_t> indices;
  indices.reserve(residuals.size());
  for (size_t idx = 0; idx < residuals.size(); ++idx) {
    if (residuals[idx] <= max_residual) {
      indices.push_back(idx);
    }
  }
  return indices;
}

template <typename T>
std::vector<T> SelectByIndices(const std::vector<T>& values,
                               const std::vector<size_t>& indices) {
  std::vector<T> selected;
  selected.reserve(indices.size());
  for (const size_t idx : indices) {
    selected.push_back(values[idx]);
  }
  return selected;
}

template <typename Estimator>
std::vector<typename Estimator::X_t> SampleX(
    const std::vector<typename Estimator::X_t>& X,
    const std::vector<size_t>& sample) {
  std::vector<typename Estimator::X_t> out(Estimator::kMinNumSamples);
  for (size_t idx = 0; idx < sample.size(); ++idx) {
    out[idx] = X[sample[idx]];
  }
  return out;
}

template <typename Estimator>
std::vector<typename Estimator::Y_t> SampleY(
    const std::vector<typename Estimator::Y_t>& Y,
    const std::vector<size_t>& sample) {
  std::vector<typename Estimator::Y_t> out(Estimator::kMinNumSamples);
  for (size_t idx = 0; idx < sample.size(); ++idx) {
    out[idx] = Y[sample[idx]];
  }
  return out;
}

template <typename Estimator, typename LocalEstimator>
TraceReport<typename Estimator::M_t> TraceLORANSAC(
    const std::string& model_name,
    const colmap::RANSACOptions& input_options,
    const std::vector<typename Estimator::X_t>& X,
    const std::vector<typename Estimator::Y_t>& Y) {
  static_assert(std::is_same_v<typename Estimator::M_t,
                               typename LocalEstimator::M_t>);
  TraceReport<typename Estimator::M_t> report;
  report.model_name = model_name;
  report.sample_size = Estimator::kMinNumSamples;
  report.min_trials = static_cast<size_t>(input_options.min_num_trials);

  if (input_options.random_seed != -1) {
    colmap::SetPRNGSeed(static_cast<unsigned>(input_options.random_seed));
  }

  const size_t num_samples = X.size();
  if (num_samples < Estimator::kMinNumSamples) {
    report.termination_reason = "sampler_exhausted";
    return report;
  }

  colmap::RANSACOptions options = input_options;
  const size_t assumed_samples = 100000;
  const size_t assumed_inliers =
      static_cast<size_t>(options.min_inlier_ratio * assumed_samples);
  const size_t initial_dyn_max_trials =
      colmap::RANSAC<Estimator>::ComputeNumTrials(assumed_inliers,
                                                  assumed_samples,
                                                  options.confidence,
                                                  options.dyn_num_trials_multiplier);
  options.max_num_trials =
      static_cast<int>(std::min<size_t>(options.max_num_trials,
                                        initial_dyn_max_trials));

  colmap::RandomSampler sampler(Estimator::kMinNumSamples);
  sampler.Initialize(num_samples);
  const size_t max_num_trials =
      std::min<size_t>(options.max_num_trials, sampler.MaxNumSamples());
  size_t dyn_max_num_trials = max_num_trials;
  report.max_trials = max_num_trials;
  report.final_dynamic_max_trials = dyn_max_num_trials;

  const double max_residual = options.max_error * options.max_error;
  std::vector<double> residuals;
  std::vector<double> best_local_residuals;
  std::vector<size_t> sample;
  std::vector<typename Estimator::M_t> sample_models;
  std::vector<typename LocalEstimator::M_t> local_models;
  std::optional<typename Estimator::M_t> best_model;
  Support best_support;
  bool best_model_is_local = false;
  bool abort = false;

  for (size_t trial = 0; trial < max_num_trials; ++trial) {
    report.executed_trials = trial + 1;
    if (abort) {
      break;
    }

    sampler.Sample(&sample);
    const auto X_rand = SampleX<Estimator>(X, sample);
    const auto Y_rand = SampleY<Estimator>(Y, sample);
    Estimator::Estimate(X_rand, Y_rand, &sample_models);
    const size_t models_in_sample = sample_models.size();

    for (size_t model_idx = 0; model_idx < sample_models.size(); ++model_idx) {
      const auto& sample_model = sample_models[model_idx];
      Estimator::Residuals(X, Y, sample_model, &residuals);
      const Support raw_support = EvaluateSupport(residuals, max_residual);

      if (IsLeftBetter(raw_support, best_support)) {
        best_support = raw_support;
        best_model = sample_model;
        best_model_is_local = false;
        Support support_after_lo = best_support;
        std::vector<LocalUpdate> local_updates;

        if (best_support.num_inliers > Estimator::kMinNumSamples &&
            best_support.num_inliers >= LocalEstimator::kMinNumSamples) {
          constexpr size_t kMaxNumLocalTrials = 10;
          for (size_t local_trial = 0; local_trial < kMaxNumLocalTrials;
               ++local_trial) {
            const std::vector<size_t> inliers =
                InlierIndices(residuals, max_residual);
            const auto X_inlier =
                SelectByIndices<typename LocalEstimator::X_t>(X, inliers);
            const auto Y_inlier =
                SelectByIndices<typename LocalEstimator::Y_t>(Y, inliers);
            LocalEstimator::Estimate(X_inlier, Y_inlier, &local_models);
            const size_t prev_best_num_inliers = best_support.num_inliers;
            const size_t local_models_in_trial = local_models.size();

            for (size_t local_model_idx = 0;
                 local_model_idx < local_models.size();
                 ++local_model_idx) {
              const auto& local_model = local_models[local_model_idx];
              LocalEstimator::Residuals(X, Y, local_model, &residuals);
              const Support local_support =
                  EvaluateSupport(residuals, max_residual);
              if (IsLeftBetter(local_support, best_support)) {
                best_support = local_support;
                best_model = local_model;
                best_model_is_local = true;
                std::swap(residuals, best_local_residuals);
                LocalUpdate update;
                update.local_trial = local_trial;
                update.local_model_index = local_model_idx;
                update.local_models_in_trial = local_models_in_trial;
                update.inlier_sample_size = inliers.size();
                update.inliers = best_support.num_inliers;
                update.residual_sum = best_support.residual_sum;
                local_updates.push_back(update);
              }
            }

            if (best_support.num_inliers <= prev_best_num_inliers) {
              break;
            }

            std::swap(residuals, best_local_residuals);
          }
        }

        support_after_lo = best_support;
        dyn_max_num_trials =
            colmap::RANSAC<Estimator>::ComputeNumTrials(
                best_support.num_inliers,
                num_samples,
                options.confidence,
                options.dyn_num_trials_multiplier);
        report.final_dynamic_max_trials = dyn_max_num_trials;

        if (report.best_updates.size() < kMaxTraceBestUpdates) {
          BestUpdate update;
          update.trial = trial;
          update.model_index = model_idx;
          update.models_in_sample = models_in_sample;
          update.sample = sample;
          update.raw_inliers = raw_support.num_inliers;
          update.raw_residual_sum = raw_support.residual_sum;
          update.lo_inliers = support_after_lo.num_inliers;
          update.lo_residual_sum = support_after_lo.residual_sum;
          update.lo_improved = IsLeftBetter(support_after_lo, raw_support);
          update.local_updates = std::move(local_updates);
          update.dynamic_max_trials = dyn_max_num_trials;
          report.best_updates.push_back(std::move(update));
        }
      }

      if (trial >= dyn_max_num_trials &&
          trial >= static_cast<size_t>(options.min_num_trials)) {
        abort = true;
        report.termination_reason = "dynamic_abort";
        break;
      }
    }
  }

  if (!best_model.has_value()) {
    report.success = false;
    return report;
  }

  report.model = best_model.value();
  report.best_model_is_local = best_model_is_local;
  report.final_inliers = best_support.num_inliers;
  report.final_residual_sum = best_support.residual_sum;
  if (best_support.num_inliers < Estimator::kMinNumSamples) {
    report.success = false;
    return report;
  }

  report.success = true;
  if (best_model_is_local) {
    LocalEstimator::Residuals(X, Y, report.model, &residuals);
  } else {
    Estimator::Residuals(X, Y, report.model, &residuals);
  }
  report.inlier_mask.resize(residuals.size());
  for (size_t idx = 0; idx < residuals.size(); ++idx) {
    report.inlier_mask[idx] = residuals[idx] <= max_residual;
  }
  report.boundary_residuals = BoundaryResiduals(residuals, max_residual);
  return report;
}

class SqliteDb {
 public:
  explicit SqliteDb(const std::string& path) {
    if (sqlite3_open_v2(path.c_str(), &db_, SQLITE_OPEN_READWRITE, nullptr) !=
        SQLITE_OK) {
      throw std::runtime_error("failed to open database: " + path);
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
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt_, nullptr) != SQLITE_OK) {
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

void BindText(sqlite3_stmt* stmt, const int index, const std::string& value) {
  if (sqlite3_bind_text(stmt, index, value.c_str(), -1, SQLITE_TRANSIENT) !=
      SQLITE_OK) {
    throw std::runtime_error("sqlite bind text failed");
  }
}

void BindInt64(sqlite3_stmt* stmt, const int index, const int64_t value) {
  if (sqlite3_bind_int64(stmt, index, value) != SQLITE_OK) {
    throw std::runtime_error("sqlite bind int failed");
  }
}

ImageRow ReadImage(sqlite3* db, const std::string& name) {
  Statement stmt(db,
                 "SELECT image_id, name, camera_id FROM images WHERE name = ?1");
  BindText(stmt.get(), 1, name);
  if (sqlite3_step(stmt.get()) != SQLITE_ROW) {
    throw std::runtime_error("missing image: " + name);
  }
  ImageRow row;
  row.image_id = static_cast<uint32_t>(sqlite3_column_int64(stmt.get(), 0));
  row.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt.get(), 1));
  row.camera_id = static_cast<uint32_t>(sqlite3_column_int64(stmt.get(), 2));
  return row;
}

std::vector<ImageRow> ReadAllImages(sqlite3* db) {
  Statement stmt(db, "SELECT image_id, name, camera_id FROM images");
  std::vector<ImageRow> images;
  while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
    ImageRow row;
    row.image_id = static_cast<uint32_t>(sqlite3_column_int64(stmt.get(), 0));
    row.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt.get(), 1));
    row.camera_id = static_cast<uint32_t>(sqlite3_column_int64(stmt.get(), 2));
    images.push_back(row);
  }
  std::sort(images.begin(), images.end(), [](const auto& left, const auto& right) {
    return left.name < right.name;
  });
  return images;
}

CameraRow ReadCamera(sqlite3* db, const uint32_t camera_id) {
  Statement stmt(
      db,
      "SELECT camera_id, model, width, height, params, prior_focal_length "
      "FROM cameras WHERE camera_id = ?1");
  BindInt64(stmt.get(), 1, camera_id);
  if (sqlite3_step(stmt.get()) != SQLITE_ROW) {
    throw std::runtime_error("missing camera_id=" + std::to_string(camera_id));
  }

  CameraRow row;
  colmap::Camera& camera = row.camera;
  camera.camera_id = static_cast<colmap::camera_t>(
      sqlite3_column_int64(stmt.get(), 0));
  camera.model_id = static_cast<colmap::CameraModelId>(
      sqlite3_column_int(stmt.get(), 1));
  camera.width = static_cast<size_t>(sqlite3_column_int64(stmt.get(), 2));
  camera.height = static_cast<size_t>(sqlite3_column_int64(stmt.get(), 3));
  const void* data = sqlite3_column_blob(stmt.get(), 4);
  const int bytes = sqlite3_column_bytes(stmt.get(), 4);
  if (data == nullptr || bytes <= 0 || bytes % static_cast<int>(sizeof(double)) != 0) {
    throw std::runtime_error("invalid camera params blob");
  }
  const size_t count = static_cast<size_t>(bytes) / sizeof(double);
  camera.params.resize(count);
  std::memcpy(camera.params.data(), data, static_cast<size_t>(bytes));
  camera.has_prior_focal_length = sqlite3_column_int(stmt.get(), 5) != 0;
  if (!camera.VerifyParams()) {
    throw std::runtime_error("camera params failed COLMAP verification");
  }
  return row;
}

std::vector<Keypoint> ReadKeypoints(sqlite3* db, const uint32_t image_id) {
  Statement stmt(db,
                 "SELECT rows, cols, data FROM keypoints WHERE image_id = ?1");
  BindInt64(stmt.get(), 1, image_id);
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
  std::vector<Keypoint> keypoints;
  keypoints.reserve(static_cast<size_t>(rows));
  for (int row = 0; row < rows; ++row) {
    Keypoint kp;
    kp.x = values[row * cols + 0];
    kp.y = values[row * cols + 1];
    keypoints.push_back(kp);
  }
  return keypoints;
}

colmap::image_pair_t ImagePairToPairId(uint32_t image_id1,
                                       uint32_t image_id2) {
  return colmap::ImagePairToPairId(image_id1, image_id2);
}

std::vector<MatchRow> ReadMatches(sqlite3* db,
                                  const uint32_t image_id1,
                                  const uint32_t image_id2) {
  Statement stmt(db, "SELECT rows, cols, data FROM matches WHERE pair_id = ?1");
  BindInt64(stmt.get(),
            1,
            static_cast<int64_t>(ImagePairToPairId(image_id1, image_id2)));
  if (sqlite3_step(stmt.get()) != SQLITE_ROW) {
    throw std::runtime_error("missing matches for image pair");
  }
  const int rows = sqlite3_column_int(stmt.get(), 0);
  const int cols = sqlite3_column_int(stmt.get(), 1);
  const void* data = sqlite3_column_blob(stmt.get(), 2);
  const int bytes = sqlite3_column_bytes(stmt.get(), 2);
  if (rows < 0 || cols != 2 ||
      bytes != rows * cols * static_cast<int>(sizeof(uint32_t))) {
    throw std::runtime_error("invalid matches blob");
  }
  const auto* values = static_cast<const uint32_t*>(data);
  const bool swapped = image_id1 > image_id2;
  std::vector<MatchRow> matches;
  matches.reserve(static_cast<size_t>(rows));
  for (int row = 0; row < rows; ++row) {
    MatchRow match;
    if (swapped) {
      match.point2D_idx1 = values[row * 2 + 1];
      match.point2D_idx2 = values[row * 2 + 0];
    } else {
      match.point2D_idx1 = values[row * 2 + 0];
      match.point2D_idx2 = values[row * 2 + 1];
    }
    matches.push_back(match);
  }
  return matches;
}

std::vector<MatchRow> ReadTwoViewInlierMatches(sqlite3* db,
                                               const uint32_t image_id1,
                                               const uint32_t image_id2) {
  Statement stmt(
      db, "SELECT rows, cols, data FROM two_view_geometries WHERE pair_id = ?1");
  BindInt64(stmt.get(),
            1,
            static_cast<int64_t>(ImagePairToPairId(image_id1, image_id2)));
  if (sqlite3_step(stmt.get()) != SQLITE_ROW) {
    return {};
  }
  const int rows = sqlite3_column_int(stmt.get(), 0);
  const int cols = sqlite3_column_int(stmt.get(), 1);
  const void* data = sqlite3_column_blob(stmt.get(), 2);
  const int bytes = sqlite3_column_bytes(stmt.get(), 2);
  if (rows <= 0 || cols != 2 || data == nullptr ||
      bytes != rows * cols * static_cast<int>(sizeof(uint32_t))) {
    return {};
  }
  const auto* values = static_cast<const uint32_t*>(data);
  const bool swapped = image_id1 > image_id2;
  std::vector<MatchRow> matches;
  matches.reserve(static_cast<size_t>(rows));
  for (int row = 0; row < rows; ++row) {
    MatchRow match;
    if (swapped) {
      match.point2D_idx1 = values[row * 2 + 1];
      match.point2D_idx2 = values[row * 2 + 0];
    } else {
      match.point2D_idx1 = values[row * 2 + 0];
      match.point2D_idx2 = values[row * 2 + 1];
    }
    matches.push_back(match);
  }
  return matches;
}

std::vector<Eigen::Vector2d> KeypointsToPoints(
    const std::vector<Keypoint>& keypoints) {
  std::vector<Eigen::Vector2d> points;
  points.reserve(keypoints.size());
  for (const Keypoint& keypoint : keypoints) {
    points.emplace_back(keypoint.x, keypoint.y);
  }
  return points;
}

colmap::FeatureMatches ToColmapMatches(const std::vector<MatchRow>& matches) {
  colmap::FeatureMatches out;
  out.reserve(matches.size());
  for (const MatchRow& match : matches) {
    out.emplace_back(match.point2D_idx1, match.point2D_idx2);
  }
  return out;
}

InitialPairGateReport EstimateInitialPairGate(
    const colmap::Camera& camera1,
    const std::vector<Keypoint>& keypoints1,
    const colmap::Camera& camera2,
    const std::vector<Keypoint>& keypoints2,
    const std::vector<MatchRow>& stored_inliers,
    const Args& args) {
  InitialPairGateReport report;
  report.input_matches = stored_inliers.size();
  report.has_stored_inliers = !stored_inliers.empty();
  if (stored_inliers.empty()) {
    return report;
  }

  const std::vector<Eigen::Vector2d> points1 = KeypointsToPoints(keypoints1);
  const std::vector<Eigen::Vector2d> points2 = KeypointsToPoints(keypoints2);
  const colmap::FeatureMatches matches = ToColmapMatches(stored_inliers);

  colmap::TwoViewGeometryOptions options;
  options.ransac_options.min_num_trials = 30;
  options.ransac_options.max_error = args.max_error_px;
  options.ransac_options.max_num_trials = args.max_trials;
  options.ransac_options.random_seed = args.random_seed;
  options.Check();

  colmap::TwoViewGeometry geometry = colmap::EstimateCalibratedTwoViewGeometry(
      camera1, points1, camera2, points2, matches, options);
  report.config = geometry.config;
  report.estimate_inliers = geometry.inlier_matches.size();
  report.pose_success = colmap::EstimateTwoViewGeometryPose(
      camera1, points1, camera2, points2, &geometry);
  report.config = geometry.config;
  report.estimate_inliers = geometry.inlier_matches.size();
  if (!report.pose_success) {
    return report;
  }

  constexpr double kRadToDeg = 180.0 / 3.14159265358979323846264338327950288;
  report.tri_angle_deg = geometry.tri_angle * kRadToDeg;
  report.translation_z = geometry.cam2_from_cam1.translation.z();
  report.abs_translation_z = std::abs(report.translation_z);
  report.accepted =
      static_cast<int>(geometry.inlier_matches.size()) >=
          args.init_min_num_inliers &&
      report.abs_translation_z < args.init_max_forward_motion &&
      report.tri_angle_deg > args.init_min_tri_angle_deg;
  return report;
}

PreparedPair PreparePair(const colmap::Camera& camera1,
                         const std::vector<Keypoint>& keypoints1,
                         const colmap::Camera& camera2,
                         const std::vector<Keypoint>& keypoints2,
                         const std::vector<MatchRow>& matches) {
  PreparedPair pair;
  pair.img1.reserve(matches.size());
  pair.img2.reserve(matches.size());
  pair.rays1.reserve(matches.size());
  pair.rays2.reserve(matches.size());
  for (const MatchRow& match : matches) {
    if (match.point2D_idx1 >= keypoints1.size() ||
        match.point2D_idx2 >= keypoints2.size()) {
      continue;
    }
    const Keypoint& kp1 = keypoints1[match.point2D_idx1];
    const Keypoint& kp2 = keypoints2[match.point2D_idx2];
    const Eigen::Vector2d p1(kp1.x, kp1.y);
    const Eigen::Vector2d p2(kp2.x, kp2.y);
    const std::optional<Eigen::Vector2d> cam1 = camera1.CamFromImg(p1);
    const std::optional<Eigen::Vector2d> cam2 = camera2.CamFromImg(p2);
    if (!cam1 || !cam2) {
      continue;
    }
    pair.img1.push_back(p1);
    pair.img2.push_back(p2);
    pair.rays1.push_back(cam1->homogeneous().normalized());
    pair.rays2.push_back(cam2->homogeneous().normalized());
  }
  return pair;
}

std::optional<size_t> SortedImageIndex(const std::vector<ImageRow>& images,
                                       const uint32_t image_id) {
  for (size_t idx = 0; idx < images.size(); ++idx) {
    if (images[idx].image_id == image_id) {
      return idx;
    }
  }
  return std::nullopt;
}

uint64_t PairSamplerSeed(const size_t left_idx, const size_t right_idx) {
  return (static_cast<uint64_t>(left_idx) << 32) ^
         static_cast<uint64_t>(right_idx) ^ 0x243f6a8885a308d3ULL;
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
    } else if (key == "--image1") {
      args.image1 = next();
    } else if (key == "--image2") {
      args.image2 = next();
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
    } else if (key == "--init-min-num-inliers") {
      args.init_min_num_inliers = std::stoi(next());
    } else if (key == "--init-min-tri-angle-deg") {
      args.init_min_tri_angle_deg = std::stod(next());
    } else if (key == "--init-max-forward-motion") {
      args.init_max_forward_motion = std::stod(next());
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (args.database.empty() || args.image1.empty() || args.image2.empty()) {
    throw std::runtime_error(
        "usage: colmap_trace_probe --database DB --image1 NAME --image2 NAME "
        "[--random-seed N]");
  }
  return args;
}

template <typename Report>
void WriteBoundaryResiduals(std::ostream& out,
                            const Report& report,
                            const int indent) {
  const std::string pad(indent, ' ');
  out << "[";
  for (size_t idx = 0; idx < report.boundary_residuals.size(); ++idx) {
    const BoundaryResidual& r = report.boundary_residuals[idx];
    if (idx > 0) {
      out << ",";
    }
    out << "\n" << pad << "  {"
        << "\"index\":" << r.index << ","
        << "\"residual\":" << FiniteOrZero(r.residual) << ","
        << "\"squared_threshold\":" << FiniteOrZero(r.squared_threshold)
        << ","
        << "\"margin\":" << FiniteOrZero(r.margin) << ","
        << "\"inlier\":" << BoolJson(r.inlier) << "}";
  }
  if (!report.boundary_residuals.empty()) {
    out << "\n" << pad;
  }
  out << "]";
}

template <typename Report>
void WriteBestUpdates(std::ostream& out, const Report& report, const int indent) {
  const std::string pad(indent, ' ');
  out << "[";
  for (size_t idx = 0; idx < report.best_updates.size(); ++idx) {
    const BestUpdate& update = report.best_updates[idx];
    if (idx > 0) {
      out << ",";
    }
    out << "\n" << pad << "  {";
    out << "\"trial\":" << update.trial << ",";
    out << "\"model_index\":" << update.model_index << ",";
    out << "\"models_in_sample\":" << update.models_in_sample << ",";
    out << "\"sample\":[";
    for (size_t sample_idx = 0; sample_idx < update.sample.size(); ++sample_idx) {
      if (sample_idx > 0) {
        out << ",";
      }
      out << update.sample[sample_idx];
    }
    out << "],";
    out << "\"raw_inliers\":" << update.raw_inliers << ",";
    out << "\"raw_residual_sum\":" << FiniteOrZero(update.raw_residual_sum)
        << ",";
    out << "\"lo_inliers\":" << update.lo_inliers << ",";
    out << "\"lo_residual_sum\":" << FiniteOrZero(update.lo_residual_sum)
        << ",";
    out << "\"lo_improved\":" << BoolJson(update.lo_improved) << ",";
    out << "\"local_updates\":[";
    for (size_t local_idx = 0; local_idx < update.local_updates.size();
         ++local_idx) {
      const LocalUpdate& local = update.local_updates[local_idx];
      if (local_idx > 0) {
        out << ",";
      }
      out << "{"
          << "\"local_trial\":" << local.local_trial << ","
          << "\"local_model_index\":" << local.local_model_index << ","
          << "\"local_models_in_trial\":" << local.local_models_in_trial
          << ","
          << "\"inlier_sample_size\":" << local.inlier_sample_size << ","
          << "\"inliers\":" << local.inliers << ","
          << "\"residual_sum\":" << FiniteOrZero(local.residual_sum) << "}";
    }
    out << "],";
    out << "\"dynamic_max_trials\":" << update.dynamic_max_trials;
    out << "}";
  }
  if (!report.best_updates.empty()) {
    out << "\n" << pad;
  }
  out << "]";
}

template <typename Report>
void WriteTrace(std::ostream& out, const Report& report, const int indent) {
  const std::string pad(indent, ' ');
  out << "{\n";
  out << pad << "  \"sample_size\":" << report.sample_size << ",\n";
  out << pad << "  \"min_trials\":" << report.min_trials << ",\n";
  out << pad << "  \"max_trials\":" << report.max_trials << ",\n";
  out << pad << "  \"executed_trials\":" << report.executed_trials << ",\n";
  out << pad << "  \"final_dynamic_max_trials\":"
      << report.final_dynamic_max_trials << ",\n";
  out << pad << "  \"termination_reason\":\""
      << JsonEscape(report.termination_reason) << "\",\n";
  out << pad << "  \"success\":" << BoolJson(report.success) << ",\n";
  out << pad << "  \"best_model_is_local\":"
      << BoolJson(report.best_model_is_local) << ",\n";
  out << pad << "  \"final_inliers\":" << report.final_inliers << ",\n";
  out << pad << "  \"final_residual_sum\":"
      << FiniteOrZero(report.final_residual_sum) << ",\n";
  out << pad << "  \"best_updates\":";
  WriteBestUpdates(out, report, indent + 2);
  out << ",\n";
  out << pad << "  \"boundary_residuals\":";
  WriteBoundaryResiduals(out, report, indent + 2);
  out << "\n" << pad << "}";
}

std::string SourceName(const int source) {
  switch (source) {
    case 2:
      return "essential";
    case 3:
      return "fundamental";
    case 6:
      return "homography";
    default:
      return "none";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    SqliteDb db(args.database);
    const ImageRow left = ReadImage(db.get(), args.image1);
    const ImageRow right = ReadImage(db.get(), args.image2);
    const std::vector<ImageRow> sorted_images = ReadAllImages(db.get());
    const auto left_index = SortedImageIndex(sorted_images, left.image_id);
    const auto right_index = SortedImageIndex(sorted_images, right.image_id);
    if (!left_index || !right_index) {
      throw std::runtime_error("failed to resolve sorted image indices");
    }

    const CameraRow camera1 = ReadCamera(db.get(), left.camera_id);
    const CameraRow camera2 = ReadCamera(db.get(), right.camera_id);
    const std::vector<Keypoint> keypoints1 = ReadKeypoints(db.get(), left.image_id);
    const std::vector<Keypoint> keypoints2 = ReadKeypoints(db.get(), right.image_id);
    const std::vector<MatchRow> matches =
        ReadMatches(db.get(), left.image_id, right.image_id);
    const std::vector<MatchRow> stored_inliers =
        ReadTwoViewInlierMatches(db.get(), left.image_id, right.image_id);
    const PreparedPair prepared =
        PreparePair(camera1.camera, keypoints1, camera2.camera, keypoints2, matches);
    const InitialPairGateReport initial_pair_gate = EstimateInitialPairGate(
        camera1.camera,
        keypoints1,
        camera2.camera,
        keypoints2,
        stored_inliers,
        args);

    colmap::RANSACOptions ransac_options;
    ransac_options.max_error = args.max_error_px;
    ransac_options.confidence = 0.999;
    ransac_options.min_num_trials = args.min_trials;
    ransac_options.max_num_trials = args.max_trials;
    ransac_options.min_inlier_ratio = 0.25;
    ransac_options.random_seed = args.random_seed;
    ransac_options.Check();

    colmap::RANSACOptions essential_options = ransac_options;
    essential_options.max_error =
        (camera1.camera.CamFromImgThreshold(args.max_error_px) +
         camera2.camera.CamFromImgThreshold(args.max_error_px)) /
        2.0;

    const auto essential =
        TraceLORANSAC<colmap::EssentialMatrixFivePointEstimator,
                      colmap::EssentialMatrixFivePointEstimator>(
            "essential", essential_options, prepared.rays1, prepared.rays2);
    const auto fundamental =
        TraceLORANSAC<colmap::FundamentalMatrixSevenPointEstimator,
                      colmap::FundamentalMatrixEightPointEstimator>(
            "fundamental", ransac_options, prepared.img1, prepared.img2);
    const auto homography =
        TraceLORANSAC<colmap::HomographyMatrixEstimator,
                      colmap::HomographyMatrixEstimator>(
            "homography", ransac_options, prepared.img1, prepared.img2);

    const size_t e_inliers = essential.final_inliers;
    const size_t f_inliers = fundamental.final_inliers;
    const size_t h_inliers = homography.final_inliers;
    const double e_f_ratio =
        f_inliers == 0 ? 0.0 : static_cast<double>(e_inliers) / f_inliers;
    const double h_f_ratio =
        f_inliers == 0 ? 0.0 : static_cast<double>(h_inliers) / f_inliers;
    const double h_e_ratio =
        e_inliers == 0 ? 0.0 : static_cast<double>(h_inliers) / e_inliers;

    int config = 1;
    int source = 0;
    size_t selected_inliers = 0;
    if ((!essential.success && !fundamental.success && !homography.success) ||
        (e_inliers < static_cast<size_t>(args.min_inliers) &&
         f_inliers < static_cast<size_t>(args.min_inliers) &&
         h_inliers < static_cast<size_t>(args.min_inliers))) {
      config = 1;
    } else if (essential.success && e_f_ratio > 0.95 &&
               e_inliers >= static_cast<size_t>(args.min_inliers)) {
      config = h_e_ratio > 0.8 ? 6 : 2;
      if (e_inliers >= f_inliers) {
        selected_inliers = e_inliers;
        source = 2;
      } else {
        selected_inliers = f_inliers;
        source = 3;
      }
      if (h_e_ratio > 0.8 && h_inliers > selected_inliers) {
        selected_inliers = h_inliers;
        source = 6;
      }
    } else if (fundamental.success &&
               f_inliers >= static_cast<size_t>(args.min_inliers)) {
      config = h_f_ratio > 0.8 ? 6 : 3;
      selected_inliers = f_inliers;
      source = 3;
      if (h_f_ratio > 0.8 && h_inliers > selected_inliers) {
        selected_inliers = h_inliers;
        source = 6;
      }
    } else if (homography.success &&
               h_inliers >= static_cast<size_t>(args.min_inliers)) {
      config = 6;
      selected_inliers = h_inliers;
      source = 6;
    }

    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"database\":\"" << JsonEscape(args.database) << "\",\n";
    std::cout << "  \"left_image\":\"" << JsonEscape(left.name) << "\",\n";
    std::cout << "  \"right_image\":\"" << JsonEscape(right.name) << "\",\n";
    std::cout << "  \"left_index\":" << *left_index << ",\n";
    std::cout << "  \"right_index\":" << *right_index << ",\n";
    std::cout << "  \"sampler_seed\":"
              << PairSamplerSeed(*left_index, *right_index) << ",\n";
    std::cout << "  \"random_seed\":" << args.random_seed << ",\n";
    std::cout << "  \"num_matches\":" << matches.size() << ",\n";
    std::cout << "  \"stored_inlier_matches\":" << stored_inliers.size()
              << ",\n";
    std::cout << "  \"active_observations\":" << prepared.img1.size() << ",\n";
    std::cout << "  \"initial_pair_gate\":{"
              << "\"input_matches\":" << initial_pair_gate.input_matches << ","
              << "\"has_stored_inliers\":"
              << BoolJson(initial_pair_gate.has_stored_inliers) << ","
              << "\"pose_success\":" << BoolJson(initial_pair_gate.pose_success)
              << ","
              << "\"accepted\":" << BoolJson(initial_pair_gate.accepted) << ","
              << "\"config\":" << initial_pair_gate.config << ","
              << "\"estimate_inliers\":" << initial_pair_gate.estimate_inliers
              << ","
              << "\"translation_z\":"
              << FiniteOrZero(initial_pair_gate.translation_z) << ","
              << "\"abs_translation_z\":"
              << FiniteOrZero(initial_pair_gate.abs_translation_z) << ","
              << "\"tri_angle_deg\":"
              << FiniteOrZero(initial_pair_gate.tri_angle_deg) << ","
              << "\"init_min_num_inliers\":"
              << args.init_min_num_inliers << ","
              << "\"init_min_tri_angle_deg\":"
              << args.init_min_tri_angle_deg << ","
              << "\"init_max_forward_motion\":"
              << args.init_max_forward_motion << "},\n";
    std::cout << "  \"classification\":{"
              << "\"config\":" << config << ","
              << "\"selected_source\":\"" << SourceName(source) << "\","
              << "\"selected_inliers\":" << selected_inliers << ","
              << "\"e_f_inlier_ratio\":" << e_f_ratio << ","
              << "\"h_f_inlier_ratio\":" << h_f_ratio << ","
              << "\"h_e_inlier_ratio\":" << h_e_ratio << "},\n";
    std::cout << "  \"essential_trace\":";
    WriteTrace(std::cout, essential, 2);
    std::cout << ",\n";
    std::cout << "  \"fundamental_trace\":";
    WriteTrace(std::cout, fundamental, 2);
    std::cout << ",\n";
    std::cout << "  \"homography_trace\":";
    WriteTrace(std::cout, homography, 2);
    std::cout << "\n";
    std::cout << "}\n";
  } catch (const std::exception& e) {
    std::cerr << "colmap_trace_probe: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
