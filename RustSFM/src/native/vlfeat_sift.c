#include "vlfeat_sift.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "covdet.h"
#include "sift.h"

enum { DESCRIPTOR_DIM = 128 };
enum { COVDET_MAX_OCTAVE_RESOLUTION = 1000 };

typedef struct LevelData {
    RustSfmVlfeatSiftKeypoint* keypoints;
    uint8_t* descriptors;
    size_t capacity;
    size_t count;
} LevelData;

#if defined(_MSC_VER)
#define RUSTSFM_THREAD_LOCAL __declspec(thread)
#else
#define RUSTSFM_THREAD_LOCAL _Thread_local
#endif

static RUSTSFM_THREAD_LOCAL int paired_allocation_fail_after = -1;

static void* paired_malloc(size_t size) {
    if (paired_allocation_fail_after == 0) {
        return NULL;
    }
    if (paired_allocation_fail_after > 0) {
        --paired_allocation_fail_after;
    }
    return malloc(size);
}

static void set_error(RustSfmVlfeatSiftFeatures* out, const char* message) {
    if (out == NULL) {
        return;
    }
    if (out->error_message != NULL) {
        free(out->error_message);
    }
    out->error_message = message != NULL ? strdup(message) : NULL;
}

static void l1_root_normalize(float* desc, int dim) {
    float l1 = 0.0f;
    for (int i = 0; i < dim; ++i) {
        l1 += fabsf(desc[i]);
    }
    if (l1 <= 1.0e-12f) {
        return;
    }
    for (int i = 0; i < dim; ++i) {
        desc[i] /= l1;
        desc[i] = sqrtf(desc[i] > 0.0f ? desc[i] : 0.0f);
    }
}

static void l2_normalize(float* desc, int dim) {
    float l2 = 0.0f;
    for (int i = 0; i < dim; ++i) {
        l2 += desc[i] * desc[i];
    }
    l2 = sqrtf(l2);
    if (l2 <= 1.0e-12f) {
        return;
    }
    for (int i = 0; i < dim; ++i) {
        desc[i] /= l2;
    }
}

static void quantize_descriptor(const float* desc_in, uint8_t* desc_out, int dim) {
    for (int i = 0; i < dim; ++i) {
        float scaled = roundf(512.0f * desc_in[i]);
        if (scaled < 0.0f) {
            scaled = 0.0f;
        }
        if (scaled > 255.0f) {
            scaled = 255.0f;
        }
        desc_out[i] = (uint8_t)scaled;
    }
}

static void transform_vlfeat_to_ubc(uint8_t* desc) {
    static const int q[8] = {0, 7, 6, 5, 4, 3, 2, 1};
    uint8_t tmp[DESCRIPTOR_DIM];
    memcpy(tmp, desc, DESCRIPTOR_DIM);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 8; ++k) {
                desc[8 * (j + 4 * i) + q[k]] = tmp[8 * (j + 4 * i) + k];
            }
        }
    }
}

static int allocate_features_output(
    RustSfmVlfeatSiftFeatures* out,
    size_t kept,
    const RustSfmVlfeatSiftKeypoint* keypoints,
    const uint8_t* descriptors) {
    if (kept == 0) {
        out->count = 0;
        return 1;
    }

    out->keypoints =
        (RustSfmVlfeatSiftKeypoint*)malloc(kept * sizeof(RustSfmVlfeatSiftKeypoint));
    out->descriptors = (float*)malloc(kept * DESCRIPTOR_DIM * sizeof(float));
    if (out->keypoints == NULL || out->descriptors == NULL) {
        free(out->keypoints);
        free(out->descriptors);
        out->keypoints = NULL;
        out->descriptors = NULL;
        out->count = 0;
        set_error(out, "out of memory");
        return 0;
    }

    out->count = kept;
    for (size_t i = 0; i < kept; ++i) {
        out->keypoints[i] = keypoints[i];
        const uint8_t* src = &descriptors[i * DESCRIPTOR_DIM];
        float* dst = &out->descriptors[i * DESCRIPTOR_DIM];
        for (int d = 0; d < DESCRIPTOR_DIM; ++d) {
            dst[d] = (float)src[d] / 512.0f;
        }
    }
    return 1;
}

static int level_reserve(LevelData* level, size_t extra) {
    if (extra > SIZE_MAX - level->count) {
        return 0;
    }
    const size_t required = level->count + extra;
    if (required <= level->capacity) {
        return 1;
    }
    size_t new_capacity = level->capacity == 0 ? extra : level->capacity;
    while (new_capacity < required) {
        if (new_capacity > SIZE_MAX / 2) {
            new_capacity = required;
            break;
        }
        new_capacity *= 2;
    }
    if (new_capacity > SIZE_MAX / sizeof(RustSfmVlfeatSiftKeypoint) ||
        new_capacity > SIZE_MAX / DESCRIPTOR_DIM) {
        return 0;
    }

    RustSfmVlfeatSiftKeypoint* new_keypoints = (RustSfmVlfeatSiftKeypoint*)paired_malloc(
        new_capacity * sizeof(RustSfmVlfeatSiftKeypoint));
    if (new_keypoints == NULL) {
        return 0;
    }
    uint8_t* new_descriptors =
        (uint8_t*)paired_malloc(new_capacity * DESCRIPTOR_DIM);
    if (new_descriptors == NULL) {
        free(new_keypoints);
        return 0;
    }

    if (level->count > 0) {
        memcpy(
            new_keypoints,
            level->keypoints,
            level->count * sizeof(RustSfmVlfeatSiftKeypoint));
        memcpy(
            new_descriptors,
            level->descriptors,
            level->count * DESCRIPTOR_DIM);
    }
    free(level->keypoints);
    free(level->descriptors);
    level->keypoints = new_keypoints;
    level->descriptors = new_descriptors;
    level->capacity = new_capacity;
    return 1;
}

static int levels_append(
    LevelData** levels,
    size_t** level_num_features,
    size_t num_levels) {
    if (num_levels == SIZE_MAX) {
        return 0;
    }
    const size_t new_count = num_levels + 1;
    if (new_count > SIZE_MAX / sizeof(LevelData) ||
        new_count > SIZE_MAX / sizeof(size_t)) {
        return 0;
    }

    LevelData* new_levels =
        (LevelData*)paired_malloc(new_count * sizeof(LevelData));
    if (new_levels == NULL) {
        return 0;
    }
    size_t* new_level_num_features =
        (size_t*)paired_malloc(new_count * sizeof(size_t));
    if (new_level_num_features == NULL) {
        free(new_levels);
        return 0;
    }

    if (num_levels > 0) {
        memcpy(new_levels, *levels, num_levels * sizeof(LevelData));
        memcpy(
            new_level_num_features,
            *level_num_features,
            num_levels * sizeof(size_t));
    }
    memset(&new_levels[num_levels], 0, sizeof(LevelData));
    new_level_num_features[num_levels] = 0;

    free(*levels);
    free(*level_num_features);
    *levels = new_levels;
    *level_num_features = new_level_num_features;
    return 1;
}

int rustsfm_vlfeat_test_paired_allocation_failure(
    int growth_path,
    int fail_allocation) {
    if ((growth_path != 0 && growth_path != 1) ||
        (fail_allocation != 0 && fail_allocation != 1)) {
        return 0;
    }

    if (growth_path == 0) {
        LevelData level;
        memset(&level, 0, sizeof(level));
        level.keypoints = (RustSfmVlfeatSiftKeypoint*)malloc(
            sizeof(RustSfmVlfeatSiftKeypoint));
        level.descriptors = (uint8_t*)malloc(DESCRIPTOR_DIM);
        if (level.keypoints == NULL || level.descriptors == NULL) {
            free(level.keypoints);
            free(level.descriptors);
            return 0;
        }
        level.capacity = 1;
        level.count = 1;
        level.keypoints[0].x = 123.0f;
        level.descriptors[0] = 77;
        RustSfmVlfeatSiftKeypoint* old_keypoints = level.keypoints;
        uint8_t* old_descriptors = level.descriptors;

        paired_allocation_fail_after = fail_allocation;
        const int grew = level_reserve(&level, 1);
        paired_allocation_fail_after = -1;
        const int preserved = !grew && level.keypoints == old_keypoints &&
                              level.descriptors == old_descriptors &&
                              level.capacity == 1 && level.count == 1 &&
                              level.keypoints[0].x == 123.0f &&
                              level.descriptors[0] == 77;
        free(level.keypoints);
        free(level.descriptors);
        return preserved;
    }

    LevelData* levels = (LevelData*)malloc(sizeof(LevelData));
    size_t* level_num_features = (size_t*)malloc(sizeof(size_t));
    if (levels == NULL || level_num_features == NULL) {
        free(levels);
        free(level_num_features);
        return 0;
    }
    memset(levels, 0, sizeof(LevelData));
    level_num_features[0] = 19;
    LevelData* old_levels = levels;
    size_t* old_level_num_features = level_num_features;

    paired_allocation_fail_after = fail_allocation;
    const int grew = levels_append(&levels, &level_num_features, 1);
    paired_allocation_fail_after = -1;
    const int preserved = !grew && levels == old_levels &&
                          level_num_features == old_level_num_features &&
                          level_num_features[0] == 19;
    free(levels);
    free(level_num_features);
    return preserved;
}

static void level_free(LevelData* level) {
    free(level->keypoints);
    free(level->descriptors);
    memset(level, 0, sizeof(*level));
}

static void levels_free(LevelData* levels, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        level_free(&levels[i]);
    }
    free(levels);
}

static int covdet_feature_compare(const void* a, const void* b) {
    const VlCovDetFeature* fa = (const VlCovDetFeature*)a;
    const VlCovDetFeature* fb = (const VlCovDetFeature*)b;
    if (fa->o != fb->o) {
        return fa->o > fb->o ? -1 : 1;
    }
    if (fa->s > fb->s) {
        return -1;
    }
    if (fa->s < fb->s) {
        return 1;
    }
    return 0;
}

static void fill_keypoint_from_frame(
    RustSfmVlfeatSiftKeypoint* kp,
    const VlFrameOrientedEllipse* frame,
    int octave,
    float response) {
    kp->x = frame->x + 0.5f;
    kp->y = frame->y + 0.5f;
    kp->a11 = frame->a11;
    kp->a12 = frame->a12;
    kp->a21 = frame->a21;
    kp->a22 = frame->a22;
    kp->octave = octave;
    kp->response = response;
    float det = frame->a11 * frame->a22 - frame->a12 * frame->a21;
    if (det < 0.0f) {
        det = -det;
    }
    kp->size = sqrtf(det);
    if (kp->size <= 1.0e-12f) {
        kp->size = 1.0f;
    }
    kp->angle = atan2f(frame->a21, frame->a11);
}

static int extract_sift_covdet(
    const uint8_t* gray_u8,
    int width,
    int height,
    const RustSfmVlfeatSiftOptions* options,
    RustSfmVlfeatSiftFeatures* out) {
    if (options->octave_resolution > COVDET_MAX_OCTAVE_RESOLUTION) {
        set_error(out, "octave_resolution too large for covdet");
        return 0;
    }

    VlCovDet* covdet = vl_covdet_new(VL_COVDET_METHOD_DOG);
    if (covdet == NULL) {
        set_error(out, "vl_covdet_new failed");
        return 0;
    }

    vl_covdet_set_first_octave(covdet, options->first_octave);
    vl_covdet_set_octave_resolution(covdet, (vl_size)options->octave_resolution);
    vl_covdet_set_peak_threshold(covdet, options->peak_threshold);
    vl_covdet_set_edge_threshold(covdet, options->edge_threshold);

    size_t image_size = (size_t)width * (size_t)height;
    float* data_float = (float*)malloc(image_size * sizeof(float));
    if (data_float == NULL) {
        vl_covdet_delete(covdet);
        set_error(out, "out of memory");
        return 0;
    }
    for (size_t i = 0; i < image_size; ++i) {
        data_float[i] = (float)gray_u8[i] / 255.0f;
    }

    if (vl_covdet_put_image(covdet, data_float, (vl_size)width, (vl_size)height)) {
        free(data_float);
        vl_covdet_delete(covdet);
        set_error(out, "vl_covdet_put_image failed");
        return 0;
    }
    free(data_float);

    vl_covdet_detect(covdet, (vl_size)options->max_num_features);
    if (options->estimate_affine_shape) {
        vl_covdet_extract_affine_shape(covdet);
    }
    if (!options->upright) {
        vl_covdet_extract_orientations(covdet);
    }

    int num_features = (int)vl_covdet_get_num_features(covdet);
    VlCovDetFeature* features = vl_covdet_get_features(covdet);
    if (num_features <= 0 || features == NULL) {
        vl_covdet_delete(covdet);
        out->count = 0;
        return 1;
    }

    qsort(features, (size_t)num_features, sizeof(VlCovDetFeature), covdet_feature_compare);

    RustSfmVlfeatSiftKeypoint* keypoints =
        (RustSfmVlfeatSiftKeypoint*)malloc((size_t)num_features *
                                           sizeof(RustSfmVlfeatSiftKeypoint));
    uint8_t* descriptors =
        (uint8_t*)malloc((size_t)num_features * DESCRIPTOR_DIM * sizeof(uint8_t));
    if (keypoints == NULL || descriptors == NULL) {
        free(keypoints);
        free(descriptors);
        vl_covdet_delete(covdet);
        set_error(out, "out of memory");
        return 0;
    }

    size_t kept = 0;
    int prev_octave_scale_idx = INT_MAX;
    for (int i = 0; i < num_features; ++i) {
        int octave_scale_idx =
            features[i].o * COVDET_MAX_OCTAVE_RESOLUTION + features[i].s;
        if (octave_scale_idx > prev_octave_scale_idx) {
            vl_covdet_delete(covdet);
            free(keypoints);
            free(descriptors);
            set_error(out, "covdet features are not sorted by octave/scale");
            return 0;
        }
        fill_keypoint_from_frame(
            &keypoints[kept],
            &features[i].frame,
            features[i].o,
            features[i].peakScore);
        if (octave_scale_idx != prev_octave_scale_idx &&
            kept >= (size_t)options->max_num_features) {
            break;
        }
        kept += 1;
        prev_octave_scale_idx = octave_scale_idx;
    }

    const size_t k_patch_resolution = 15;
    const size_t k_patch_side = 2 * k_patch_resolution + 1;
    const double k_patch_relative_extent = 7.5;
    const double k_patch_relative_smoothing = 1.0;
    const double k_patch_step = k_patch_relative_extent / (double)k_patch_resolution;
    const double k_sigma =
        k_patch_relative_extent / (3.0 * (4.0 + 1.0) / 2.0) / k_patch_step;

    float dsp_min_scale = 1.0f;
    float dsp_scale_step = 0.0f;
    int dsp_num_scales = 1;
    if (options->domain_size_pooling) {
        dsp_min_scale = options->dsp_min_scale;
        dsp_scale_step = (options->dsp_max_scale - options->dsp_min_scale) /
                         (float)options->dsp_num_scales;
        dsp_num_scales = options->dsp_num_scales;
        if (dsp_num_scales <= 0) {
            vl_covdet_delete(covdet);
            free(keypoints);
            free(descriptors);
            set_error(out, "dsp_num_scales must be > 0");
            return 0;
        }
    }

    float* patch = (float*)malloc(k_patch_side * k_patch_side * sizeof(float));
    float* patch_xy = (float*)malloc(2 * k_patch_side * k_patch_side * sizeof(float));
    float* scaled_descriptors =
        (float*)malloc((size_t)dsp_num_scales * DESCRIPTOR_DIM * sizeof(float));
    float desc[DESCRIPTOR_DIM];
    if (patch == NULL || patch_xy == NULL || scaled_descriptors == NULL) {
        free(patch);
        free(patch_xy);
        free(scaled_descriptors);
        vl_covdet_delete(covdet);
        free(keypoints);
        free(descriptors);
        set_error(out, "out of memory");
        return 0;
    }

    VlSiftFilt* sift = vl_sift_new(16, 16, 1, 3, 0);
    if (sift == NULL) {
        free(patch);
        free(patch_xy);
        free(scaled_descriptors);
        vl_covdet_delete(covdet);
        free(keypoints);
        free(descriptors);
        set_error(out, "vl_sift_new failed");
        return 0;
    }
    vl_sift_set_magnif(sift, 3.0);

    for (size_t i = 0; i < kept; ++i) {
        for (int s = 0; s < dsp_num_scales; ++s) {
            const float dsp_scale = dsp_min_scale + (float)s * dsp_scale_step;
            VlFrameOrientedEllipse scaled_frame = features[i].frame;
            scaled_frame.a11 *= dsp_scale;
            scaled_frame.a12 *= dsp_scale;
            scaled_frame.a21 *= dsp_scale;
            scaled_frame.a22 *= dsp_scale;

            if (!vl_covdet_extract_patch_for_frame(covdet,
                                                   patch,
                                                   k_patch_resolution,
                                                   k_patch_relative_extent,
                                                   k_patch_relative_smoothing,
                                                   scaled_frame)) {
                memset(&scaled_descriptors[s * DESCRIPTOR_DIM], 0,
                       DESCRIPTOR_DIM * sizeof(float));
                continue;
            }

            vl_imgradient_polar_f(patch_xy,
                                  patch_xy + 1,
                                  2,
                                  2 * k_patch_side,
                                  patch,
                                  k_patch_side,
                                  k_patch_side,
                                  k_patch_side);

            vl_sift_calc_raw_descriptor(sift,
                                        patch_xy,
                                        &scaled_descriptors[s * DESCRIPTOR_DIM],
                                        (int)k_patch_side,
                                        (int)k_patch_side,
                                        (double)k_patch_resolution,
                                        (double)k_patch_resolution,
                                        k_sigma,
                                        0.0);
        }

        if (options->domain_size_pooling) {
            for (int d = 0; d < DESCRIPTOR_DIM; ++d) {
                float sum = 0.0f;
                for (int s = 0; s < dsp_num_scales; ++s) {
                    sum += scaled_descriptors[s * DESCRIPTOR_DIM + d];
                }
                desc[d] = sum / (float)dsp_num_scales;
            }
        } else {
            memcpy(desc, scaled_descriptors, DESCRIPTOR_DIM * sizeof(float));
        }

        if (options->normalization_l1_root) {
            l1_root_normalize(desc, DESCRIPTOR_DIM);
        } else {
            l2_normalize(desc, DESCRIPTOR_DIM);
        }

        quantize_descriptor(desc, &descriptors[i * DESCRIPTOR_DIM], DESCRIPTOR_DIM);
        transform_vlfeat_to_ubc(&descriptors[i * DESCRIPTOR_DIM]);
    }

    free(patch);
    free(patch_xy);
    free(scaled_descriptors);
    vl_sift_delete(sift);
    vl_covdet_delete(covdet);

    int ok = allocate_features_output(out, kept, keypoints, descriptors);
    free(keypoints);
    free(descriptors);
    return ok;
}

static int extract_sift_standard(
    const uint8_t* gray_u8,
    int width,
    int height,
    const RustSfmVlfeatSiftOptions* options,
    RustSfmVlfeatSiftFeatures* out) {
    VlSiftFilt* sift = vl_sift_new(
        width,
        height,
        options->num_octaves,
        options->octave_resolution,
        options->first_octave);
    if (sift == NULL) {
        set_error(out, "vl_sift_new failed");
        return 0;
    }

    vl_sift_set_peak_thresh(sift, options->peak_threshold);
    vl_sift_set_edge_thresh(sift, options->edge_threshold);

    size_t image_size = (size_t)width * (size_t)height;
    float* data_float = (float*)malloc(image_size * sizeof(float));
    if (data_float == NULL) {
        vl_sift_delete(sift);
        set_error(out, "out of memory");
        return 0;
    }
    for (size_t i = 0; i < image_size; ++i) {
        data_float[i] = (float)gray_u8[i] / 255.0f;
    }

    LevelData* levels = NULL;
    size_t* level_num_features = NULL;
    size_t num_levels = 0;
    int first_octave = 1;

    while (1) {
        if (first_octave) {
            if (vl_sift_process_first_octave(sift, data_float)) {
                break;
            }
            first_octave = 0;
        } else if (vl_sift_process_next_octave(sift)) {
            break;
        }

        vl_sift_detect(sift);
        const VlSiftKeypoint* vl_keypoints = vl_sift_get_keypoints(sift);
        const int num_keypoints = vl_sift_get_nkeypoints(sift);
        if (num_keypoints <= 0) {
            continue;
        }

        size_t level_idx = 0;
        int prev_level = -1;
        float desc[DESCRIPTOR_DIM];

        for (int i = 0; i < num_keypoints; ++i) {
            if (vl_keypoints[i].is != prev_level) {
                if (i > 0) {
                    levels[num_levels - 1].count = level_idx;
                }

                if (!levels_append(&levels, &level_num_features, num_levels)) {
                    free(data_float);
                    levels_free(levels, num_levels);
                    free(level_num_features);
                    vl_sift_delete(sift);
                    set_error(out, "out of memory");
                    return 0;
                }

                LevelData* level = &levels[num_levels];
                num_levels += 1;

                size_t reserve_count =
                    (size_t)options->max_num_orientations * (size_t)num_keypoints;
                if (!level_reserve(level, reserve_count)) {
                    free(data_float);
                    levels_free(levels, num_levels);
                    free(level_num_features);
                    vl_sift_delete(sift);
                    set_error(out, "out of memory");
                    return 0;
                }
                level_idx = 0;
            }

            level_num_features[num_levels - 1] += 1;
            prev_level = vl_keypoints[i].is;

            double angles[4];
            int num_orientations;
            if (options->upright) {
                num_orientations = 1;
                angles[0] = 0.0;
            } else {
                num_orientations =
                    vl_sift_calc_keypoint_orientations(sift, angles, &vl_keypoints[i]);
            }

            int num_used_orientations = num_orientations;
            if (num_used_orientations > options->max_num_orientations) {
                num_used_orientations = options->max_num_orientations;
            }

            for (int o = 0; o < num_used_orientations; ++o) {
                if (!level_reserve(levels + num_levels - 1, 1)) {
                    free(data_float);
                    levels_free(levels, num_levels);
                    free(level_num_features);
                    vl_sift_delete(sift);
                    set_error(out, "out of memory");
                    return 0;
                }

                LevelData* level = &levels[num_levels - 1];
                RustSfmVlfeatSiftKeypoint* kp = &level->keypoints[level_idx];
                kp->x = vl_keypoints[i].x + 0.5f;
                kp->y = vl_keypoints[i].y + 0.5f;
                kp->size = vl_keypoints[i].sigma;
                kp->angle = (float)angles[o];
                kp->response = vl_keypoints[i].sigma;
                kp->octave = vl_keypoints[i].o;
                kp->a11 = vl_keypoints[i].sigma;
                kp->a12 = (float)angles[o];
                kp->a21 = 0.0f;
                kp->a22 = 0.0f;

                vl_sift_calc_keypoint_descriptor(
                    sift, desc, &vl_keypoints[i], angles[o]);
                if (options->normalization_l1_root) {
                    l1_root_normalize(desc, DESCRIPTOR_DIM);
                } else {
                    l2_normalize(desc, DESCRIPTOR_DIM);
                }

                uint8_t* row = &level->descriptors[level_idx * DESCRIPTOR_DIM];
                quantize_descriptor(desc, row, DESCRIPTOR_DIM);
                transform_vlfeat_to_ubc(row);
                level_idx += 1;
            }
        }

        if (num_levels > 0) {
            levels[num_levels - 1].count = level_idx;
        }
    }

    free(data_float);
    vl_sift_delete(sift);

    size_t first_level_to_keep = 0;
    size_t num_features = 0;
    size_t num_features_with_orientations = 0;
    for (size_t i = num_levels; i-- > 0;) {
        num_features += level_num_features[i];
        num_features_with_orientations += levels[i].count;
        if (num_features > (size_t)options->max_num_features) {
            first_level_to_keep = i;
            break;
        }
    }

    if (num_features_with_orientations == 0) {
        levels_free(levels, num_levels);
        free(level_num_features);
        out->count = 0;
        return 1;
    }

    RustSfmVlfeatSiftKeypoint* keypoints = (RustSfmVlfeatSiftKeypoint*)malloc(
        num_features_with_orientations * sizeof(RustSfmVlfeatSiftKeypoint));
    uint8_t* descriptors =
        (uint8_t*)malloc(num_features_with_orientations * DESCRIPTOR_DIM * sizeof(uint8_t));
    if (keypoints == NULL || descriptors == NULL) {
        free(keypoints);
        free(descriptors);
        levels_free(levels, num_levels);
        free(level_num_features);
        set_error(out, "out of memory");
        return 0;
    }

    size_t k = 0;
    for (size_t i = first_level_to_keep; i < num_levels; ++i) {
        for (size_t j = 0; j < levels[i].count; ++j) {
            keypoints[k] = levels[i].keypoints[j];
            memcpy(&descriptors[k * DESCRIPTOR_DIM],
                   &levels[i].descriptors[j * DESCRIPTOR_DIM],
                   DESCRIPTOR_DIM);
            k += 1;
        }
    }

    levels_free(levels, num_levels);
    free(level_num_features);
    int ok = allocate_features_output(out, k, keypoints, descriptors);
    free(keypoints);
    free(descriptors);
    return ok;
}

int rustsfm_vlfeat_extract_sift(
    const uint8_t* gray_u8,
    int width,
    int height,
    const RustSfmVlfeatSiftOptions* options,
    RustSfmVlfeatSiftFeatures* out) {
    if (out == NULL || gray_u8 == NULL || options == NULL) {
        return 0;
    }
    memset(out, 0, sizeof(*out));

    if (width <= 0 || height <= 0) {
        set_error(out, "invalid image dimensions");
        return 0;
    }
    if (options->max_num_features <= 0 || options->octave_resolution <= 0) {
        set_error(out, "invalid SIFT extraction options");
        return 0;
    }

    if (options->force_covariant_extractor || options->estimate_affine_shape ||
         options->domain_size_pooling) {
        return extract_sift_covdet(gray_u8, width, height, options, out);
    }

    if (options->max_num_orientations <= 0) {
        set_error(out, "invalid SIFT extraction options");
        return 0;
    }

    return extract_sift_standard(gray_u8, width, height, options, out);
}

void rustsfm_vlfeat_free_features(RustSfmVlfeatSiftFeatures* features) {
    if (features == NULL) {
        return;
    }
    free(features->keypoints);
    free(features->descriptors);
    free(features->error_message);
    features->keypoints = NULL;
    features->descriptors = NULL;
    features->count = 0;
    features->error_message = NULL;
}
