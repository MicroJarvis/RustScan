#ifndef RUSTSFM_VLFEAT_SIFT_H
#define RUSTSFM_VLFEAT_SIFT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RustSfmVlfeatSiftOptions {
    int max_num_features;
    int first_octave;
    int num_octaves;
    int octave_resolution;
    float peak_threshold;
    float edge_threshold;
    int max_num_orientations;
    int upright;
    int normalization_l1_root;
    int estimate_affine_shape;
    int domain_size_pooling;
    float dsp_min_scale;
    float dsp_max_scale;
    int dsp_num_scales;
    int force_covariant_extractor;
} RustSfmVlfeatSiftOptions;

typedef struct RustSfmVlfeatSiftKeypoint {
    float x;
    float y;
    float size;
    float angle;
    float response;
    int32_t octave;
    float a11;
    float a12;
    float a21;
    float a22;
} RustSfmVlfeatSiftKeypoint;

typedef struct RustSfmVlfeatSiftFeatures {
    RustSfmVlfeatSiftKeypoint* keypoints;
    float* descriptors;
    size_t count;
    char* error_message;
} RustSfmVlfeatSiftFeatures;

int rustsfm_vlfeat_extract_sift(
    const uint8_t* gray_u8,
    int width,
    int height,
    const RustSfmVlfeatSiftOptions* options,
    RustSfmVlfeatSiftFeatures* out);

void rustsfm_vlfeat_free_features(RustSfmVlfeatSiftFeatures* features);

#ifdef __cplusplus
}
#endif

#endif
