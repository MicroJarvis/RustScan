#ifndef RUSTSFM_COLMAP_IMAGE_H
#define RUSTSFM_COLMAP_IMAGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RustSfmColmapGrayImage {
    uint8_t* data;
    int width;
    int height;
    char* error_message;
} RustSfmColmapGrayImage;

int rustsfm_colmap_load_grayscale_u8(
    const char* path,
    RustSfmColmapGrayImage* out);

void rustsfm_colmap_free_gray_image(RustSfmColmapGrayImage* image);

#ifdef __cplusplus
}
#endif

#endif
