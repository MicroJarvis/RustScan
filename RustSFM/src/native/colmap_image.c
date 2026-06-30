#include "colmap_image.h"

#include <stdlib.h>
#include <string.h>

#include <FreeImage.h>

static void ensure_freeimage_initialized(void) {
    static int initialized = 0;
    if (!initialized) {
        FreeImage_Initialise(FALSE);
        initialized = 1;
    }
}

static void set_error(RustSfmColmapGrayImage* out, const char* message) {
    if (out == NULL) {
        return;
    }
    if (out->error_message != NULL) {
        free(out->error_message);
    }
    out->error_message = message != NULL ? strdup(message) : NULL;
}

static int is_grey_bitmap(FIBITMAP* bitmap) {
    return FreeImage_GetColorType(bitmap) == FIC_MINISBLACK ||
           FreeImage_GetColorType(bitmap) == FIC_MINISWHITE;
}

static int is_supported_bitmap(FIBITMAP* bitmap) {
    const FREE_IMAGE_COLOR_TYPE color_type = FreeImage_GetColorType(bitmap);
    return color_type == FIC_MINISBLACK || color_type == FIC_MINISWHITE ||
           color_type == FIC_RGB;
}

int rustsfm_colmap_load_grayscale_u8(
    const char* path,
    RustSfmColmapGrayImage* out) {
    if (out == NULL || path == NULL) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    ensure_freeimage_initialized();

    const FREE_IMAGE_FORMAT format = FreeImage_GetFileType(path, 0);
    if (format == FIF_UNKNOWN) {
        set_error(out, "unknown image format");
        return 0;
    }

    FIBITMAP* bitmap = FreeImage_Load(format, path, 0);
    if (bitmap == NULL) {
        set_error(out, "FreeImage_Load failed");
        return 0;
    }

    if (!is_grey_bitmap(bitmap)) {
        if (FreeImage_GetBPP(bitmap) != 24) {
            FIBITMAP* converted_24 = FreeImage_ConvertTo24Bits(bitmap);
            FreeImage_Unload(bitmap);
            if (converted_24 == NULL) {
                set_error(out, "FreeImage_ConvertTo24Bits failed");
                return 0;
            }
            bitmap = converted_24;
        }
        FIBITMAP* converted_grey = FreeImage_ConvertToGreyscale(bitmap);
        FreeImage_Unload(bitmap);
        if (converted_grey == NULL) {
            set_error(out, "FreeImage_ConvertToGreyscale failed");
            return 0;
        }
        bitmap = converted_grey;
    }

    if (!is_supported_bitmap(bitmap)) {
        FreeImage_Unload(bitmap);
        set_error(out, "unsupported bitmap color type");
        return 0;
    }

    const int width = (int)FreeImage_GetWidth(bitmap);
    const int height = (int)FreeImage_GetHeight(bitmap);
    if (width <= 0 || height <= 0) {
        FreeImage_Unload(bitmap);
        set_error(out, "invalid image dimensions");
        return 0;
    }

    const size_t num_pixels = (size_t)width * (size_t)height;
    uint8_t* data = (uint8_t*)malloc(num_pixels);
    if (data == NULL) {
        FreeImage_Unload(bitmap);
        set_error(out, "out of memory");
        return 0;
    }

    size_t i = 0;
    for (int y = 0; y < height; ++y) {
        const uint8_t* line =
            (const uint8_t*)FreeImage_GetScanLine(bitmap, height - 1 - y);
        for (int x = 0; x < width; ++x) {
            data[i++] = line[x];
        }
    }

    FreeImage_Unload(bitmap);
    out->data = data;
    out->width = width;
    out->height = height;
    return 1;
}

void rustsfm_colmap_free_gray_image(RustSfmColmapGrayImage* image) {
    if (image == NULL) {
        return;
    }
    free(image->data);
    free(image->error_message);
    image->data = NULL;
    image->error_message = NULL;
    image->width = 0;
    image->height = 0;
}
