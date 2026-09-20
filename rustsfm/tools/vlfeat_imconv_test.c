/* Bitwise gate against the frozen, separately compiled scalar oracle. */
#include "imopv.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void scalar_imconvcol_vf(float*, vl_size, const float*, vl_size, vl_size,
                        vl_size, const float*, vl_index, vl_index, int, unsigned);
static uint32_t state = 1;
static float sample(void) {
    state = state * 1664525u + 1013904223u;
    return (float)((int32_t)(state >> 8) - 8388608) / 8388608.0f;
}
static uint32_t bits(float v) { uint32_t u; memcpy(&u, &v, 4); return u; }
static int run_cases(int simd) {
    vl_set_simd_enabled(simd);
    state = 1;
    const int widths[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 193};
    const int heights[] = {1, 2, 7, 16, 33, 157};
    /* Include shifted supports and kernels wider than the image: the public
     * convolution contract does not require a centered Gaussian kernel. */
    /* Negative-only supports cover no interior tap, the final valid row,
     * and the transition into bottom padding at each tested image height. */
    const int supports[][2] = {{0,0}, {-1,1}, {-4,4}, {-12,12}, {-2,5}, {-7,0}, {0,7}, {-7,-2},
                               {-1,-1}, {-2,-1}, {-7,-7}, {-8,-6}, {-16,-16}, {-17,-15},
                               {-33,-33}, {-34,-32}, {-157,-157}, {-158,-156}, {-159,-158}};
    size_t cases = 0;
    for (size_t wi = 0; wi < sizeof(widths)/sizeof(*widths); ++wi)
    for (size_t hi = 0; hi < sizeof(heights)/sizeof(*heights); ++hi)
    for (size_t fi = 0; fi < sizeof(supports)/sizeof(*supports); ++fi)
    for (int step = 1; step <= 3; ++step)
    for (int transpose = 0; transpose <= 1; ++transpose)
    for (int pad = 0; pad <= 1; ++pad)
    for (int stride_pad = 0; stride_pad <= 3; stride_pad += 3) {
        int w = widths[wi], h = heights[hi], ss = w + stride_pad;
        int dh = (h - 1)/step + 1, ds = (transpose ? dh : w) + stride_pad;
        size_t n = (size_t)ds * (transpose ? w : dh) + 8;
        float *storage = malloc(((size_t)ss*h + 1)*sizeof(float));
        float *src = storage + 1; /* Deliberately unaligned SIMD loads. */
        float *a = malloc(n*sizeof(float)), *b = malloc(n*sizeof(float));
        float filter[25];
        if (!storage || !a || !b) abort();
        for (int i = 0; i < ss*h; ++i) src[i] = sample();
        for (int i = 0; i <= supports[fi][1]-supports[fi][0]; ++i) filter[i] = sample();
        for (size_t i = 0; i < n; ++i) a[i] = b[i] = -12345.0f;
        unsigned flags = (transpose ? VL_TRANSPOSE : 0) |
                         (pad ? VL_PAD_BY_CONTINUITY : VL_PAD_BY_ZERO);
        scalar_imconvcol_vf(a, ds, src, w, h, ss, filter, supports[fi][0], supports[fi][1], step, flags);
        vl_imconvcol_vf(b, ds, src, w, h, ss, filter, supports[fi][0], supports[fi][1], step, flags);
        for (size_t i = 0; i < n; ++i) if (bits(a[i]) != bits(b[i])) {
            fprintf(stderr, "FAIL w=%d h=%d support=[%d,%d] step=%d transpose=%d pad=%d stride_pad=%d index=%zu scalar=%08x actual=%08x\n",
                    w,h,supports[fi][0],supports[fi][1],step,transpose,pad,stride_pad,i,bits(a[i]),bits(b[i]));
            return 1;
        }
        free(storage); free(a); free(b); ++cases;
    }
    printf("PASS: %zu convolution cases, simd=%d (including destination padding guards)\n", cases, simd);
    return 0;
}
int main(void) {
    return run_cases(0) || run_cases(1);
}
