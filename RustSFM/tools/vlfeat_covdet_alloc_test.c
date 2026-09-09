/* Instrument only bridge-owned allocations, without changing its production API. */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "covdet.h"
#include "sift.h"
static int live, calls, fail_at = -1, fail_sift;
static void *tracked_malloc(size_t size) {
    if (calls++ == fail_at) return NULL;
    void *p = malloc(size);
    if (p) ++live;
    return p;
}
static void tracked_free(void *p) { if (p) --live; free(p); }
static char *tracked_strdup(const char *s) {
    char *p = tracked_malloc(strlen(s) + 1);
    if (p) strcpy(p, s);
    return p;
}
static VlSiftFilt *tracked_sift_new(int w, int h, int o, int s, int first) {
    return fail_sift ? NULL : vl_sift_new(w, h, o, s, first);
}
#define malloc tracked_malloc
#define free tracked_free
#define strdup tracked_strdup
#define vl_sift_new tracked_sift_new
#include "../src/native/vlfeat_sift.c"
#undef malloc
#undef free
#undef strdup
#undef vl_sift_new

static int run(int failure, int sift_failure, int profile) {
    unsigned char gray[193 * 157];
    for (int y = 0; y < 157; ++y) for (int x = 0; x < 193; ++x)
        gray[y*193+x] = ((x/13+y/17)%2 == 0 ? 180 : 20) + (x*7+y*11+x*y%31)%55;
    RustSfmVlfeatSiftOptions options = {0};
    options.max_num_features = 8192;
    options.first_octave = -1;
    options.num_octaves = -1;
    options.octave_resolution = 3;
    options.peak_threshold = 0.0066666667f;
    options.edge_threshold = 10;
    options.max_num_orientations = 2;
    options.normalization_l1_root = 1;
    options.domain_size_pooling = 1;
    options.dsp_min_scale = 1.0f/6.0f;
    options.dsp_max_scale = 3;
    options.dsp_num_scales = 10;
    RustSfmVlfeatSiftFeatures out;
    RustSfmVlfeatSiftTiming timing;
    calls = 0; fail_at = failure; fail_sift = sift_failure;
    int ok = rustsfm_vlfeat_extract_sift(gray, 193, 157, &options, &out, profile ? &timing : NULL);
    int allocations = calls;
    if (failure < 0 && !sift_failure && (!ok || !out.count)) abort();
    if ((failure >= 0 || sift_failure) && ok) abort();
    rustsfm_vlfeat_free_features(&out);
    if (live != 0) {
        fprintf(stderr, "FAIL: allocation=%d sift_failure=%d profile=%d live=%d\n", failure, sift_failure, profile, live);
        exit(1);
    }
    return allocations;
}
int main(void) {
    for (int profile = 0; profile <= 1; ++profile) {
        int allocations = run(-1, 0, profile);
        for (int i = 0; i < allocations; ++i) run(i, 0, profile);
        run(-1, 1, profile);
        printf("PASS: covdet profile=%d success, %d bridge allocation failures, vl_sift_new failure; zero live allocations\n", profile, allocations);
    }
    return 0;
}
