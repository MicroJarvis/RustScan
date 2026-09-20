/* Cold-process SIFT regression. No SIFT call is allowed before the start gate.
 * Include the real translation unit to optionally count constructor exp calls;
 * neither the table stores nor fast_expn reads are wrapped or synchronized. */
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
#ifdef COUNT_INIT
static atomic_uint init_calls;
static _Thread_local int in_constructor;
static double counted_exp(double x) {
  if (in_constructor) atomic_fetch_add_explicit(&init_calls, 1, memory_order_relaxed);
  return exp(x);
}
#define exp counted_exp
#endif
#include "sift.c"
#ifdef COUNT_INIT
#undef exp
#endif

#define WIDTH 97
#define HEIGHT 83
#define WORKERS 8
#define REPEATS 12
static pthread_mutex_t gate_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gate_cond = PTHREAD_COND_INITIALIZER;
static int ready, start;
static unsigned char *expected;
static size_t expected_size;
static vl_sift_pix image[WIDTH * HEIGHT];
static vl_sift_pix gradient[2 * WIDTH * HEIGHT];

static void put(FILE *file, const void *data, size_t size) {
  CHECK(fwrite(data, 1, size, file) == size);
}

static void extract(FILE *out) {
  VlSiftFilt *f;
  vl_sift_pix descriptor[128];
  int status, total = 0;
#ifdef COUNT_INIT
  in_constructor = 1;
#endif
  f = vl_sift_new(WIDTH, HEIGHT, -1, 3, 0);
#ifdef COUNT_INIT
  in_constructor = 0;
#endif
  CHECK(f != NULL);
  /* Cover the raw (covdet consumer) path as well as detected keypoints. */
  for (int a = 0; a < 4; ++a) {
    vl_sift_calc_raw_descriptor(f, gradient, descriptor, WIDTH, HEIGHT,
                               48.0, 41.0, 3.0, a * 0.37);
    put(out, descriptor, sizeof(descriptor));
  }
  status = vl_sift_process_first_octave(f, image);
  while (status == VL_ERR_OK) {
    vl_sift_detect(f);
    int count = vl_sift_get_nkeypoints(f);
    const VlSiftKeypoint *keys = vl_sift_get_keypoints(f);
    put(out, &count, sizeof(count));
    for (int i = 0; i < count; ++i) {
      const VlSiftKeypoint *k = keys + i;
      double angles[4];
      int n = vl_sift_calc_keypoint_orientations(f, angles, k);
      /* Explicit fields: never compare struct padding. */
      put(out, &k->o, sizeof(k->o));
      put(out, &k->ix, sizeof(k->ix));
      put(out, &k->iy, sizeof(k->iy));
      put(out, &k->is, sizeof(k->is));
      put(out, &k->x, sizeof(k->x));
      put(out, &k->y, sizeof(k->y));
      put(out, &k->s, sizeof(k->s));
      put(out, &k->sigma, sizeof(k->sigma));
      put(out, &n, sizeof(n));
      put(out, angles, n * sizeof(*angles));
      for (int a = 0; a < n; ++a) {
        vl_sift_calc_keypoint_descriptor(f, descriptor, k, angles[a]);
        put(out, descriptor, sizeof(descriptor));
        ++total;
      }
    }
    status = vl_sift_process_next_octave(f);
  }
  CHECK(status == VL_ERR_EOF);
  CHECK(total > 0);
  put(out, &total, sizeof(total));
  vl_sift_delete(f);
}

static unsigned char *read_bytes(FILE *file, size_t *size) {
  CHECK(fseek(file, 0, SEEK_END) == 0);
  long length = ftell(file);
  CHECK(length > 0);
  *size = (size_t)length;
  unsigned char *data = malloc(*size);
  CHECK(data != NULL);
  rewind(file);
  CHECK(fread(data, 1, *size, file) == *size);
  return data;
}

static void compare(void) {
  FILE *out = tmpfile();
  CHECK(out != NULL);
  extract(out);
  size_t size;
  unsigned char *data = read_bytes(out, &size);
  CHECK(size == expected_size);
  CHECK(memcmp(data, expected, size) == 0);
  free(data);
  CHECK(fclose(out) == 0);
}

static void *worker(void *unused) {
  (void)unused;
  CHECK(pthread_mutex_lock(&gate_mutex) == 0);
  ++ready;
  CHECK(pthread_cond_broadcast(&gate_cond) == 0);
  while (!start) CHECK(pthread_cond_wait(&gate_cond, &gate_mutex) == 0);
  CHECK(pthread_mutex_unlock(&gate_mutex) == 0);
  for (int i = 0; i < REPEATS; ++i) compare();
  return NULL;
}

int main(int argc, char **argv) {
  CHECK(argc == 3);
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      int i = y * WIDTH + x;
      image[i] = (vl_sift_pix)(((x / 7 + y / 9) % 2) * 150 +
                              (x * 13 + y * 17 + x * y * 3) % 101);
      gradient[2 * i] = 1.0f + (i % 19) * 0.125f;
      gradient[2 * i + 1] = (i % 31) * 0.2f;
    }
  }
  if (!strcmp(argv[1], "baseline")) {
    FILE *out = fopen(argv[2], "wbx");
    CHECK(out != NULL);
    extract(out);
    CHECK(fclose(out) == 0);
    puts("PASS serial baseline written (exclusive create)");
    return 0;
  }
  FILE *in = fopen(argv[2], "rb");
  CHECK(in != NULL);
  expected = read_bytes(in, &expected_size);
  CHECK(fclose(in) == 0);
  if (!strcmp(argv[1], "serial")) {
    for (int i = 0; i < REPEATS; ++i) compare();
  } else {
    CHECK(!strcmp(argv[1], "cold"));
    pthread_t threads[WORKERS];
    for (int i = 0; i < WORKERS; ++i) CHECK(pthread_create(threads + i, NULL, worker, NULL) == 0);
    CHECK(pthread_mutex_lock(&gate_mutex) == 0);
    while (ready != WORKERS) CHECK(pthread_cond_wait(&gate_cond, &gate_mutex) == 0);
    start = 1;
    CHECK(pthread_cond_broadcast(&gate_cond) == 0);
    CHECK(pthread_mutex_unlock(&gate_mutex) == 0);
    for (int i = 0; i < WORKERS; ++i) CHECK(pthread_join(threads[i], NULL) == 0);
  }
#ifdef COUNT_INIT
  unsigned calls = atomic_load_explicit(&init_calls, memory_order_relaxed);
  printf("constructor exp calls=%u (expected 257)\n", calls);
  CHECK(calls == 257);
#endif
  printf("PASS %s: bitwise output bytes=%zu, workers=%d, repeats=%d\n",
         argv[1], expected_size, !strcmp(argv[1], "cold") ? WORKERS : 1, REPEATS);
  free(expected);
  return 0;
}
