// CPU routed experts via llama.cpp's own optimized k-quant kernels (libggml-cpu):
// activations quantized to Q8_K (as llama.cpp does on CPU), dot with
// ggml_vec_dot_q{4,5,6}_K_q8_K. Same interface as cpu_moe.c:moe_cpu.
// Build: gcc -O3 -march=native -fopenmp -shared -fPIC cpu_moe_ggml.c -o libcpumoe_ggml.so \
//        -L$B -lggml-cpu -lggml-base -Wl,-rpath,$B
#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <sched.h>
#include <pthread.h>
#include <omp.h>
#include <unistd.h>
#include <sys/syscall.h>

void ggml_cpu_init(void);
void quantize_row_q8_K(const float *x, void *y, int64_t k);
void ggml_vec_dot_q4_K_q8_K(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);

#define Q8K_BYTES 292   /* float d + int8 qs[256] + int16 bsums[16] */

typedef void (*vdot_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

#if defined(__AVX512BW__)
// AVX-512BW Q4_K / Q5_K x Q8_K dot products (ggml's x86 k-quant paths are AVX2 only). Same
// integer math as ggml's (exact int32 block sums); only the float accumulation order differs.
// Per 128 weights: one 64-byte quant load, nibble split, a 128-bit-lane shuffle so each zmm
// holds two consecutive 32-weight sub-blocks matching one contiguous 64-byte Q8 load, then
// maddubs and a madd with the sub-block scales (vpermw from the 8 unpacked scales).
#include <immintrin.h>
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; } bq4k;
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } bq5k;
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } bq8k;

static inline void kq_scales(const uint8_t *s12, uint32_t utmp[4]) {
    const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    __builtin_memcpy(utmp, s12, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

// In-dot software prefetch `g_pfb` bytes ahead (0 = off). Rows of an expert matrix are
// contiguous, so this runs ahead across row ends, which the L2 streamer cannot do at 4 KB page
// boundaries; prefetching whole rows ahead from the row loops instead was 3x slower.
static int g_pfb = 0;
void cpumoe_set_pf_bytes(int b) { g_pfb = b; }
#define KQ_PF(p, nbytes) do { if (g_pfb) { const char *q_ = (const char *)(p) + g_pfb;   \
        for (int o_ = 0; o_ < (nbytes); o_ += 64) _mm_prefetch(q_ + o_, _MM_HINT_T0); } } while (0)

static inline __m512i sc_idx(int a, int b) {     // int16 lanes 0-15 -> a, 16-31 -> b
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(a)), _mm256_set1_epi16(b), 1);
}

// one superblock's int32 sum (scales applied) from the 4 nibble-split zmm halves
#define KQ_HALF(lo, hi, q8, k, sc, sumi) do {                                         \
        __m512i A = _mm512_shuffle_i64x2(lo, hi, 0x44), B = _mm512_shuffle_i64x2(lo, hi, 0xEE); \
        __m512i pa = _mm512_maddubs_epi16(A, _mm512_loadu_si512((const void *)((q8) + 128 * (k)))); \
        __m512i pb = _mm512_maddubs_epi16(B, _mm512_loadu_si512((const void *)((q8) + 128 * (k) + 64))); \
        pa = _mm512_madd_epi16(_mm512_permutexvar_epi16(I[2 * (k)], sc), pa);           \
        pb = _mm512_madd_epi16(_mm512_permutexvar_epi16(I[2 * (k) + 1], sc), pb);       \
        sumi = _mm512_add_epi32(sumi, _mm512_add_epi32(pa, pb));                        \
    } while (0)

static inline __m128 kq_mins(const bq8k *y, const uint32_t utmp[4], float dmin, __m128 acc_m) {
    const __m256i q8sums = _mm256_loadu_si256((const __m256i *)y->bsums);
    const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
    const __m128i mins = _mm_cvtepu8_epi16(_mm_set_epi32(0, 0, utmp[3], utmp[2]));
    return _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(_mm_madd_epi16(mins, q8s)), acc_m);
}

static void dot_q4K_512(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
    const bq4k *x = vx; const bq8k *y = vy;
    const int nb = n / 256;
    const __m512i m4 = _mm512_set1_epi8(0xF);
    const __m512i I[4] = {sc_idx(0, 1), sc_idx(2, 3), sc_idx(4, 5), sc_idx(6, 7)};
    __m512 acc = _mm512_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();
    uint32_t utmp[4];
    for (int i = 0; i < nb; ++i) {
        KQ_PF(&x[i], 144);
        const float yd = y[i].d;
        const float d = yd * _cvtsh_ss(x[i].d), dmin = -yd * _cvtsh_ss(x[i].dmin);
        kq_scales(x[i].scales, utmp);
        acc_m = kq_mins(&y[i], utmp, dmin, acc_m);
        const __m512i sc = _mm512_castsi128_si512(_mm_cvtepu8_epi16(_mm_set_epi32(0, 0, utmp[1], utmp[0])));
        const int8_t *q8 = y[i].qs;
        __m512i sumi = _mm512_setzero_si512();
        for (int k = 0; k < 2; ++k) {
            const __m512i q = _mm512_loadu_si512((const void *)(x[i].qs + 64 * k));
            const __m512i lo = _mm512_and_si512(q, m4), hi = _mm512_and_si512(_mm512_srli_epi16(q, 4), m4);
            KQ_HALF(lo, hi, q8, k, sc, sumi);
        }
        acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
    }
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    *s = _mm512_reduce_add_ps(acc) + _mm_cvtss_f32(acc_m);
}

static void dot_q5K_512(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
    const bq5k *x = vx; const bq8k *y = vy;
    const int nb = n / 256;
    const __m512i m4 = _mm512_set1_epi8(0xF), v16 = _mm512_set1_epi8(16);
    const __m512i I[4] = {sc_idx(0, 1), sc_idx(2, 3), sc_idx(4, 5), sc_idx(6, 7)};
    // 64-byte chunk k covers ggml chunks j = 2k (bytes 0-31) and 2k+1 (32-63); the low nibbles
    // of chunk j take high bit 2j of qh, the high nibbles bit 2j+1
    __m512i HL[2], HH[2];
    for (int k = 0; k < 2; ++k) {
        HL[k] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi8(1 << (4 * k))), _mm256_set1_epi8(1 << (4 * k + 2)), 1);
        HH[k] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi8(1 << (4 * k + 1))), _mm256_set1_epi8((char)(1 << (4 * k + 3))), 1);
    }
    __m512 acc = _mm512_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();
    uint32_t utmp[4];
    for (int i = 0; i < nb; ++i) {
        KQ_PF(&x[i], 176);
        const float yd = y[i].d;
        const float d = yd * _cvtsh_ss(x[i].d), dmin = -yd * _cvtsh_ss(x[i].dmin);
        kq_scales(x[i].scales, utmp);
        acc_m = kq_mins(&y[i], utmp, dmin, acc_m);
        const __m512i sc = _mm512_castsi128_si512(_mm_cvtepu8_epi16(_mm_set_epi32(0, 0, utmp[1], utmp[0])));
        const __m512i hb = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)x[i].qh));
        const int8_t *q8 = y[i].qs;
        __m512i sumi = _mm512_setzero_si512();
        for (int k = 0; k < 2; ++k) {
            const __m512i q = _mm512_loadu_si512((const void *)(x[i].qs + 64 * k));
            __m512i lo = _mm512_and_si512(q, m4), hi = _mm512_and_si512(_mm512_srli_epi16(q, 4), m4);
            lo = _mm512_mask_add_epi8(lo, _mm512_test_epi8_mask(hb, HL[k]), lo, v16);
            hi = _mm512_mask_add_epi8(hi, _mm512_test_epi8_mask(hb, HH[k]), hi, v16);
            KQ_HALF(lo, hi, q8, k, sc, sumi);
        }
        acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
    }
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    *s = _mm512_reduce_add_ps(acc) + _mm_cvtss_f32(acc_m);
}
#endif

static int g_avx512 = 0;
void cpumoe_set_avx512(int on) { g_avx512 = on; }

static vdot_t pick(int qt) {
#if defined(__AVX512BW__)
    if (g_avx512 && qt == 12) return dot_q4K_512;
    if (g_avx512 && qt == 13) return dot_q5K_512;
#endif
    return qt == 12 ? ggml_vec_dot_q4_K_q8_K : qt == 13 ? ggml_vec_dot_q5_K_q8_K : ggml_vec_dot_q6_K_q8_K;
}

// test hook: one row dot with the selected kernel (activations quantized here)
float cpumoe_dot_test(int qt, int n, const uint8_t *row, const float *x, int avx512) {
    int keep = g_avx512; g_avx512 = avx512;
    uint8_t *xq = malloc((size_t)(n / 256) * Q8K_BYTES);
    quantize_row_q8_K(x, xq, n);
    float s; pick(qt)(n, &s, 0, row, 0, xq, 0, 1);
    free(xq); g_avx512 = keep;
    return s;
}

// Team size for moe_cpu. Explicit because inside a torch process these OpenMP calls bind to
// torch's bundled libgomp (loaded RTLD_GLOBAL), whose default team is torch's thread count.
static int g_nth = 0;

void cpumoe_init(void) { ggml_cpu_init(); }

// Pin each thread of the OpenMP team to one CPU (thread t -> cpus[t]). Done here rather than
// with OMP_PROC_BIND because torch's bundled libgomp also reads that variable and pins the
// Python main thread to cpu 0 first, which collapses this runtime's places onto one CPU.
int cpumoe_pin(int n, const int *cpus) {   // returns team size * 1000 + failed pins
    int team = 0, bad = 0;
    #pragma omp parallel num_threads(n) reduction(+:bad)
    {
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpus[omp_get_thread_num()], &s);
        bad += pthread_setaffinity_np(pthread_self(), sizeof s, &s) != 0;
        #pragma omp master
        team = omp_get_num_threads();
    }
    g_nth = n;
    return team * 1000 + bad;
}

void moe_cpu(int E, const uint8_t **G, const uint8_t **U, const uint8_t **D,
             int qg, int qu, int qd, int nff, int ne, long g_rb, long u_rb, long d_rb,
             const float *x, const float *w, float lim, float *h /*E*nff*/, float *y /*ne*/) {
    vdot_t vg = pick(qg), vu = pick(qu), vd = pick(qd);
    uint8_t *xq = malloc((size_t)(ne / 256) * Q8K_BYTES);
    quantize_row_q8_K(x, xq, ne);
    int nth = g_nth > 0 ? g_nth : omp_get_max_threads();
    #pragma omp parallel for collapse(2) schedule(static) num_threads(nth)
    for (int e = 0; e < E; e++)
        for (int r = 0; r < nff; r++) {
            float g, u;
            vg(ne, &g, 0, G[e] + (size_t)r * g_rb, 0, xq, 0, 1);
            vu(ne, &u, 0, U[e] + (size_t)r * u_rb, 0, xq, 0, 1);
            g = g < lim ? g : lim;
            u = u < -lim ? -lim : (u > lim ? lim : u);
            h[(size_t)e * nff + r] = g / (1.f + expf(-g)) * u;
        }
    size_t hb = (size_t)(nff / 256) * Q8K_BYTES;
    uint8_t *hq = malloc(hb * E);
    for (int e = 0; e < E; e++) quantize_row_q8_K(h + (size_t)e * nff, hq + hb * e, nff);
    #pragma omp parallel for schedule(static) num_threads(nth)
    for (int r = 0; r < ne; r++) {
        float acc = 0.f, v;
        for (int e = 0; e < E; e++) {
            vd(nff, &v, 0, D[e] + (size_t)r * d_rb, 0, hq + hb * e, 0, 1);
            acc += w[e] * v;
        }
        y[r] = acc;
    }
    free(xq); free(hq);
}


// ---------------- NUMA-split variant ----------------
// Each expert matrix is split by rows into g_nodes equal parts; part k's pages live on NUMA
// node k (cpumoe_place) and are computed by the threads pinned to node k: team threads are
// grouped node by node (cpumoe_pin with node-ordered cpus, g_nodes groups of g_nth/g_nodes).
static int g_nodes = 1;
void cpumoe_set_nodes(int n) { g_nodes = n > 0 ? n : 1; }

// Optional per-call profile of moe_cpu_numa (cpumoe_prof(1) resets and enables). Sums in seconds:
// [0] calls, [1] experts, [2] total, [3] wake (entry -> last thread running), [4] phase 1 wall,
// [5] phase 1 mean thread busy, [6] quantize h, [7] phase 3 wall, [8] phase 3 mean busy, [9] join.
#include <time.h>
static int g_prof = 0;
static double g_ps[10];
static inline double pnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
#define PMAXT 64
static double g_pth[2][PMAXT];          /* per-thread busy sums: phase 1, phase 3 */
void cpumoe_prof(int on) { g_prof = on; for (int i = 0; i < 10; i++) g_ps[i] = 0; for (int i = 0; i < PMAXT; i++) g_pth[0][i] = g_pth[1][i] = 0; }
void cpumoe_prof_get(double *out) { for (int i = 0; i < 10; i++) out[i] = g_ps[i]; }
void cpumoe_prof_threads(double *out, int n) { for (int i = 0; i < n; i++) { out[i] = g_pth[0][i]; out[n + i] = g_pth[1][i]; } }


void moe_cpu_numa(int E, const uint8_t **G, const uint8_t **U, const uint8_t **D,
                  int qg, int qu, int qd, int nff, int ne, long g_rb, long u_rb, long d_rb,
                  const float *x, const float *w, float lim, float *h, float *y) {
    vdot_t vg = pick(qg), vu = pick(qu), vd = pick(qd);
    int nth = g_nth > 0 ? g_nth : omp_get_max_threads();
    int NN = g_nodes, tpn = nth / NN;
    const int prof = g_prof && nth <= PMAXT;
    double pt[4][PMAXT], T0 = prof ? pnow() : 0;
    uint8_t *xq = malloc((size_t)(ne / 256) * Q8K_BYTES);
    quantize_row_q8_K(x, xq, ne);
    size_t hb = (size_t)(nff / 256) * Q8K_BYTES;
    uint8_t *hq = malloc(hb * E);
    #pragma omp parallel num_threads(nth)
    {
        int tid = omp_get_thread_num(), node = tid / tpn, j = tid % tpn;
        if (prof) pt[0][tid] = pnow();
        if (node >= NN) { node = NN - 1; j = tpn; }          /* leftover threads idle in phase 1/3 */
        int a = (int)((long)nff * node / NN), half = (int)((long)nff * (node + 1) / NN) - a;   /* node's rows */
        int r0 = a + (int)((long)j * half / tpn), r1 = a + (int)((long)(j + 1) * half / tpn);
        if (j >= tpn) r1 = r0;
        for (int e = 0; e < E; e++)
            for (int r = r0; r < r1; r++) {
                float g, u;
                vg(ne, &g, 0, G[e] + (size_t)r * g_rb, 0, xq, 0, 1);
                vu(ne, &u, 0, U[e] + (size_t)r * u_rb, 0, xq, 0, 1);
                g = g < lim ? g : lim;
                u = u < -lim ? -lim : (u > lim ? lim : u);
                h[(size_t)e * nff + r] = g / (1.f + expf(-g)) * u;
            }
        if (prof) pt[1][tid] = pnow();
        #pragma omp barrier
        for (int e = tid; e < E; e += nth) quantize_row_q8_K(h + (size_t)e * nff, hq + hb * e, nff);
        #pragma omp barrier
        if (prof) pt[2][tid] = pnow();
        int a2 = (int)((long)ne * node / NN), half2 = (int)((long)ne * (node + 1) / NN) - a2;
        int s0 = a2 + (int)((long)j * half2 / tpn), s1 = a2 + (int)((long)(j + 1) * half2 / tpn);
        if (j >= tpn) s1 = s0;
        for (int r = s0; r < s1; r++) {
            float acc = 0.f, v;
            for (int e = 0; e < E; e++) {
                vd(nff, &v, 0, D[e] + (size_t)r * d_rb, 0, hq + hb * e, 0, 1);
                acc += w[e] * v;
            }
            y[r] = acc;
        }
        if (prof) pt[3][tid] = pnow();
    }
    free(xq); free(hq);
    if (prof) {
        double T1 = pnow(), mx[4] = {0, 0, 0, 0}, b1 = 0, b3 = 0;
        for (int t = 0; t < nth; t++) {
            for (int k = 0; k < 4; k++) if (pt[k][t] > mx[k]) mx[k] = pt[k][t];
            b1 += pt[1][t] - pt[0][t]; b3 += pt[3][t] - pt[2][t];
            g_pth[0][t] += pt[1][t] - pt[0][t]; g_pth[1][t] += pt[3][t] - pt[2][t];
        }
        g_ps[0] += 1; g_ps[1] += E; g_ps[2] += T1 - T0; g_ps[3] += mx[0] - T0; g_ps[4] += mx[1] - mx[0];
        g_ps[5] += b1 / nth; g_ps[6] += mx[2] - mx[1]; g_ps[7] += mx[3] - mx[2]; g_ps[8] += b3 / nth; g_ps[9] += T1 - mx[3];
    }
}

// One call per MoE layer with no per-call marshaling: the layer's three expert tensors are
// bound once (cpumoe_bind_layer); a call names the routed experts to compute by a bit mask over
// the 8 routed slots (sel[j], w[j] for each set bit j, in slot order).
#define BIND_MAXL 128
typedef struct { const uint8_t *g, *u, *d; int qg, qu, qd, nff, ne; long g_rb, u_rb, d_rb; } bound_t;
static bound_t g_bound[BIND_MAXL];
static float *g_hbuf = 0;
static size_t g_hcap = 0;

void cpumoe_bind_layer(int L, const uint8_t *g, const uint8_t *u, const uint8_t *d, int qg, int qu, int qd,
                       int nff, int ne, long g_rb, long u_rb, long d_rb) {
    if (L < 0 || L >= BIND_MAXL) return;
    g_bound[L] = (bound_t){g, u, d, qg, qu, qd, nff, ne, g_rb, u_rb, d_rb};
}

int moe_cpu_layer(int L, unsigned mask, const int64_t *sel, const float *w8, const float *x, float lim, float *y) {
    const bound_t *b = &g_bound[L];
    // not on this stack: the team reads these while this thread works next to them
    static struct { const uint8_t *G[32], *U[32], *D[32]; float w[32]; } __attribute__((aligned(64))) a;
    const uint8_t **G = a.G, **U = a.U, **D = a.D;
    float *w = a.w;
    int E = 0;
    for (int j = 0; j < 32; j++)
        if (mask >> j & 1) {
            size_t e = (size_t)sel[j];
            G[E] = b->g + e * b->nff * b->g_rb; U[E] = b->u + e * b->nff * b->u_rb; D[E] = b->d + e * b->ne * b->d_rb;
            w[E++] = w8[j];
        }
    if (!E) return 0;
    if (g_hcap < (size_t)E * b->nff) {
        free(g_hbuf); g_hcap = (size_t)32 * b->nff; g_hbuf = malloc(g_hcap * sizeof(float));
    }
    moe_cpu_numa(E, G, U, D, b->qg, b->qu, b->qd, b->nff, b->ne, b->g_rb, b->u_rb, b->d_rb, x, w, lim, g_hbuf, y);
    return E;
}

// T-token NUMA-split variant (speculative verification): the union of the T tokens' experts,
// w[e*T + t] = token t's weight for expert e (0: t does not use e). Each weight row is read
// once and dotted with every token that uses it (the second dot hits L1).
#define TMAX 4
void moe_cpu_numa_t(int E, int T, const uint8_t **G, const uint8_t **U, const uint8_t **D,
                    int qg, int qu, int qd, int nff, int ne, long g_rb, long u_rb, long d_rb,
                    const float *x /*T*ne*/, const float *w /*E*T*/, float lim,
                    float *h /*E*T*nff*/, float *y /*T*ne*/) {
    vdot_t vg = pick(qg), vu = pick(qu), vd = pick(qd);
    int nth = g_nth > 0 ? g_nth : omp_get_max_threads();
    int NN = g_nodes, tpn = nth / NN;
    size_t xb = (size_t)(ne / 256) * Q8K_BYTES, hb = (size_t)(nff / 256) * Q8K_BYTES;
    uint8_t *xq = malloc(xb * T);
    for (int t = 0; t < T; t++) quantize_row_q8_K(x + (size_t)t * ne, xq + xb * t, ne);
    uint8_t *hq = malloc(hb * E * T);
    #pragma omp parallel num_threads(nth)
    {
        int tid = omp_get_thread_num(), node = tid / tpn, j = tid % tpn;
        if (node >= NN) { node = NN - 1; j = tpn; }
        int a = (int)((long)nff * node / NN), half = (int)((long)nff * (node + 1) / NN) - a;   /* node's rows */
        int r0 = a + (int)((long)j * half / tpn), r1 = a + (int)((long)(j + 1) * half / tpn);
        if (j >= tpn) r1 = r0;
        for (int e = 0; e < E; e++)
            for (int r = r0; r < r1; r++) {
                for (int t = 0; t < T; t++) {
                    if (w[e * T + t] == 0.f) continue;
                    float g, u;
                    vg(ne, &g, 0, G[e] + (size_t)r * g_rb, 0, xq + xb * t, 0, 1);
                    vu(ne, &u, 0, U[e] + (size_t)r * u_rb, 0, xq + xb * t, 0, 1);
                    g = g < lim ? g : lim;
                    u = u < -lim ? -lim : (u > lim ? lim : u);
                    h[((size_t)e * T + t) * nff + r] = g / (1.f + expf(-g)) * u;
                }
            }
        #pragma omp barrier
        for (int k = tid; k < E * T; k += nth)
            if (w[k] != 0.f) quantize_row_q8_K(h + (size_t)k * nff, hq + hb * k, nff);
        #pragma omp barrier
        int a2 = (int)((long)ne * node / NN), half2 = (int)((long)ne * (node + 1) / NN) - a2;
        int s0 = a2 + (int)((long)j * half2 / tpn), s1 = a2 + (int)((long)(j + 1) * half2 / tpn);
        if (j >= tpn) s1 = s0;
        for (int r = s0; r < s1; r++) {
            float acc[TMAX] = {0}, v;
            for (int e = 0; e < E; e++)
                for (int t = 0; t < T; t++) {
                    float we = w[e * T + t];
                    if (we == 0.f) continue;
                    vd(nff, &v, 0, D[e] + (size_t)r * d_rb, 0, hq + hb * (e * T + t), 0, 1);
                    acc[t] += we * v;
                }
            for (int t = 0; t < T; t++) y[(size_t)t * ne + r] = acc[t];
        }
    }
    free(xq); free(hq);
}

// moe_cpu_numa_t on a bound layer (cpumoe_bind_layer): bit 8*t + j of mask set = compute token
// t's routed slot j (sel[t*8 + j], w8[t*8 + j]). The union is built in the order the Python
// wrapper used (token 0's slots, then each later token's new experts), so sums come out the same.
int moe_cpu_layer_t(int L, int T, unsigned mask, const int64_t *sel, const float *w8, const float *x, float lim, float *y) {
    const bound_t *b = &g_bound[L];
    static struct { const uint8_t *G[32], *U[32], *D[32]; int64_t id[32]; float w[32 * TMAX]; } __attribute__((aligned(64))) a;
    if (T < 1 || T > TMAX) return -1;
    int E = 0;
    for (int t = 0; t < T; t++)
        for (int j = 0; j < 8; j++) {
            if (!(mask >> (8 * t + j) & 1)) continue;
            int64_t e = sel[t * 8 + j];
            int k = 0;
            while (k < E && a.id[k] != e) k++;
            if (k == E) {
                a.id[E] = e;
                a.G[E] = b->g + (size_t)e * b->nff * b->g_rb; a.U[E] = b->u + (size_t)e * b->nff * b->u_rb;
                a.D[E] = b->d + (size_t)e * b->ne * b->d_rb;
                for (int s = 0; s < T; s++) a.w[E * T + s] = 0.f;
                E++;
            }
            a.w[k * T + t] = w8[t * 8 + j];
        }
    if (!E) return 0;
    if (g_hcap < (size_t)E * T * b->nff) {
        free(g_hbuf); g_hcap = (size_t)32 * TMAX * b->nff; g_hbuf = malloc(g_hcap * sizeof(float));
    }
    moe_cpu_numa_t(E, T, a.G, a.U, a.D, b->qg, b->qu, b->qd, b->nff, b->ne, b->g_rb, b->u_rb, b->d_rb, x, a.w, lim, g_hbuf, y);
    return E;
}

// Move the (mapped) pages fully inside [addr, addr+len) to NUMA node `node`.
// Returns pages that could not be moved (status < 0, or every page of a batch whose call failed).
long cpumoe_place(const uint8_t *addr, size_t len, int node) {
    long pg = sysconf(_SC_PAGESIZE);
    uintptr_t lo = ((uintptr_t)addr + pg - 1) / pg * pg, hi = ((uintptr_t)addr + len) / pg * pg;
    if (hi <= lo) return 0;
    long n = (long)((hi - lo) / pg), bad = 0;
    enum { B = 4096 };
    void *pages[B]; int nodes[B], status[B];
    for (long i = 0; i < n; i += B) {
        int m = (int)(n - i < B ? n - i : B);
        for (int k = 0; k < m; k++) { pages[k] = (void *)(lo + (uintptr_t)(i + k) * pg); nodes[k] = node; }
        if (syscall(SYS_move_pages, 0, (unsigned long)m, pages, nodes, status, 2 /* MPOL_MF_MOVE */) < 0) {
            bad += m;                                  /* the call failed (e.g. EPERM): statuses are not valid */
            continue;
        }
        for (int k = 0; k < m; k++) bad += status[k] < 0;
    }
    return bad;
}

// Fraction of mapped pages in [addr, addr+len) on each node (query only), into frac[nn].
void cpumoe_where(const uint8_t *addr, size_t len, int nn, double *frac) {
    long pg = sysconf(_SC_PAGESIZE);
    uintptr_t lo = ((uintptr_t)addr + pg - 1) / pg * pg, hi = ((uintptr_t)addr + len) / pg * pg;
    for (int k = 0; k < nn; k++) frac[k] = 0;
    if (hi <= lo) return;
    long n = (long)((hi - lo) / pg), tot = 0;
    enum { B = 4096 };
    void *pages[B]; int status[B];
    for (long i = 0; i < n; i += B) {
        int m = (int)(n - i < B ? n - i : B);
        for (int k = 0; k < m; k++) pages[k] = (void *)(lo + (uintptr_t)(i + k) * pg);
        if (syscall(SYS_move_pages, 0, (unsigned long)m, pages, NULL, status, 0) < 0) continue;
        for (int k = 0; k < m; k++) { tot++; if (status[k] >= 0 && status[k] < nn) frac[status[k]] += 1; }
    }
    for (int k = 0; k < nn; k++) frac[k] /= tot ? tot : 1;
}
