// CPU routed-expert MoE for glm5-next from raw ggml k-quant bytes (Q4_K=12, Q5_K=13, Q6_K=14).
// Same block math as the validated GPU kernels (qmv.py), fp32 accumulate.
//   moe_cpu: y = sum_e w[e] * down_e( silu(min(gate_e x, L)) * clamp(up_e x, -L, L) )
// One call per layer handles all of that layer's CPU experts; OpenMP over (expert,row).
// Build: gcc -O3 -march=native -ffast-math -fopenmp -shared -fPIC cpu_moe.c -o libcpumoe.so
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <immintrin.h>

static inline float h2f(const uint8_t *p) { uint16_t h; memcpy(&h, p, 2); return _cvtsh_ss(h); }

static inline void sm_k4(int j, const uint8_t *q, float *s, float *m) {
    if (j < 4) { *s = q[j] & 63; *m = q[j + 4] & 63; }
    else { *s = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// xs[b*8 + j] = sum of x over sub-block j of block b (row-independent; precomputed per vector)
static void xsums(const float *x, int nblk, float *xs) {
    for (int b = 0; b < nblk; b++)
        for (int j = 0; j < 8; j++) {
            float s = 0; const float *p = x + b * 256 + j * 32;
            for (int l = 0; l < 32; l++) s += p[l];
            xs[b * 8 + j] = s;
        }
}

static float dot_q45k(const uint8_t *row, const float *x, const float *xs, int nblk, int is5) {
    const int BB = is5 ? 176 : 144, QS = is5 ? 48 : 16;
    float acc = 0.f;
    for (int b = 0; b < nblk; b++) {
        const uint8_t *bp = row + (size_t)b * BB;
        const float d = h2f(bp), dmin = h2f(bp + 2);
        const uint8_t *sc = bp + 4, *qs = bp + QS, *qh = bp + 16;
        const float *xb = x + b * 256;
        for (int c = 0; c < 4; c++) {
            float s0, m0, s1, m1;
            sm_k4(2 * c, sc, &s0, &m0); sm_k4(2 * c + 1, sc, &s1, &m1);
            const uint8_t *q = qs + 32 * c;
            const float *xl = xb + 64 * c, *xh = xl + 32;
            float ql = 0.f, qh_ = 0.f;
            if (is5) {
                const int sl = 2 * c, sh = 2 * c + 1;
                for (int l = 0; l < 32; l++) {
                    ql  += (float)((q[l] & 15) | (((qh[l] >> sl) & 1) << 4)) * xl[l];
                    qh_ += (float)((q[l] >> 4) | (((qh[l] >> sh) & 1) << 4)) * xh[l];
                }
            } else {
                for (int l = 0; l < 32; l++) { ql += (float)(q[l] & 15) * xl[l]; qh_ += (float)(q[l] >> 4) * xh[l]; }
            }
            acc += d * (s0 * ql + s1 * qh_) - dmin * (m0 * xs[b * 8 + 2 * c] + m1 * xs[b * 8 + 2 * c + 1]);
        }
    }
    return acc;
}

static float dot_q6k(const uint8_t *row, const float *x, int nblk) {
    float acc = 0.f;
    for (int b = 0; b < nblk; b++) {
        const uint8_t *bp = row + (size_t)b * 210;
        const float d = h2f(bp + 208);
        const int8_t *sc = (const int8_t *)(bp + 192);
        for (int n = 0; n < 2; n++) {
            const uint8_t *A = bp + 64 * n, *B = A + 32, *h = bp + 128 + 32 * n;
            const float *xb = x + b * 256 + 128 * n;
            const int8_t *s = sc + 8 * n;
            for (int half = 0; half < 2; half++) {
                float u1 = 0, u2 = 0, u3 = 0, u4 = 0;
                for (int l = 16 * half; l < 16 * half + 16; l++) {
                    u1 += (float)(((A[l] & 15) | ((h[l] & 3) << 4)) - 32) * xb[l];
                    u2 += (float)(((B[l] & 15) | (((h[l] >> 2) & 3) << 4)) - 32) * xb[32 + l];
                    u3 += (float)(((A[l] >> 4) | (((h[l] >> 4) & 3) << 4)) - 32) * xb[64 + l];
                    u4 += (float)(((B[l] >> 4) | (((h[l] >> 6) & 3) << 4)) - 32) * xb[96 + l];
                }
                acc += d * (s[0 + half] * u1 + s[2 + half] * u2 + s[4 + half] * u3 + s[6 + half] * u4);
            }
        }
    }
    return acc;
}

static inline float rowdot(int qt, const uint8_t *row, const float *x, const float *xs, int nblk) {
    if (qt == 14) return dot_q6k(row, x, nblk);
    return dot_q45k(row, x, xs, nblk, qt == 13);
}

// Single matvec (for validation): y[rows] = W x
void qmv_cpu(const uint8_t *W, int qt, int rows, long row_bytes, const float *x, int cols, float *y) {
    int nblk = cols / 256;
    float *xs = malloc(sizeof(float) * nblk * 8); xsums(x, nblk, xs);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) y[r] = rowdot(qt, W + (size_t)r * row_bytes, x, xs, nblk);
    free(xs);
}

void moe_cpu(int E, const uint8_t **G, const uint8_t **U, const uint8_t **D,
             int qg, int qu, int qd, int nff, int ne, long g_rb, long u_rb, long d_rb,
             const float *x, const float *w, float lim, float *h /*E*nff*/, float *y /*ne*/) {
    int nb_in = ne / 256, nb_ff = nff / 256;
    float *xs = malloc(sizeof(float) * nb_in * 8); xsums(x, nb_in, xs);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int e = 0; e < E; e++)
        for (int r = 0; r < nff; r++) {
            float g = rowdot(qg, G[e] + (size_t)r * g_rb, x, xs, nb_in);
            float u = rowdot(qu, U[e] + (size_t)r * u_rb, x, xs, nb_in);
            g = g < lim ? g : lim;
            u = u < -lim ? -lim : (u > lim ? lim : u);
            h[(size_t)e * nff + r] = g / (1.f + expf(-g)) * u;
        }
    float *hs = malloc(sizeof(float) * E * nb_ff * 8);
    for (int e = 0; e < E; e++) xsums(h + (size_t)e * nff, nb_ff, hs + (size_t)e * nb_ff * 8);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < ne; r++) {
        float acc = 0.f;
        for (int e = 0; e < E; e++)
            acc += w[e] * rowdot(qd, D[e] + (size_t)r * d_rb, h + (size_t)e * nff, hs + (size_t)e * nb_ff * 8, nb_ff);
        y[r] = acc;
    }
    free(xs); free(hs);
}
