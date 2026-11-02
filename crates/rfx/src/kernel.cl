// The predictor of pred.rs on a GPU: one work-group of 32 work-items per
// segment, the segment's samples walked in order by the whole group.
//
// mode 0 (coding): `yin` holds the samples, `out` receives the predictions.
// mode 1 (decoding): `yin` holds the residuals, `out` receives the samples.
//
// It must give the CPU's bits, so the arithmetic is the CPU's: f32 with every
// product and sum rounded on its own (no statement holds a multiply and an
// add on the same value), division and square root correctly rounded, the dot
// product as the same binary tree, the Cholesky solve in the same order of
// operations per element.
//
// On NVIDIA with up to 32 regressors (FAST) the group is one warp and each
// work-item holds one regressor and one weight in registers: the dot
// product's tree and the shift of the own history are lane shuffles, with no
// barrier per sample. Elsewhere the same steps go through local memory.
//
// Defines: OWN, TAPS, USEREF, STRIDE, EVERY (powers of two), D (regressors),
// DP (leaves), and NVIDIA on NVIDIA devices.
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

#define LANES 32
#define BLK 256
#define K2 (2 * TAPS + 1)
#define NTRI (D * (D + 1) / 2)

#if defined(NVIDIA) && D <= 32
#define FAST 1
#endif

#ifdef NVIDIA
// NVIDIA's compiler fuses a*b+c into a fused multiply-add whatever the
// pragma says, and folds an f64 division detour into an f32 division it does
// not round correctly: ask for the rounded instructions.
static inline float MUL(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
static inline float fdiv(float a, float b) {
    float r;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
static inline float fsq(float a) {
    float r;
    asm("sqrt.rn.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
static inline float shfl_down(float v, int delta) {
    float r;
    asm volatile("shfl.sync.down.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(r) : "f"(v), "r"(delta));
    return r;
}
static inline float shfl_up(float v, int delta) {
    float r;
    asm volatile("shfl.sync.up.b32 %0, %1, %2, 0x0, 0xffffffff;" : "=f"(r) : "f"(v), "r"(delta));
    return r;
}
static inline float shfl_first(float v) {
    float r;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;" : "=f"(r) : "f"(v));
    return r;
}
#else
static inline float MUL(float a, float b) { return a * b; }
static inline float fdiv(float a, float b) { return (float)((double)a / (double)b); }
static inline float fsq(float a) { return (float)sqrt((double)a); }
#endif

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void rfx_seg(__global const int* yin, __global const float* refwin, __global int* out,
             __global const int* lens, const int S, const int RW, const float lam, const int mode)
{
    const int lane = get_local_id(0);
    const int seg = get_group_id(0);
    const int n = lens[seg];
    __global const int* X = yin + (long)seg * S;
    __global const float* R0 = refwin + (long)seg * RW;
    __global int* O = out + (long)seg * S;

    __local float a[D * D], m[D * D], b[D], w[D], r[32], sv[D];
    __local uchar2 tri[NTRI];
    __local float rwl[BLK + K2];
    __local int xl[BLK], ol[BLK];
#ifndef FAST
    __local float t[DP];
    __local int ring[64];
#endif
    __local int okf;
    __local float ridge;

    for (int i = lane; i < D * D; i += LANES) a[i] = 0.0f;
    for (int i = lane; i < D; i += LANES) { b[i] = 0.0f; w[i] = 0.0f; }
    for (int p = lane; p < D; p += LANES) {
        const int base = p * (p + 1) / 2;
        for (int q = 0; q <= p; q++) tri[base + q] = (uchar2)((uchar)p, (uchar)q);
    }
#ifndef FAST
    for (int i = lane; i < 64; i += LANES) ring[i] = 0;
#endif
    r[lane] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef FAST
    float rv = 0.0f;  // regressor `lane`
    float wv = 0.0f;  // weight `lane`
#endif

    for (int i0 = 0; i0 < n; i0 += BLK) {
        const int cnt = min(BLK, n - i0);
        for (int j = lane; j < cnt; j += LANES) xl[j] = X[i0 + j];
#if USEREF
        for (int j = lane; j < cnt + K2 - 1; j += LANES) rwl[j] = R0[i0 + j];
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int jj = 0; jj < cnt; jj++) {
            const int i = i0 + jj;
            int pred;

#ifdef FAST
#if USEREF
            if (lane >= OWN) rv = (lane < D) ? rwl[jj + lane - OWN] : 0.0f;
#endif
            float tl = MUL(rv, wv);
            for (int st = DP / 2; st >= 1; st >>= 1) {
                float v = shfl_down(tl, st);
                tl = tl + v;
            }
            float s0 = shfl_first(tl);
            float pf = floor(s0 + 0.5f);
            pf = clamp(pf, -4194304.0f, 4194304.0f);
            pred = (int)pf;
#else
            for (int idx = lane; idx < D; idx += LANES) {
                float v = 0.0f;
                if (idx < OWN) {
                    if (i > idx) v = (float)ring[(i - 1 - idx) & 63];
                }
#if USEREF
                else {
                    v = rwl[jj + idx - OWN];
                }
#endif
                r[idx] = v;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int q = lane; q < DP; q += LANES) {
                float lf = 0.0f;
                if (q < D) lf = MUL(r[q], w[q]);
                t[q] = lf;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int st = DP / 2; st >= 1; st >>= 1) {
                for (int q = lane; q < st; q += LANES) {
                    float v = t[q + st];
                    float s = t[q];
                    t[q] = s + v;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            float pf = floor(t[0] + 0.5f);
            pf = clamp(pf, -4194304.0f, 4194304.0f);
            pred = (int)pf;
#endif

            int y;
            if (mode == 0) {
                y = xl[jj];
                if (lane == 0) ol[jj] = pred;
            } else {
                y = pred + xl[jj];
                if (lane == 0) ol[jj] = y;
            }
            const float yf = (float)y;
#ifndef FAST
            if (lane == 0) ring[i & 63] = y;
#endif

#ifndef NO_UPDATE
            if ((i & (STRIDE - 1)) == 0) {
#ifdef FAST
                r[lane] = rv;
                barrier(CLK_LOCAL_MEM_FENCE);
#endif
                for (int p = lane; p < D; p += LANES) {
                    float tt = MUL(r[p], yf);
                    float lb = MUL(lam, b[p]);
                    b[p] = lb + tt;
                }
                for (int e = lane; e < NTRI; e += LANES) {
                    const uchar2 pq = tri[e];
                    const int ai = pq.x * D + pq.y;
                    float tt = MUL(r[pq.x], r[pq.y]);
                    float la = MUL(lam, a[ai]);
                    a[ai] = la + tt;
                }
#ifdef FAST
                barrier(CLK_LOCAL_MEM_FENCE);
#endif
            }
#endif
#ifndef FAST
            barrier(CLK_LOCAL_MEM_FENCE);
#endif

#ifdef FAST
            {
                float up = shfl_up(rv, 1);
                if (lane == 0) rv = yf;
                else if (lane < OWN) rv = up;
            }
#endif

#ifndef NO_SOLVE
            if ((i & (EVERY - 1)) == EVERY - 1 && i > 4 * D) {
#ifdef FAST
                barrier(CLK_LOCAL_MEM_FENCE);
#endif
                for (int idx = lane; idx < D * D; idx += LANES) m[idx] = a[idx];
                for (int idx = lane; idx < D; idx += LANES) sv[idx] = b[idx];
                barrier(CLK_LOCAL_MEM_FENCE);
                if (lane == 0) {
                    float tr = 0.0f;
                    for (int q = 0; q < D; q++) tr = tr + m[q * D + q];
                    float rq = fdiv(tr, (float)D);
                    float rs = MUL(rq, 1e-6f);
                    ridge = rs + 1e-6f;
                    okf = 1;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int idx = lane; idx < D; idx += LANES) m[idx * D + idx] = m[idx * D + idx] + ridge;
                barrier(CLK_LOCAL_MEM_FENCE);

                for (int j = 0; j < D; j++) {
                    if (lane == 0) {
                        float v = m[j * D + j];
                        for (int k = 0; k < j; k++) {
                            float pr = MUL(m[j * D + k], m[j * D + k]);
                            v = v - pr;
                        }
                        if (v <= 0.0f) { okf = 0; m[j * D + j] = 1.0f; }
                        else m[j * D + j] = fsq(v);
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    for (int ii = j + 1 + lane; ii < D; ii += LANES) {
                        float v = m[ii * D + j];
                        for (int k = 0; k < j; k++) {
                            float pr = MUL(m[ii * D + k], m[j * D + k]);
                            v = v - pr;
                        }
                        m[ii * D + j] = fdiv(v, m[j * D + j]);
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                for (int ii = 0; ii < D; ii++) {
                    if (lane == 0) sv[ii] = fdiv(sv[ii], m[ii * D + ii]);
                    barrier(CLK_LOCAL_MEM_FENCE);
                    for (int rr = ii + 1 + lane; rr < D; rr += LANES) {
                        float pr = MUL(m[rr * D + ii], sv[ii]);
                        sv[rr] = sv[rr] - pr;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                for (int ii = D - 1; ii >= 0; ii--) {
                    if (lane == 0) sv[ii] = fdiv(sv[ii], m[ii * D + ii]);
                    barrier(CLK_LOCAL_MEM_FENCE);
                    for (int rr = lane; rr < ii; rr += LANES) {
                        float pr = MUL(m[ii * D + rr], sv[ii]);
                        sv[rr] = sv[rr] - pr;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (okf) {
                    for (int idx = lane; idx < D; idx += LANES) w[idx] = sv[idx];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
#ifdef FAST
                wv = (lane < D) ? w[lane] : 0.0f;
#endif
            }
#endif
        }
        for (int j = lane; j < cnt; j += LANES) O[i0 + j] = ol[j];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
