// The predictor of pred.rs on a GPU: one work-group of 32 work-items per
// segment, the segment's samples walked in order by the whole group.
//
// mode 0 (coding): `yin` holds the samples, `out` receives the predictions.
// mode 1 (decoding): `yin` holds the residuals, `out` receives the samples.
//
// It must give the CPU's bits, so the arithmetic is the CPU's: f32 with every
// product and sum rounded on its own (no statement holds a multiply and an
// add on the same value), division and square root through f64 (correctly
// rounded for f32 operands), the dot product as the same binary tree, the
// Cholesky solve in the same order of operations per element.
//
// Defines: OWN, TAPS, USEREF, STRIDE, EVERY, D (regressors), DP (leaves).
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

#define LANES 32
#define BLK 256
#define K2 (2 * TAPS + 1)

#ifdef NVIDIA
// The compiler folds the f64 detour into an f32 division, which NVIDIA does
// not round correctly by default: ask for the rounded instructions.
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
#else
static inline float fdiv(float a, float b) { return (float)((double)a / (double)b); }
static inline float fsq(float a) { return (float)sqrt((double)a); }
#endif

// A product that the compiler cannot fuse into a following add: NVIDIA's
// compiler contracts a*b+c into a fused multiply-add whatever the pragma says,
// the CPU does not, so the product is rounded on its own (`mul.rn`).
#ifdef NVIDIA
static inline float MUL(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
#else
static inline float MUL(float a, float b) { return a * b; }
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

    __local float a[D * D], m[D * D], b[D], w[D], r[D], sv[D], t[DP];
    __local float rwl[BLK + K2];
    __local int xl[BLK], ol[BLK], ring[64];
    __local int okf;
    __local float ridge;

    for (int i = lane; i < D * D; i += LANES) a[i] = 0.0f;
    for (int i = lane; i < D; i += LANES) { b[i] = 0.0f; w[i] = 0.0f; }
    for (int i = lane; i < 64; i += LANES) ring[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i0 = 0; i0 < n; i0 += BLK) {
        const int cnt = min(BLK, n - i0);
        for (int j = lane; j < cnt; j += LANES) xl[j] = X[i0 + j];
#if USEREF
        for (int j = lane; j < cnt + K2 - 1; j += LANES) rwl[j] = R0[i0 + j];
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int jj = 0; jj < cnt; jj++) {
            const int i = i0 + jj;

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
            const int pred = (int)pf;

            int y;
            if (mode == 0) {
                y = xl[jj];
                if (lane == 0) ol[jj] = pred;
            } else {
                y = pred + xl[jj];
                if (lane == 0) ol[jj] = y;
            }
            if (lane == 0) ring[i & 63] = y;
            const float yf = (float)y;

            if (i % STRIDE == 0) {
                for (int p = lane; p < D; p += LANES) {
                    float tt = MUL(r[p], yf);
                    float lb = MUL(lam, b[p]);
                    b[p] = lb + tt;
                }
                for (int p = 0; p < D; p++) {
                    const float rp = r[p];
                    for (int q = lane; q <= p; q += LANES) {
                        float tt = MUL(rp, r[q]);
                        float la = MUL(lam, a[p * D + q]);
                        a[p * D + q] = la + tt;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (i % EVERY == EVERY - 1 && i > 4 * D) {
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
            }
        }
        for (int j = lane; j < cnt; j += LANES) O[i0 + j] = ol[j];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
