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
#define SYNC() asm volatile("bar.warp.sync 0xffffffff;" ::: "memory")
#else
#define SYNC() barrier(CLK_LOCAL_MEM_FENCE)
#endif

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
static inline float shfl_at(float v, int src) {
    float r;
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(r) : "f"(v), "r"(src));
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

    __local float a[D * D], m[D * D], b[D], w[D], r[32], sv[D], rd[D];
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
    SYNC();

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
        SYNC();

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
            SYNC();
            for (int q = lane; q < DP; q += LANES) {
                float lf = 0.0f;
                if (q < D) lf = MUL(r[q], w[q]);
                t[q] = lf;
            }
            SYNC();
            for (int st = DP / 2; st >= 1; st >>= 1) {
                for (int q = lane; q < st; q += LANES) {
                    float v = t[q + st];
                    float s = t[q];
                    t[q] = s + v;
                }
                SYNC();
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
                SYNC();
#endif
                for (int p = lane; p < D; p += LANES) {
                    float tt = MUL(r[p], yf);
                    float lb = MUL(lam, b[p]);
                    b[p] = lb + tt;
                }
                // Four elements of the lower triangle per lane at a time, the
                // loads together before the stores.
                for (int e = lane; e < NTRI; e += 4 * LANES) {
                    const int n1 = e + LANES, n2 = e + 2 * LANES, n3 = e + 3 * LANES;
                    const uchar2 p0 = tri[e];
                    const uchar2 p1 = (n1 < NTRI) ? tri[n1] : p0;
                    const uchar2 p2 = (n2 < NTRI) ? tri[n2] : p0;
                    const uchar2 p3 = (n3 < NTRI) ? tri[n3] : p0;
                    const int i0_ = p0.x * D + p0.y, i1_ = p1.x * D + p1.y, i2_ = p2.x * D + p2.y, i3_ = p3.x * D + p3.y;
                    const float a0 = a[i0_], a1 = a[i1_], a2 = a[i2_], a3 = a[i3_];
                    const float x0 = r[p0.x], y0 = r[p0.y], x1 = r[p1.x], y1 = r[p1.y];
                    const float x2 = r[p2.x], y2 = r[p2.y], x3 = r[p3.x], y3 = r[p3.y];
                    const float t0 = MUL(x0, y0), t1 = MUL(x1, y1), t2 = MUL(x2, y2), t3 = MUL(x3, y3);
                    const float l0 = MUL(lam, a0), l1 = MUL(lam, a1), l2 = MUL(lam, a2), l3 = MUL(lam, a3);
                    a[i0_] = l0 + t0;
                    if (n1 < NTRI) a[i1_] = l1 + t1;
                    if (n2 < NTRI) a[i2_] = l2 + t2;
                    if (n3 < NTRI) a[i3_] = l3 + t3;
                }
#ifdef FAST
                SYNC();
#endif
            }
#endif
#ifndef FAST
            SYNC();
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
                SYNC();
#endif
                for (int idx = lane; idx < D * D; idx += LANES) m[idx] = a[idx];
                for (int idx = lane; idx < D; idx += LANES) sv[idx] = b[idx];
                SYNC();
                if (lane == 0) {
                    float tr = 0.0f;
                    for (int q = 0; q < D; q++) tr = tr + m[q * D + q];
                    float rq = fdiv(tr, (float)D);
                    float rs = MUL(rq, 1e-6f);
                    ridge = rs + 1e-6f;
                    okf = 1;
                }
                SYNC();
                for (int idx = lane; idx < D; idx += LANES) m[idx * D + idx] = m[idx * D + idx] + ridge;
                SYNC();

                // Cholesky, a column at a time: the pivot (inverted once), the
                // column times that inverse, then the rest of the lower triangle
                // loses one product per element, in the order of the columns.
                for (int j = 0; j < D; j++) {
                    if (lane == 0) {
                        float v = m[j * D + j];
                        float dj = 1.0f;
                        if (v <= 0.0f) okf = 0;
                        else dj = fsq(v);
                        m[j * D + j] = dj;
                        rd[j] = fdiv(1.0f, dj);
                    }
                    SYNC();
                    const float rj = rd[j];
                    for (int ii = j + 1 + lane; ii < D; ii += LANES) m[ii * D + j] = MUL(m[ii * D + j], rj);
                    SYNC();
#ifndef NO_TRAIL
                    // One row per lane; the loads of a batch of four are issued
                    // together, the compiler cannot tell the store of one
                    // element from the loads of the next.
                    for (int p = j + 1 + lane; p < D; p += LANES) {
                        const float lpj = m[p * D + j];
                        int q = j + 1;
                        for (; q + 3 <= p; q += 4) {
                            float a0 = m[p * D + q], a1 = m[p * D + q + 1], a2 = m[p * D + q + 2], a3 = m[p * D + q + 3];
                            float b0 = m[q * D + j], b1 = m[(q + 1) * D + j], b2 = m[(q + 2) * D + j], b3 = m[(q + 3) * D + j];
                            float c0 = MUL(lpj, b0), c1 = MUL(lpj, b1), c2 = MUL(lpj, b2), c3 = MUL(lpj, b3);
                            m[p * D + q] = a0 - c0;
                            m[p * D + q + 1] = a1 - c1;
                            m[p * D + q + 2] = a2 - c2;
                            m[p * D + q + 3] = a3 - c3;
                        }
                        for (; q <= p; q++) {
                            float a0 = m[p * D + q];
                            float c0 = MUL(lpj, m[q * D + j]);
                            m[p * D + q] = a0 - c0;
                        }
                    }
#endif
                    SYNC();
                }
#ifdef FAST
                // Substitutions on a register per lane: the unknown `ii` is
                // scaled by its inverted pivot and taken off the others.
#ifndef NO_SUBST
                float x = (lane < D) ? sv[lane] : 0.0f;
                for (int ii = 0; ii < D; ii++) {
                    float si = MUL(shfl_at(x, ii), rd[ii]);
                    if (lane == ii) x = si;
                    else if (lane > ii && lane < D) {
                        float pr = MUL(m[lane * D + ii], si);
                        x = x - pr;
                    }
                }
                for (int ii = D - 1; ii >= 0; ii--) {
                    float si = MUL(shfl_at(x, ii), rd[ii]);
                    if (lane == ii) x = si;
                    else if (lane < ii) {
                        float pr = MUL(m[ii * D + lane], si);
                        x = x - pr;
                    }
                }
#endif
                if (okf && lane < D) w[lane] = x;
                SYNC();
                wv = (lane < D) ? w[lane] : 0.0f;
#else
                for (int ii = 0; ii < D; ii++) {
                    if (lane == 0) sv[ii] = MUL(sv[ii], rd[ii]);
                    SYNC();
                    for (int rr = ii + 1 + lane; rr < D; rr += LANES) {
                        float pr = MUL(m[rr * D + ii], sv[ii]);
                        sv[rr] = sv[rr] - pr;
                    }
                    SYNC();
                }
                for (int ii = D - 1; ii >= 0; ii--) {
                    if (lane == 0) sv[ii] = MUL(sv[ii], rd[ii]);
                    SYNC();
                    for (int rr = lane; rr < ii; rr += LANES) {
                        float pr = MUL(m[ii * D + rr], sv[ii]);
                        sv[rr] = sv[rr] - pr;
                    }
                    SYNC();
                }
                if (okf) {
                    for (int idx = lane; idx < D; idx += LANES) w[idx] = sv[idx];
                }
                SYNC();
#endif
            }
#endif
        }
        for (int j = lane; j < cnt; j += LANES) O[i0 + j] = ol[j];
        SYNC();
    }
}
