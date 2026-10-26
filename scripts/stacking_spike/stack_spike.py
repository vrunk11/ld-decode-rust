"""Spike: multi-source FM stacking prototype (numpy only).

Synthetic NTSC-like FM video, sampled by several "captures" with their own
time-base error (offset, ppm drift, wobble, line-rate jitter) and independent
noise. Measures (a) delay-tracking residual, (b) estimator quality.
"""
import sys, time
import numpy as np

FS = 40e6
OS = 4
FSO = FS * OS
HZ_IRE = 1.7e6 / 140.0
IRE0 = 8.1e6
PAD = 300e-6
FSC = 3.579545e6


# ---------------------------------------------------------------- generation
def make_message(n_os, rng):
    LINE = 10169
    ire = np.zeros(n_os)
    t = np.arange(LINE) / FSO
    bs = int(5.8e-6 * FSO)
    be = int(5.8e-6 * FSO + 9 / FSC * FSO)
    act0 = int(9.4e-6 * FSO)
    act1 = int(62.5e-6 * FSO)
    for L in range(n_os // LINE + 1):
        a = L * LINE
        seg = np.zeros(LINE)
        seg[:752] = -40
        seg[bs:be] = 20 * np.sin(2 * np.pi * FSC * t[bs:be])
        pos = act0
        lvl = rng.uniform(10, 90)
        while pos < act1:
            w = int(rng.uniform(1.0e-6, 5e-6) * FSO)
            lvl = np.clip(lvl + rng.normal(0, 25), 5, 95)
            seg[pos:min(pos + w, act1)] = lvl
            pos += w
        ph = rng.uniform(0, 2 * np.pi)
        seg[act0:act1] += 15 * np.sin(2 * np.pi * FSC * t[act0:act1] + ph)
        m = min(LINE, n_os - a)
        ire[a:a + m] = seg[:m]
    X = np.fft.rfft(ire)
    f = np.fft.rfftfreq(n_os, 1 / FSO)
    X *= np.exp(-0.5 * np.log(2) * (f / 5.0e6) ** 2)
    return np.fft.irfft(X, n_os)


def timebase(N, rng, eps, off, wob_amp, wob_f, wob_ph, jit_std, jit_corr=320):
    n = np.arange(N)
    t = n / FS * (1 + eps) + off
    t = t + wob_amp * np.sin(2 * np.pi * wob_f * n / FS + wob_ph)
    w = rng.normal(size=N + 8 * jit_corr)
    k = np.exp(-0.5 * (np.arange(-4 * jit_corr, 4 * jit_corr + 1) / jit_corr) ** 2)
    k /= np.sqrt((k ** 2).sum())
    j = np.convolve(w, k, "valid")[:N]
    return t + jit_std * j


def capture(phi, t, cnr_db, rng):
    idx = (t + PAD) * FSO
    ph = np.interp(idx, np.arange(len(phi)), phi)
    x = np.cos(ph) + rng.normal(size=len(t)) * np.sqrt(10 ** (-cnr_db / 10))
    return x, ph


def analytic(x):
    N = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(N, 1 / FS)
    H = np.zeros(N)
    up = (f >= 2.5e6) & (f < 3.5e6)
    H[up] = 0.5 * (1 - np.cos(np.pi * (f[up] - 2.5e6) / 1e6))
    mid = (f >= 3.5e6) & (f <= 12.5e6)
    H[mid] = 1.0
    dn = (f > 12.5e6) & (f <= 13.5e6)
    H[dn] = 0.5 * (1 + np.cos(np.pi * (f[dn] - 12.5e6) / 1e6))
    z = np.fft.ifft(X * 2 * H)
    # out-of-band noise estimate (real-signal variance)
    sel = (f > 16e6) & (f < 19.5e6)
    sig2 = np.mean(np.abs(X[sel]) ** 2) / N
    return z, sig2


def lowpass(x, fc, width=1e6):
    N = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(N, 1 / FS)
    H = np.ones_like(f)
    t = (f > fc - width / 2) & (f < fc + width / 2)
    H[t] = 0.5 * (1 + np.cos(np.pi * (f[t] - (fc - width / 2)) / width))
    H[f >= fc + width / 2] = 0
    return np.fft.irfft(X * H, N)


# ------------------------------------------------------------ source prep
class Src:
    pass


def prep(x):
    s = Src()
    z, sig2 = analytic(x)
    s.z = z
    s.sig2 = sig2
    p = np.empty_like(z)
    p[1:] = z[1:] * np.conj(z[:-1])
    p[0] = p[1]
    s.p = p
    s.ph = p / np.maximum(np.abs(p), 1e-12)
    s.f = np.mod(np.angle(p), 2 * np.pi) * FS / (2 * np.pi)
    # local carrier amplitude^2: block median interpolated
    a2 = np.abs(z) ** 2
    B = 128
    nb = len(a2) // B
    med = np.median(a2[: nb * B].reshape(nb, B), axis=1)
    cen = (np.arange(nb) + 0.5) * B
    s.A2 = np.interp(np.arange(len(a2)), cen, med)
    s.rel = np.minimum(1.0, np.abs(p) / np.maximum(s.A2, 1e-12)) ** 2
    s.base = lowpass(np.clip(s.f, 6e6, 11e6), 3e6)
    s.base -= s.base.mean()
    return s


# ------------------------------------------------------------- alignment
def xcorr_track(bP, bS, centers, W, R, tp):
    """For windows centred at `centers` (P index), return offset m per window
    (sub-sample) such that bS[n + tp + m] ~ bP[n]. tp: int array per window."""
    PADN = 16384
    bPp = np.concatenate([np.zeros(PADN), bP, np.zeros(PADN)])
    bSp = np.concatenate([np.zeros(PADN), bS, np.zeros(PADN)])
    L = W + 2 * R
    L = int(2 ** np.ceil(np.log2(L)))
    ia = (centers - W // 2 + PADN).astype(int)
    P = bPp[ia[:, None] + np.arange(W)[None, :]]
    P = P - P.mean(axis=1, keepdims=True)
    pn = np.sqrt((P ** 2).sum(axis=1)) + 1e-12
    ib = (centers - W // 2 + tp - R + PADN).astype(int)
    S = bSp[ib[:, None] + np.arange(W + 2 * R)[None, :]]
    FP = np.fft.rfft(P, L, axis=1)
    FSg = np.fft.rfft(S, L, axis=1)
    C = np.fft.irfft(np.conj(FP) * FSg, L, axis=1)[:, : 2 * R + 1]
    cs = np.concatenate([np.zeros((len(S), 1)), np.cumsum(S, axis=1)], axis=1)
    cs2 = np.concatenate([np.zeros((len(S), 1)), np.cumsum(S * S, axis=1)], axis=1)
    sm = cs[:, W: W + 2 * R + 1] - cs[:, : 2 * R + 1]
    sm2 = cs2[:, W: W + 2 * R + 1] - cs2[:, : 2 * R + 1]
    sn = np.sqrt(np.maximum(sm2 - sm * sm / W, 1e-9))
    C = C / (pn[:, None] * sn)
    j = np.argmax(C, axis=1)
    jj = np.clip(j, 1, 2 * R - 1)
    y0 = C[np.arange(len(j)), jj]
    ym = C[np.arange(len(j)), jj - 1]
    yp = C[np.arange(len(j)), jj + 1]
    den = ym - 2 * y0 + yp
    d = np.where(np.abs(den) > 1e-12, 0.5 * (ym - yp) / den, 0.0)
    d = np.clip(d, -1, 1)
    conf = C[np.arange(len(j)), j] / (np.median(np.abs(C), axis=1) + 1e-12)
    return (jj + d) - R, conf


def smooth_knots(v, med=5, sig=2.0):
    h = med // 2
    vp = np.pad(v, h, mode="edge")
    vm = np.array([np.median(vp[i:i + med]) for i in range(len(v))])
    if sig <= 0:
        return vm
    r = int(4 * sig)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(vm, r, mode="edge"), k, "valid")


def track_delay(bP, bS, N):
    n = np.arange(N)
    # level 1: wide windows (coarse lock stand-in for VBI / fileLoc, +-6000)
    W1, hop1, R1 = 16384, 8192, 6000
    cen1 = np.arange(W1 // 2 + 4096, N - W1 // 2 - 4096, hop1)
    m1, _ = xcorr_track(bP, bS, cen1, W1, R1, np.zeros(len(cen1), int))
    t1 = smooth_knots(m1, 5, 1.0)
    tau = np.interp(n, cen1, t1)
    tau0 = int(round(np.median(m1)))
    # level 2: ~1 line windows around the level-1 prediction
    W, hop, R = 2048, 512, 150
    cen = np.arange(W // 2 + 4096, N - W // 2 - 4096, hop)
    tp = np.rint(np.interp(cen, n, tau)).astype(int)
    m, conf = xcorr_track(bP, bS, cen, W, R, tp)
    tk = smooth_knots(tp + m, 5, 2.0)
    tau = np.interp(n, cen, tk)
    return tau, tau0


# ------------------------------------------------------------- estimators
D0 = 2 * np.pi * IRE0 / FS


def fetch(s, tau, N, mode):
    """Source quantities read at n + tau(n) (dynamic offset, no resampling of
    RF).  mode 'int' = nearest sample, 'frac' = linear interpolation of the
    unit phasors / reliabilities."""
    n = np.arange(N)
    q = n + tau
    if mode == "int":
        k = np.clip(np.rint(q).astype(int), 1, N - 1)
        return s.ph[k], s.A2[k], s.rel[k], s.f[k]
    k0 = np.clip(np.floor(q).astype(int), 1, N - 2)
    a = q - k0
    ph = (1 - a) * s.ph[k0] + a * s.ph[k0 + 1]
    ph = ph / np.maximum(np.abs(ph), 1e-12)
    A2 = (1 - a) * s.A2[k0] + a * s.A2[k0 + 1]
    rel = (1 - a) * s.rel[k0] + a * s.rel[k0 + 1]
    f = np.mod(np.angle(ph), 2 * np.pi) * FS / (2 * np.pi)
    return ph, A2, rel, f


def local_rate(tau, w=256):
    g = np.gradient(tau)
    k = np.ones(w) / w
    return np.convolve(np.pad(g, w // 2, mode="edge"), k, "same")[w // 2: w // 2 + len(tau)]


def est_all(P, others, taus, N, mode, rate=None):
    """Returns dict name -> raw f estimate (Hz) on the primary's grid."""
    out = {}
    out["single_P"] = P.f.copy()
    items = [(P.ph, P.A2, P.rel, P.f, P.sig2, None)]
    for s, tau in zip(others, taus):
        ph, A2, rel, f, = fetch(s, tau, N, mode)
        if rate is not None:
            r = rate(tau)
            ph = ph * np.exp(1j * D0 * r)
            f = np.mod(np.angle(ph), 2 * np.pi) * FS / (2 * np.pi)
        items.append((ph, A2, rel, f, s.sig2, tau))
    # baseline: static inverse-variance weighted mean of f (Hz domain)
    w = np.array([1.0 / it[4] for it in items])
    F = np.stack([it[3] for it in items])
    out["mean_w"] = (w[:, None] * F).sum(0) / w.sum()
    if len(items) >= 3:
        out["median"] = np.median(F, axis=0)
    # joint: accumulate weighted unit phasors
    g = np.stack([it[1] * it[2] / it[4] for it in items])
    PH = np.stack([it[0] for it in items])
    Sx = (g * PH).sum(0)
    out["joint"] = np.mod(np.angle(Sx), 2 * np.pi) * FS / (2 * np.pi)
    # robust: soft down-weighting by residual vs consensus
    gw = g.copy()
    for _ in range(2):
        Sx = (gw * PH).sum(0)
        cons = Sx / np.maximum(np.abs(Sx), 1e-12)
        r = np.abs(np.angle(PH * np.conj(cons)[None, :]))
        sd = np.sqrt(np.stack([it[4] / np.maximum(it[1], 1e-9) for it in items])) * 1.5
        gw = g / (1.0 + (r / np.maximum(sd, 1e-6)) ** 2)
    Sx = (gw * PH).sum(0)
    out["joint_robust"] = np.mod(np.angle(Sx), 2 * np.pi) * FS / (2 * np.pi)
    return out


def score(f, truth_f, truth_lp, edge=8192):
    f = np.clip(f, 1.5e6, 30e6)
    e = lowpass(f, 4.2e6) - truth_lp
    sl = slice(edge, -edge)
    rms = np.sqrt(np.mean(e[sl] ** 2)) / HZ_IRE
    bad = (np.abs(f - truth_f)[sl] > 2e6)
    clicks = int(np.sum(bad[1:] & ~bad[:-1]))
    return rms, clicks


# ---------------------------------------------------------------- scenario
def build(cnrs, seed, ppm=20e-6, wob=(3e-6, 4e-6), jit=30e-9, N=2_000_000, jit_corr=320):
    rng = np.random.default_rng(seed)
    n_os = int((N / FS + 2 * PAD) * FSO) + 64
    msg = make_message(n_os, rng)
    phi = 2 * np.pi * np.cumsum(IRE0 + HZ_IRE * msg) / FSO
    tbs = []
    tbs.append(timebase(N, rng, 0.0, 0.0, wob[0], 29.97, rng.uniform(0, 6.28), jit, jit_corr))
    for i in range(1, len(cnrs)):
        tbs.append(timebase(N, rng, ppm * (1 if i % 2 else -1) * i, 37.4e-6 * i,
                            wob[1], 29.97, rng.uniform(0, 6.28), jit, jit_corr))
    caps, phs = [], []
    for tb, c in zip(tbs, cnrs):
        x, ph = capture(phi, tb, c, rng)
        caps.append(x)
        phs.append(ph)
    # truth in the primary's sample domain (per-sample phase increment)
    dtrue = np.empty(N)
    dtrue[1:] = phs[0][1:] - phs[0][:-1]
    dtrue[0] = dtrue[1]
    truth_f = dtrue * FS / (2 * np.pi)
    # true delay tau_i(n): k such that t_i[k] = t_0[n]
    taus_true = []
    for tb in tbs[1:]:
        k = np.interp(tbs[0], tb, np.arange(N))
        taus_true.append(k - np.arange(N))
    return caps, truth_f, taus_true


def main():
    cnrs = [float(v) for v in sys.argv[1].split(",")]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    t0 = time.time()
    jc = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    caps, truth_f, taus_true = build(cnrs, seed, jit_corr=jc)
    N = len(truth_f)
    truth_lp = lowpass(truth_f, 4.2e6)
    srcs = [prep(x) for x in caps]
    P = srcs[0]
    print(f"jitter corr {jc} smp   cnr dB: {cnrs}   seed {seed}   gen+prep {time.time()-t0:.1f}s")
    print("noise sigma^2 est:", [f"{s.sig2:.4f}" for s in srcs])

    taus_est = []
    for i, s in enumerate(srcs[1:]):
        te, tau0 = track_delay(P.base, s.base, N)
        taus_est.append(te)
        res = (te - taus_true[i])[8192:-8192]
        rng_t = taus_true[i]
        print(f"src{i+1}: true tau range {rng_t.min():.0f}..{rng_t.max():.0f} samples, "
              f"coarse {tau0}, residual RMS {np.sqrt(np.mean(res**2)):.3f} smp "
              f"(p99 {np.percentile(np.abs(res),99):.2f})")

    hdr = f"{'case':34s} {'estimator':14s} {'RMS IRE':>8s} {'clicks':>7s}"
    print(hdr)
    cases = [
        ("oracle tau, int, no ratecorr", taus_true, "int", None),
        ("oracle tau, frac, no ratecorr", taus_true, "frac", None),
        ("oracle tau, frac, ratecorr", taus_true, "frac", local_rate),
        ("est tau, int, no ratecorr", taus_est, "int", None),
        ("est tau, frac, ratecorr", taus_est, "frac", local_rate),
    ]
    for name, taus, mode, rate in cases:
        outs = est_all(P, srcs[1:], taus, N, mode, rate)
        for k, f in outs.items():
            rms, clk = score(f, truth_f, truth_lp)
            print(f"{name:34s} {k:14s} {rms:8.3f} {clk:7d}")
    for i, s in enumerate(srcs):
        rms, clk = score(s.f, truth_f, truth_lp) if i == 0 else (None, None)
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
