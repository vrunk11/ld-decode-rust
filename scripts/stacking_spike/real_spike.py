"""Idea 1 on a REAL RF capture: warp the real signal with a different
time-base per source, add independent noise to each, stack, and compare with the
noise-free warped original (reference).  The capture's own noise is common to
all copies, so only the *added* noise can be removed by stacking.

usage: real_spike.py <file.s16> <add_cnr_db[,..]> <jitter_corr> [seed] [start_sample]
"""
import sys
import numpy as np
from stack_spike import *

UP = 16


def upsample(x):
    n = len(x)
    X = np.fft.rfft(x)
    Y = np.zeros(n * UP // 2 + 1, complex)
    Y[: len(X)] = X
    return np.fft.irfft(Y, n * UP) * UP


def sample_at(xu, t_idx):
    """cubic-free: linear interpolation on the 16x grid (error ~ -60 dB)."""
    return np.interp(t_idx * UP, np.arange(len(xu)), xu)


def main():
    path = sys.argv[1]
    add = [float(v) for v in sys.argv[2].split(",")]
    jc = int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    start = int(sys.argv[5]) if len(sys.argv) > 5 else 1_000_000
    N = 2_000_000
    M = 8000
    raw = np.fromfile(path, dtype="<i2", count=N + 2 * M, offset=2 * start).astype(np.float64)
    raw -= raw.mean()
    xu = upsample(raw)
    rng = np.random.default_rng(seed)
    nsrc = len(add)
    # time bases in *sample units of the chunk* (offset M keeps indices inside)
    tbs = [timebase(N, rng, 0.0, 0.0, 3e-6, 29.97, rng.uniform(0, 6.28), 30e-9, jc)]
    for i in range(1, nsrc):
        tbs.append(timebase(N, rng, 20e-6 * (1 if i % 2 else -1) * i, 37.4e-6 * i,
                            4e-6, 29.97, rng.uniform(0, 6.28), 30e-9, jc))
    clean = [sample_at(xu, tb * FS + M) for tb in tbs]
    # signal power in the carrier band -> noise level for the requested added CNR
    z0, _ = analytic(clean[0])
    psig = np.mean(np.abs(z0) ** 2) / 4.0 * 2.0   # real-signal band power: |z|^2/2 (z has gain 2)
    psig = np.mean(np.abs(z0) ** 2) / 2.0
    caps = []
    for c, cn in zip(clean, add):
        sig2 = psig / (0.5 * 10 ** (cn / 10))
        caps.append(c + rng.normal(size=N) * np.sqrt(sig2))
    ref = prep(clean[0])
    truth_f = ref.f
    truth_lp = lowpass(np.clip(truth_f, 1.5e6, 30e6), 4.2e6)
    srcs = [prep(x) for x in caps]
    P = srcs[0]
    taus_true = []
    for tb in tbs[1:]:
        k = np.interp(tbs[0], tb, np.arange(N))
        taus_true.append(k - np.arange(N))
    print(f"file {path}  start {start}  added-noise CNR {add} dB  jitter-corr {jc}  seed {seed}")
    print("noise sigma^2 est (real+added):", [f"{s.sig2:.4g}" for s in srcs], " reference's own:", f"{ref.sig2:.4g}")
    te = []
    for i, s in enumerate(srcs[1:]):
        t, t0 = track_delay(P.base, s.base, N)
        te.append(t)
        r = (t - taus_true[i])[8192:-8192]
        print(f"src{i+1}: tau range {taus_true[i][8192:-8192].min():.0f}..{taus_true[i][8192:-8192].max():.0f}"
              f"  tracker residual RMS {np.sqrt(np.mean(r**2)):.3f} smp  p99 {np.percentile(np.abs(r),99):.2f}")
    names = ["single_P", "mean_w", "joint", "joint_robust"]
    rs, _ = score(ref.f, truth_f, truth_lp)
    print(f"reference-vs-itself check {rs:.3f}; error vs noise-free original, RMS IRE / clicks")
    print(f"{'alignment':22s}" + "".join(f"{k:>14s}" for k in names))
    for label, taus, mode, rate in [("oracle tau, frac", taus_true, "frac", local_rate),
                                    ("oracle tau, int", taus_true, "int", None),
                                    ("tracker, frac", te, "frac", local_rate)]:
        o = est_all(P, srcs[1:], taus, N, mode, rate)
        row = ""
        for k in names:
            r_, c_ = score(o[k], truth_f, truth_lp)
            row += f"{r_:8.3f}/{c_:<5d}"
        print(f"{label:22s}" + row)


if __name__ == "__main__":
    main()
