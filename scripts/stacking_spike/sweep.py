import sys, numpy as np
from stack_spike import *
cnrs=[float(v) for v in sys.argv[1].split(",")]; jc=int(sys.argv[2]); seed=int(sys.argv[3]) if len(sys.argv)>3 else 1
caps, truth_f, taus_true = build(cnrs, seed, jit_corr=jc)
N=len(truth_f); n=np.arange(N)
truth_lp=lowpass(truth_f,4.2e6)
srcs=[prep(x) for x in caps]; P=srcs[0]
print(f"cnr {cnrs} jitter-corr {jc} seed {seed}")
rng=np.random.default_rng(5)
def smooth_err(sig):
    if sig==0: return np.zeros(N)
    w=rng.normal(size=N+4000); k=np.exp(-0.5*(np.arange(-1000,1001)/500.0)**2); k/=np.sqrt((k**2).sum())
    return sig*np.convolve(w,k,'valid')[:N]
names=["single_P","mean_w","joint","joint_robust"]+(["median"] if len(cnrs)>=3 else [])
print(f"{'align err (smp RMS)':22s}"+"".join(f"{k:>14s}" for k in names)+"   (RMS IRE / clicks)")
def line(label,taus,mode="frac",rate=local_rate):
    o=est_all(P,srcs[1:],taus,N,mode,rate)
    row=""
    for k in names:
        r,c=score(o[k],truth_f,truth_lp); row+=f"{r:8.2f}/{c:<5d}"
    print(f"{label:22s}"+row)
for sig in (0,0.1,0.25,0.5,1.0,1.5):
    taus=[t+smooth_err(sig) for t in taus_true]
    line(f"oracle+err {sig}",taus)
line("oracle, int mode",taus_true,"int")
line("oracle, frac, no rate",taus_true,"frac",None)
te=[]
for s in srcs[1:]:
    t,_=track_delay(P.base,s.base,N); te.append(t)
res=[np.sqrt(np.mean((t-tt)[8192:-8192]**2)) for t,tt in zip(te,taus_true)]
line(f"tracker (resid {np.mean(res):.2f})",te)
