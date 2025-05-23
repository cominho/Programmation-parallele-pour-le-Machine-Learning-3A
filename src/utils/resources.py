import os, platform
try:
    import psutil
except ImportError:
    psutil = None

def configure_resources():
    """Pin process to cores 0–3 and seed RNG."""
    import numpy as np; np.random.seed(42)
    sys = platform.system()
    if sys == 'Linux':
        os.sched_setaffinity(0, {0,1,2,3})
    elif sys == 'Windows' and psutil:
        psutil.Process().cpu_affinity([0,1,2,3])
    # macOS limits can be added here
