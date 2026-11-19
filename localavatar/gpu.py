"""GPU memory sampling via nvidia-smi, plus the optional GPU-busy gate.

Adapted from the gpu module of kodros-localclip.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time


def reserved_names() -> list[str]:
    """Process-name fragments from LOCALAVATAR_GPU_BLOCK (comma separated, case-insensitive). Empty by default."""
    raw = os.environ.get("LOCALAVATAR_GPU_BLOCK", "")
    return [n.strip().lower() for n in raw.split(",") if n.strip()]


def gpu_reserved_running() -> bool:
    """True when a process whose command line contains a listed fragment is running.

    With LOCALAVATAR_GPU_BLOCK unset or empty this is always False and /proc is not scanned.
    """
    names = reserved_names()
    if not names:
        return False
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode(errors="ignore").lower()
        except OSError:
            continue
        if any(n in cmd for n in names):
            return True
    return False


# Runs in a child process so a failed or leaked context never touches this one.
_CUDA_PROBE = """
import ctypes, sys
try:
    lib = ctypes.CDLL("libcuda.so.1")
except OSError:
    sys.exit(0)                      # no CUDA driver: torch will use the CPU on its own
if lib.cuInit(0) != 0:
    sys.exit(0)
n = ctypes.c_int()
if lib.cuDeviceGetCount(ctypes.byref(n)) != 0 or n.value == 0:
    sys.exit(0)
dev = ctypes.c_int()
lib.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
rc = lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
if rc == 0:
    lib.cuCtxDestroy_v2(ctx)
print(rc)
"""


def cuda_context_error(timeout: float = 30.0) -> str | None:
    """Return a human-readable reason if a GPU exists but a new CUDA context cannot be created.

    Seen 2026-10-29: the shared RTX 4080 refused every new context (driver code 210, kernel log
    "GspRmAlloc failed ... NV_ERR_NO_MEMORY") while 6 GB showed as free. SadTalker then died
    minutes in with an opaque torch trace; this check fails fast instead.
    """
    import sys

    try:
        r = subprocess.run([sys.executable, "-c", _CUDA_PROBE], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"CUDA probe did not finish ({type(e).__name__})"
    out = r.stdout.strip()
    if r.returncode != 0 or not out or out == "0":
        return None if r.returncode == 0 else f"CUDA probe crashed (exit {r.returncode})"
    return (f"the GPU refused to create a CUDA context (driver error {out}); other processes are probably holding "
            "its context memory. Free GPU programs or run on the CPU (--cpu, very slow)")


def _query(args: list[str]) -> list[list[str]]:
    try:
        r = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []
    return [[c.strip() for c in line.split(",")] for line in r.stdout.strip().splitlines() if line.strip()]


def free_mib() -> int | None:
    rows = _query(["--query-gpu=memory.free"])
    return int(rows[0][0]) if rows and rows[0][0].isdigit() else None


def _descendants(root: int) -> set[int]:
    kids: dict[int, list[int]] = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(pid))
    out, stack = {root}, [root]
    while stack:
        for c in kids.get(stack.pop(), []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


class GPUMonitor:
    """Samples whole-GPU memory and this process tree's GPU memory twice a second."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.available = bool(_query(["--query-gpu=name"]))

    def _run(self) -> None:
        me = os.getpid()
        while not self._stop.is_set():
            g = _query(["--query-gpu=memory.used,memory.total,utilization.gpu"])
            apps = _query(["--query-compute-apps=pid,used_memory"])
            tree = _descendants(me)
            own = sum(int(m) for p, m in apps if p.isdigit() and int(p) in tree and m.isdigit())
            if g and all(x.isdigit() for x in g[0]):
                used, total, util = (int(x) for x in g[0])
                self.samples.append({"t": time.time(), "used_mib": used, "total_mib": total,
                                     "util_pct": util, "own_mib": own})
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.available:
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.available:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        if not self.samples:
            return {"available": self.available, "samples": 0}
        s = self.samples
        return {
            "available": True,
            "samples": len(s),
            "total_mib": s[0]["total_mib"],
            "baseline_used_mib": s[0]["used_mib"],
            "peak_used_mib": max(x["used_mib"] for x in s),
            "peak_localavatar_mib": max(x["own_mib"] for x in s),
            "peak_util_pct": max(x["util_pct"] for x in s),
            "note": "used/util are whole-GPU (shared); localavatar_mib is this process tree only (nvidia-smi, 0.5 s samples)",
        }
