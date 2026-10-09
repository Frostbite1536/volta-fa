#!/usr/bin/env python3
"""Static sm_70 resource report for a .cu file, no GPU and no nvcc needed.

NVRTC compiles CUDA C++ to compute_70 PTX, then ptxas assembles it for sm_70
with -v and we print registers, spill bytes and shared memory per kernel.
Exit 1 if any kernel spills (INV-PERF-003) or uses > 96 KB static smem.

Usage: .venv/bin/python bin/sm70-report.py src/attention/foo.cu [-D NAME=VAL ...] [--keep-ptx DIR]
Templates are only compiled when instantiated: add explicit instantiations or
`extern "C"` wrappers in the file under test.
"""
import argparse
import ctypes
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
NV = next((ROOT / ".venv/lib").glob("python3*/site-packages/nvidia"), None)
if NV is None:
    sys.exit("no .venv with nvidia wheels; see docs/TOOLCHAIN.md")
PTXAS = NV / "cuda_nvcc/bin/ptxas"
INCLUDES = [NV / "cuda_runtime/include", NV / "cuda_nvcc/include", ROOT / "include"]
SMEM_LIMIT = 96 * 1024


def nvrtc_compile(src: pathlib.Path, defines: list[str]) -> str:
    lib = ctypes.CDLL(str(NV / "cuda_nvrtc/lib/libnvrtc.so.12"))
    prog = ctypes.c_void_p()
    code = src.read_bytes()
    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), code, str(src).encode(), 0, None, None)
    if rc:
        sys.exit(f"nvrtcCreateProgram failed: {rc}")
    opts = [b"--gpu-architecture=compute_70", b"-std=c++17", b"-lineinfo", b"-default-device"]
    opts += [f"-I{p}".encode() for p in INCLUDES]
    opts += [f"-D{d}".encode() for d in defines]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
    size = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    lib.nvrtcGetProgramLog(prog, log)
    if log.value.strip():
        print(log.value.decode(), file=sys.stderr)
    if rc:
        sys.exit(f"NVRTC compile failed (rc={rc})")
    lib.nvrtcGetPTXSize(prog, ctypes.byref(size))
    ptx = ctypes.create_string_buffer(size.value)
    lib.nvrtcGetPTX(prog, ptx)
    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return ptx.value.decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("-D", dest="defines", action="append", default=[])
    ap.add_argument("--keep-ptx", type=pathlib.Path)
    a = ap.parse_args()

    ptx = nvrtc_compile(a.src, a.defines)
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "k.ptx"
        p.write_text(ptx)
        if a.keep_ptx:
            a.keep_ptx.mkdir(parents=True, exist_ok=True)
            (a.keep_ptx / (a.src.stem + ".ptx")).write_text(ptx)
        r = subprocess.run([str(PTXAS), "-arch=sm_70", "-v", "-o", str(pathlib.Path(td) / "k.cubin"), str(p)],
                           capture_output=True, text=True)
    if r.returncode:
        print(r.stderr, file=sys.stderr)
        return 2

    bad = False
    kernel = None
    rows = []
    for line in r.stderr.splitlines():
        m = re.search(r"Compiling entry function '(\S+)'", line)
        if m:
            kernel = {"name": m.group(1), "regs": 0, "spill_st": 0, "spill_ld": 0, "smem": 0, "stack": 0}
            rows.append(kernel)
            continue
        if kernel is None:
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m:
            kernel["stack"], kernel["spill_st"], kernel["spill_ld"] = map(int, m.groups())
        m = re.search(r"Used (\d+) registers", line)
        if m:
            kernel["regs"] = int(m.group(1))
        m = re.search(r"(\d+) bytes smem", line)
        if m:
            kernel["smem"] = int(m.group(1))

    print(f"{'kernel':60} {'regs':>5} {'spill_st':>8} {'spill_ld':>8} {'stack':>6} {'smem':>7}")
    for k in rows:
        flag = ""
        if k["spill_st"] or k["spill_ld"]:
            flag, bad = " SPILL", True
        if k["smem"] > SMEM_LIMIT:
            flag, bad = flag + " SMEM>96K", True
        print(f"{k['name'][:60]:60} {k['regs']:>5} {k['spill_st']:>8} {k['spill_ld']:>8} {k['stack']:>6} {k['smem']:>7}{flag}")
    print("note: static smem only; dynamic smem is set at launch and must be budgeted in the design doc.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
