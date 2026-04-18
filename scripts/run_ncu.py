"""One-shot Nsight Compute (ncu) attach for kernel-level profiling.

Two modes:
  --roofline  captures the SpeedOfLight + RooflineChart + Compute/Memory workload
              sections needed to plot a roofline (FLOPs, bytes, arithmetic intensity)
  default     captures Compute/Memory workload sections for cache-thrashing and
              occupancy diagnostics

ncu is expensive (serialized kernel replay), so bound the capture with --kernel-count
and either a --pid to attach to (e.g. the running vLLM worker) or a --launch command.

Examples:
  python run_ncu.py --pid 12345 --kernel-count 50 --roofline \
      --out /tmp/qwen_roofline.ncu-rep
  python run_ncu.py --launch "python bench.py" --kernel-count 100
"""
import argparse
import os
import subprocess
import sys


DEFAULT_SECTIONS = "ComputeWorkloadAnalysis,MemoryWorkloadAnalysis"
ROOFLINE_SECTIONS = "SpeedOfLight,SpeedOfLight_RooflineChart,ComputeWorkloadAnalysis,MemoryWorkloadAnalysis"


def build_cmd(args):
    cmd = ["ncu", "--target-processes", "all", "--replay-mode", "application"]
    sections = ROOFLINE_SECTIONS if args.roofline else DEFAULT_SECTIONS
    for s in sections.split(","):
        cmd += ["--section", s.strip()]
    if args.kernel_count:
        cmd += ["--launch-count", str(args.kernel_count)]
    if args.kernel_regex:
        cmd += ["--kernel-name", f"regex:{args.kernel_regex}"]
    if args.out:
        cmd += ["-o", args.out, "--force-overwrite"]
    else:
        cmd += ["--csv"]
    if args.pid:
        cmd += ["--attach", str(args.pid)]
        return cmd
    if args.launch:
        cmd += ["--"] + args.launch.split()
        return cmd
    raise SystemExit("Must pass either --pid or --launch")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, help="PID of the running vLLM worker to attach to")
    ap.add_argument("--launch", help="Command to launch and profile (alternative to --pid)")
    ap.add_argument("--kernel-count", type=int, default=50, help="Max kernels to capture (default: 50)")
    ap.add_argument("--kernel-regex", help="Only profile kernels matching this regex (e.g. 'flash_attn|gemm')")
    ap.add_argument("--roofline", action="store_true", help="Capture roofline-chart inputs (FLOPs, bytes, AI)")
    ap.add_argument("--out", help="Output .ncu-rep path (omit for CSV to stdout)")
    args = ap.parse_args()
    cmd = build_cmd(args)
    print("[run_ncu] " + " ".join(cmd), file=sys.stderr, flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
