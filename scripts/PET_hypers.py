#!/usr/bin/env python3
import os, sys, subprocess, itertools, shlex
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = sys.executable                     # current Python
PET = os.path.join(os.path.dirname(__file__), "PET.py")
OUT_DIR = "results/PET"

# ----- Define your grid here -----
grid = dict(
    # tune DC first; you can add 'nll' or 'mlem' if you want
    img_size    = [256],
    epochs      = [10001],               
    lr          = [0.005],
    dose        = [5.0],
    bin_mash    = [2],
    ring_factor = [4.0],
    device      = ["cuda"],             
    seed        = [0],
    poisson_seed= [0],
    )

# ----- Util: reproduce PET.py's output path to skip finished runs -----
def expected_out_file(args):
    # PET.py writes to: out_dir = args.output_dir or output_dir/save_tag
    # filename includes (dose, scheduled, seed, poisson_seed, BINMASH, IMGSIZE)
    out_dir = os.path.join(OUT_DIR, args["save_tag"]) if args.get("save_tag") else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    fname = f"dose={args['dose']}_scheduled={args['use_scheduler']}_seed={args['seed']}_pseed={args['poisson_seed']}_BINMASH={args['bin_mash']}_IMGSIZE={args['img_size']}.pkl"
    return os.path.join(out_dir, fname)

def mk_save_tag(a):
    # Encode everything that affects the run (so folders are unique & readable)
    # Keep it short + filesystem-friendly
    return ("losses={losses}_lr={lr}_ep={epochs}_rf={ring_factor}_dev={device}"
            .format(**a)).replace("/", "_")

def to_cmd(a):
    cmd = [
        PY, PET,
        "--img_size", str(a["img_size"]),
        "--epochs", str(a["epochs"]),
        "--lr", str(a["lr"]),
        "--output_dir", OUT_DIR,
        "--device", a["device"],
        "--dose", str(a["dose"]),
        "--bin_mash", str(a["bin_mash"]),
        "--ring_factor", str(a["ring_factor"]),
        "--seed", str(a["seed"]),
        "--poisson_seed", str(a["poisson_seed"]),
        "--losses", a["losses"],
        "--save_tag", a["save_tag"],
    ]
    if a["use_scheduler"]:
        cmd.append("--use_scheduler")
    return cmd

def run_once(a):
    out_file = expected_out_file(a)
    if os.path.exists(out_file):
        return f"SKIP  {out_file}"
    cmd = to_cmd(a)
    # For clean logs per run:
    log_dir = os.path.join(OUT_DIR, a["save_tag"])
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "stdout.txt")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        return f"FAIL  {' '.join(shlex.quote(c) for c in cmd)} (see {log_path})"
    return f"DONE  {out_file}"

def main():
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    jobs = []
    for combo in itertools.product(*vals):
        a = dict(zip(keys, combo))
        a["save_tag"] = mk_save_tag(a)
        jobs.append(a)

    # ----- adjust parallelism -----
    MAX_JOBS = 1 if any(a["device"] == "cuda" for a in jobs) else os.cpu_count() or 2

    print(f"Prepared {len(jobs)} runs; max concurrent = {MAX_JOBS}")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_JOBS) as ex:
        futs = {ex.submit(run_once, a): a for a in jobs}
        for fut in as_completed(futs):
            msg = fut.result()
            print(msg)
            results.append(msg)

    print("\nSummary:")
    for r in results:
        print(" -", r)

if __name__ == "__main__":
    main()
