# GPGPU-Sim Schedulers and BFS/SpMV Benchmarks

This package is for collaborators who already have GPGPU-Sim and an H200
configuration installed. Share the entire folder. Applying the scheduler patch
requires rebuilding the simulator. The BFS and SpMV CUDA sources are copied
unchanged from the original experiments. Raw datasets and private experiment
logs are not included.

## 1. Apply and build the scheduler patch

Follow [scheduler/README.md](scheduler/README.md) to check the required commit
and source hashes, apply `gpgpusim-schedulers.patch`, and rebuild. Do not apply
the patch again to an already patched checkout. Use a separate checkout if your
source version differs.

## 2. Prepare the inputs and build the benchmark

Activate the same CUDA environment used to build your simulator, then run:

```bash
cd benchmarks
bash download_data.sh
nvcc --cudart shared \
  -gencode arch=compute_75,code=compute_75 \
  src/real-data/real_bench.cu -o real_bench
```

- **BFS:** SNAP ca-GrQc collaboration graph; 5,242 vertices and 28,980 stored
  directed edges after preprocessing. The highest-degree vertex becomes root 0.
- **SpMV:** NIST bcsstk17 structural engineering matrix; 10,974 by 10,974,
  with 428,650 CSR entries after symmetric expansion.
- Sources, transformations, and SHA256 hashes are recorded in
  `benchmarks/data/manifest.json`.
- `.txt.gz` contains a compressed graph edge list. `.mtx.gz` contains a compressed
  Matrix Market matrix. The preparation script converts both inputs to PCSR1 CSR.

## 3. Run with your existing H200 configuration

Activate the patched GPGPU-Sim environment, then return to the benchmark folder.
On a cluster, execute the benchmark inside a CPU Slurm job under your own account.

```bash
cd /path/to/gpgpu-sim_distribution
source setup_environment release
export PTX_SIM_MODE_FUNC=0
cd /path/to/research-share/benchmarks
ldd ./real_bench | grep libcudart
# Confirm that libcudart resolves to the patched GPGPU-Sim library.
bash run.sh gto
bash run.sh ldav_issue
bash run.sh ldav_fetch
```

Each command runs four cases: BFS and SpMV, each with stride 1 and stride 32.
Every case must pass its CPU-reference checks. Logs are written under
`benchmarks/runs/PRESET/`. Existing run directories are not overwritten.

The nine main presets are `lrr`, `gto`, `stall`, `dav_issue`, `dav_fetch`,
`lav_issue`, `lav_fetch`, `ldav_issue`, and `ldav_fetch`. Additional issue-only
ablations using the original per-warp EMA are `lav_ema` and `ldav_ema`.
To use another configuration location or a fresh output directory:

```bash
H200_CONFIG_DIR=/path/to/H200-config OUTPUT_DIR=runs-second bash run.sh ldav_fetch
```

LDAV uses alpha=0.5. DAV uses divergence only; LAV uses locality only.
LAV/LDAV issue/fetch presets use the per-PC predictor. Keep the input,
compiler/PTX target, and H200 hardware configuration identical across comparisons.
Use a different OUTPUT_DIR for each repetition. Preserve source, configuration,
runtime, and data hashes with your results.

## Algorithms and correctness checks

**BFS** uses destination-owned pull traversal. Each level reads incoming CSR
and the previous frontier, then writes the next frontier and distances in a
separate kernel launch. Every vertex distance is checked against an independent
CPU queue traversal over outgoing edges. The atomic vertex-push variant is not
included.

**SpMV** assigns one CSR row to each CUDA thread. FP32 GPU results are checked
against a CPU double-precision reference using a per-row forward-error bound.
Successful runs print `REAL_PASS workload=bfs` or `REAL_PASS workload=spmv`.
Reference checksums must match between layouts. Stride 32 preserves the logical
operation but changes both coalescing and memory footprint; it is not a pure
cache-capacity control.

The original integrated source passed correctness, hardware-consistency, and
repeat-determinism checks across 165 H200 comparison runs. The two CUDA source
files and preparation script in this package are unchanged copies, verified by
SHA256. A fresh CUDA rebuild was not performed during packaging.

Partial instruction-buffer refill is not implemented. Hardware overhead below
15% and performance at high SM utilization remain unverified. Performance gains
are not guaranteed. The experimental H200 configuration assumes a perfect
instruction/constant cache.

Check package integrity from this folder with `sha256sum -c SHA256SUMS`.
Follow the original distribution terms for GPGPU-Sim and each public dataset.
The Korean version is available in [README.ko.md](README.ko.md).
