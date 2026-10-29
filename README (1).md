# GPGPU-Sim Scheduler Patch

For collaborators with GPGPU-Sim and SM90_H200 already installed. This package
provides a source patch, so rebuild using your own GCC/CUDA environment.

## Apply the patch

Required base commit: `c4cc73d3a1802f64b1283d3ae06ed1dcdb6e3bab`.
If your source differs, use a separate checkout of that commit rather than
forcing the patch. For example, from your existing repository:

```bash
git worktree add --detach ../gpgpu-sim-schedulers c4cc73d3a1802f64b1283d3ae06ed1dcdb6e3bab
```

Replace SHARE and the checkout path below with your actual paths:

```bash
SHARE=/absolute/path/to/research-share/scheduler
cd /path/to/gpgpu-sim-checkout
git rev-parse HEAD
python3 "$SHARE/verify_sources.py" base .
git apply --check "$SHARE/gpgpusim-schedulers.patch"
git apply "$SHARE/gpgpusim-schedulers.patch"
python3 "$SHARE/verify_sources.py" patched .

# First activate the CUDA/GCC environment used for your simulator build.
source setup_environment release
make -j4
source setup_environment release
export PTX_SIM_MODE_FUNC=0
```

Do not reapply the patch to an already patched checkout. Check its state with
`verify_sources.py patched .`. The base files were reconstructed by reversing
the local patch history; the verifier checks their hashes before application.
Compatibility with other source versions is not guaranteed.

## Select a scheduler

Files under `presets/` contain four policy options, not complete H200 hardware
configurations. Use the helper below to replace existing policy options without
creating duplicates:

```bash
python3 "$SHARE/make_run_config.py" \
  --base "$GPGPUSIM_ROOT/configs/tested-cfgs/SM90_H200" \
  --preset ldav_fetch --out /absolute/path/to/new-run
cd /absolute/path/to/new-run
/absolute/path/benchmark ... > simulation.log 2>&1
```

| Preset | Policy |
|---|---|
| lrr / gto | Existing baseline schedulers |
| stall | Runnable warps first, then warps without outstanding long-latency destinations, then GTO |
| dav_issue / dav_fetch | Divergence only, with baseline / priority fetch |
| lav_issue / lav_fetch | Locality only, with baseline / priority fetch |
| ldav_issue / ldav_fetch | Combined score, with baseline / priority fetch |
| lav_ema / ldav_ema | Issue-only ablations using the original per-warp EMA |

LAV/LDAV issue/fetch presets use the per-PC predictor. LDAV uses alpha=0.5;
DAV and LAV select the divergence-only and locality-only endpoints respectively.
For manual configuration, specify each option exactly once:

```text
-gpgpu_scheduler ldav
-gpgpu_ld_alpha 500
-gpgpu_locality_predictor 1
-gpgpu_priority_fetch 1
```

Predictor 0 selects per-warp EMA; 1 selects a 16-entry per-warp PC EMA table.
Fetch 0 retains round-robin fetch; 1 enables score-based fetch for dav/lav/ldav
only. The stall baseline is defined by this project and does not claim to
reproduce a particular published scheduler.

## Validation and limitations

The consolidated patch passed `git apply --check` and forward application in
an isolated temporary directory. All five resulting files matched the current
experimental source byte-for-byte using LF line endings.

The experimental source passed baseline regression checks and new-mode smoke
tests with CUDA 12.4.1, GCC 11.5.0, release mode, and H200. Its 165-run
BFS/SpMV/compute comparison passed correctness and repeat-determinism checks.
A fresh CUDA rebuild was not performed during packaging.

Performance gains are not guaranteed. Priority fetch refills only empty
two-slot instruction buffers; partial-buffer refill is not implemented.
Hardware overhead below 15% and performance at high SM utilization remain
unverified. The H200 perfect instruction-cache assumption is retained.
Host-side timing instrumentation does not measure GPU hardware cost.
See IMPLEMENTATION.md for the implementation details.

MANIFEST.json records base and patched source hashes. The original GPGPU-Sim
distribution is not included, and this patch does not change its copyright or
license notices. The Korean version is available in [README.ko.md](README.ko.md).
