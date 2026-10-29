# Proposal continuation

Environment remains CUDA 12.4.1, GCC 11.5.0, release, performance simulation,
and the SM90_H200 hardware configuration. PTX is compute_75.

Readiness instrumentation observes buffer occupancy, pending instruction-cache
misses, raw barrier/async/atomic flags, head-PC mismatch, scoreboard collision,
and the first live warp in the actual scheduler ordering. Empty top buffers
are split by whether another warp issued that cycle. These counters overlap;
raw flags do not prove readiness. Pipeline/dispatch stalls use the existing
ready_inst and issued_inst decision, with no second waiting() invocation.

The bounded PC predictor keeps 16 entries per warp. Each entry tracks a global
load/store instruction PC and its EMA sharing score. A first observation
replaces the neutral 0.5 prior; later observations use beta 0.5. Round-robin
replacement and warp-reset clearing bound storage and prevent warp-slot
history leakage. The currently buffered memory instruction selects the PC;
otherwise the last observed memory PC supplies the score. No future address
evaluation occurs. Both predictors are evaluated on the same observed samples,
including cold starts. PC warm-coverage is reported separately.

Priority fetch is an optional selection policy over native refill candidates.
The native decoder writes a two-instruction bundle, so admission still requires
an empty buffer. This is a limitation relative to the proposal's literal
"buffer not full" condition: partial refill is not implemented. Existing
wrong-path flushing and pending-cache behavior remain intact. Fetch ties use
the last issued warp on that SM, then oldest dynamic warp ID, then slot order;
issue ties retain scheduler-local GTO.

The supplied SM90_H200 configuration sets perfect_inst_const_cache=1.
Consequently pending instruction-cache misses are zero by construction in
this main comparison. These runs evaluate fetch arbitration under a perfect
instruction/constant-cache assumption, not instruction-cache miss reduction.

The project-defined stall-aware baseline orders runnable warps before waiting
warps, then favors warps with no outstanding long-latency destination registers,
then applies GTO. It does not claim to reproduce a published stall-aware
scheduler. Pipeline availability is checked by the normal issue loop.

The nine-policy matrix is LRR/GTO/stall plus DAV/LAV/LDAV each with issue-only
or priority fetch. LAV/LDAV use the PC predictor. Two extra issue-only EMA
ablations isolate the predictor change. Alpha stays 0.5; no workload-specific
tuning. Every configuration runs three times on real BFS and SpMV in native
and padded layouts, plus a 16384-thread compute control: 165 runs total.

Host timing is separate from simulated GPU cycles. No hardware area/timing
claim is established by software timing. The per-warp PC table adds storage
and comparison logic whose <=15% hardware-overhead goal remains unverified.
Simple DRAM controller counters remain unavailable. The known BFS L1-bypass
assertion is not suppressed or repeated in this main comparison.

## Configuration switches

For a fresh run directory copied from SM90_H200, select each setting once:

```text
-gpgpu_scheduler ldav
-gpgpu_ld_alpha 500
-gpgpu_locality_predictor 1
-gpgpu_priority_fetch 1
```

Predictor 0 selects the original per-warp EMA; 1 selects bounded PC history.
Fetch 0 retains round-robin; 1 enables score-based bundle fetch. The fetch
switch supports dav/lav/ldav only; lrr/gto/stall use fetch 0. Avoid duplicate
settings in the configuration file. The experiment runner writes these values
and refuses to overwrite any existing simulation log.

Remote experiment artifacts live in ~/experiments/proposal-nine/:
source/changes.patch, source/source.tar.gz, SHA256SUMS, runtime/, hardware/,
data/, runs/, report.md, validation-report.md, metrics.csv, summary.csv,
detailed-metrics.json, and ablation.json. Reports are produced only after all
165 runs pass validation. The original experiments remain in their own folders.

For this distribution, apply the single consolidated patch as described in README.md.
