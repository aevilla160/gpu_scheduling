# CCWS-style score-based warp throttling on H200 (GPGPU-Sim 4.2.0)

Wenqi Liao. Issue-scheduler changes only; L1 and every other H200 setting unchanged (configs/tested-cfgs/SM90_H200).

## What it does
Each warp keeps a lost-locality score (as in CCWS, Rogers et al., MICRO 2012):
- Each warp remembers the cache lines it touched recently (256-entry table).
- An L1 miss on a line the warp touched before = its data was evicted by another warp. Score += LLS_INC.
- Every cycle each score drops by (score >> LLS_DECAY) + 1, so held-back warps are not starved.
- Per scheduler, warps are taken oldest first; each weighs 1024 + score; warps are admitted while the
  total stays <= LLS_K x 1024 (LLS_K default 8 = all 8 warps per scheduler on H200). The rest do not issue
  this cycle (same gate as static warp limiting).

Enabled with environment variables at run time (no config change):
    LLS_MODE=6 LLS_INC=448 LLS_DECAY=12      # final setting
with the run config made by make_run_config.py --preset lav_issue.

## Patches (patches/)
Applied in this order on top of our gpgpu-sim-schedulers tree (commit c4cc73d3 plus the LAV scheduler work).
Only 06-08 are the CCWS work; 01-05 add the hooks it builds on (warp limit, lost-locality tracking, DAWS gate).
    01 lav_limit       LAV_WARP_LIMIT static limit inside the LAV scheduler
    02 lost_locality   per-warp lost-locality table and score (LLS_MODE 1, 2)
    03 lls_feedback    window feedback controller (LLS_MODE 3)
    04 daws            DAWS footprint gate (LLS_MODE 4)
    05 daws2           DAWS v2 (LLS_MODE 5)
    06 ccws2           CCWS cutoff, age-ordered admission (LLS_MODE 6)
    07 ccws2b          block all issue for warps past the cutoff (LLS_GATE=1 restores load-only gate)
    08 ccws2c          time decay of scores (LLS_DECAY)
Easiest: use ccws-full.diff (the full `git diff` of Wenqi's tree) instead of the separate patches:
    cd gpgpu-sim-schedulers && git checkout c4cc73d3 && git apply ccws-full.diff
    source setup_environment release && make -j4

## Inputs (inputs/)
    python3 gen_long_rows.py  longrow82.csr 131072 82    # contended: 82 random nnz per row
    python3 gen_mixed_rows.py mixed.csr                  # not contended: 25% long (200 nnz) / 75% short (8 nnz) warps
Put them in benchmarks/data/prepared/. real_bench, 512-thread blocks.

## Results (cycles, H200)
| Setting                    | longrow82          | mixed            |
|----------------------------|--------------------|------------------|
| GTO                        | 224,657            | 122,406          |
| Best static limit (N=5)    | 160,082 (-28.7%)   | no limit is best |
| Load-only gate             | 351,467 (slower)   | 332,391 (slower) |
| Full gate, no decay        | 166,223 (-26.0%)   | 246,590 (+101%, starvation) |
| Decay 12, INC 256          | 191,221 (-14.9%)   | 122,406 (0.0%)   |
| Decay 12, INC 384          | 181,421 (-19.2%)   | 121,871 (-0.4%)  |
| Decay 12, INC 448 (final)  | 178,105 (-20.7%)   | 121,156 (-1.0%)  |
| Decay 12, INC 512          | 173,650 (-22.7%)   | 131,217 (+7.2%)  |

bcsstk17 (spmv_bench, SPMV_BLOCK=512): GTO 58,747, CCWS INC 448 58,882 (+0.2%).
bcsstk17 b1024: GTO 74,078, best static N=6 67,995 (-8.2%), CCWS 73,628 (-0.6%).
bcsstk17 has little temporal reuse (7.7% of reuse), so the score rarely triggers.

## Example run (from research-share-large/benchmarks)
    B="$HOME/gpu-project/gpgpu-sim_distribution/configs/tested-cfgs/SM90_H200"
    D=runs-longrow82-ccws/x/spmv-longrow82-s1
    python3 ../scheduler/make_run_config.py --base "$B" --preset lav_issue --out $D
    (cd $D && env LLS_MODE=6 LLS_INC=448 LLS_DECAY=12 ../../../real_bench spmv ../../../data/prepared/longrow82.csr 1 > simulation.log 2>&1)
    grep '^gpu_tot_sim_cycle' $D/simulation.log | tail -1

Full write-up: https://claude.ai/artifact/8fKQiY3X9fyBvz7dJdm3An
