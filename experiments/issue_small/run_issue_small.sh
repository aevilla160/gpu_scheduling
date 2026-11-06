#!/usr/bin/env bash
# Issue-only scheduler comparison on a small synthetic input with a chosen block size.
# Commands execute only when this script is run on the Linux cluster.
# The shared benchmark source is copied, never edited; fetch stays round-robin.
#   INPUT_SIZE=1024 BLOCK=192 bash run_issue_small.sh      # 6 warps per block (default)
#   BLOCK=768 bash run_issue_small.sh                # 6 warps per issue scheduler
#   SIM_OPTS="-gpgpu_unified_l1d_size 32" RUN_TAG=l1kb32 BLOCK=768 bash run_issue_small.sh   # 32 KB L1, not the stock preset
set -eo pipefail
SIMROOT="${SIMROOT:-/home/UFAD/alex.villa/gpgpu-sim-divergence}"
SCHEDULER_DIR="${SCHEDULER_DIR:-/home/UFAD/alex.villa/research-share/scheduler}"
BENCH_SRC="${BENCH_SRC:-/home/UFAD/alex.villa/research-share-large/benchmarks/src/proposal/proposal_bench.cu}"
OUT_PARENT="${OUT_PARENT:-/home/UFAD/alex.villa/gpgpu-sim_distribution/experiments/regression}"
INPUT_SIZE="${INPUT_SIZE:-1024}"
BLOCK="${BLOCK:-192}"
SIM_OPTS="${SIM_OPTS:-}"       # simulator options appended to every run's config; a later option wins
LONG_ROW="${LONG_ROW:-}"       # length given to the 48-entry SpMV rows in the benchmark copy
RUN_TAG="${RUN_TAG:-small}"    # names the result directory; must be set with SIM_OPTS or LONG_ROW
read -r -a POLICIES <<< "${PRESETS:-lrr gto dav_issue lav_issue ldav_issue}"
read -r -a WORKS <<< "${WORKLOADS:-spmv bfs}"
(( BLOCK % 32 == 0 && BLOCK >= 32 && BLOCK <= 1024 )) || { echo 'BLOCK must be a multiple of 32 in 32..1024' >&2; exit 1; }
[[ "$INPUT_SIZE" =~ ^[0-9]+$ ]] && (( INPUT_SIZE >= 8 && INPUT_SIZE <= 1048576 )) || { echo "INPUT_SIZE must be an integer in 8..1048576" >&2; exit 1; }
[[ -z "$SIM_OPTS$LONG_ROW" || "$RUN_TAG" != small ]] || { echo 'Set RUN_TAG with SIM_OPTS or LONG_ROW so results stay separate from the stock runs' >&2; exit 1; }
[[ -z "$LONG_ROW" || "$LONG_ROW" =~ ^[0-9]+$ ]] || { echo 'LONG_ROW must be a non-negative integer' >&2; exit 1; }

source "${CONDA_SH:-/home/UFAD/alex.villa/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-gpgpusim}"
export CUDA_INSTALL_PATH="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 "$SCHEDULER_DIR/verify_sources.py" patched "$SIMROOT"
cd "$SIMROOT"
source ./setup_environment release
SIMLIB="$GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG"
export LD_LIBRARY_PATH="$SIMLIB:$LD_LIBRARY_PATH"
export CUOBJDUMP_SIM_FILE=jj PTX_SIM_MODE_FUNC=0
unset PTX_SIM_USE_PTX_FILE PTX_SIM_KERNELFILE

mkdir -p "$OUT_PARENT"
OUT=$(mktemp -d "$OUT_PARENT/issue_${RUN_TAG}_n${INPUT_SIZE}_b${BLOCK}_$(date +%Y%m%d_%H%M%S)_XXXX")
echo "Results: $OUT"
printf 'SIM_OPTS=%s\nLONG_ROW=%s\n' "$SIM_OPTS" "$LONG_ROW" > "$OUT/sim_opts.txt"

# Same kernels and generators; only the launch geometry changes from 512 threads per block.
SRC="$OUT/proposal_bench_b$BLOCK.cu"
APP="$OUT/proposal_bench_b$BLOCK.sim"
# The copy also lifts the generator's 65536 size cap so inputs can exceed one wave of blocks.
sed -e "s#+511)/512,512>>>#+$((BLOCK-1)))/$BLOCK,$BLOCK>>>#g" -e 's#size>65536#size>1048576#' \
  ${LONG_ROW:+-e "s#r%29==0 ? 48#r%29==0 ? $LONG_ROW#"} "$BENCH_SRC" > "$SRC"
[[ -z "$LONG_ROW" ]] || grep -q "r%29==0 ? $LONG_ROW :" "$SRC" || { echo 'Long-row rewrite did not match the generator' >&2; exit 1; }
[[ $(grep -c '>>>' "$SRC") == 3 && $(grep -c ",$BLOCK>>>" "$SRC") == 3 ]] || { echo 'Launch geometry rewrite did not match the three expected kernels' >&2; exit 1; }
grep -q 'size>1048576' "$SRC" || { echo 'Size cap rewrite did not match the benchmark' >&2; exit 1; }
"$CUDA_INSTALL_PATH/bin/nvcc" -O3 -std=c++17 --cudart=shared \
  -gencode arch=compute_75,code=compute_75 \
  -Xlinker=--enable-new-dtags -Xlinker=-rpath -Xlinker="$SIMLIB:$CUDA_INSTALL_PATH/lib" \
  "$SRC" -o "$APP"
resolved=$(ldd "$APP" | awk '/libcudart/{print $3; exit}')
[[ -n "$resolved" && "$(readlink -f "$resolved")" == "$(readlink -f "$SIMLIB/libcudart.so")" ]] || { echo "Benchmark resolves libcudart to '$resolved', not the simulator build" >&2; exit 1; }
sha256sum "$BENCH_SRC" "$SRC" "$APP" "$SIMLIB/libcudart.so" > "$OUT/build.sha256"
git -C "$SIMROOT" rev-parse HEAD > "$OUT/simulator-commit.txt"

for work in "${WORKS[@]}"; do
  for policy in "${POLICIES[@]}"; do
    run="$OUT/${work}_$policy"
    python3 "$SCHEDULER_DIR/make_run_config.py" --base "$SIMROOT/configs/tested-cfgs/SM90_H200" \
      --preset "$policy" --out "$run" > /dev/null
    [[ -z "$SIM_OPTS" ]] || printf '\n%s\n' "$SIM_OPTS" >> "$run/gpgpusim.config"
    status=0
    ( cd "$run" && timeout "${TIME_LIMIT:-1h}" "$APP" "$work" "$INPUT_SIZE" > sim.log 2> stderr.log ) || status=$?
    echo "$status" > "$run/exit_code.txt"
    if [[ "$status" != 0 ]] || ! grep -q "^BENCHMARK_PASS workload=$work " "$run/sim.log"; then
      echo "FAILED: $run (exit $status)" >&2; tail -n 5 "$run/stderr.log" >&2; exit 1
    fi
    echo "$work $policy: PASS"
  done
done

python3 - "$OUT" <<'PY'
import csv, re, sys
from pathlib import Path
out = Path(sys.argv[1]); rows = []
for run in sorted(p for p in out.iterdir() if (p/'sim.log').is_file()):
    text = (run/'sim.log').read_text(errors='replace')
    def last(key, cast=float):
        m = re.findall(r'^\s*' + re.escape(key) + r'\s*=\s*([-+\d.eE]+)\s*$', text, re.M)
        return cast(m[-1]) if m else ''
    hist = {}
    for line in text.splitlines():
        if 'W0_Scoreboard:' in line:
            hist = {k: int(v) for k, v in re.findall(r'(Stall|W\d+(?:_[A-Za-z]+)?):(\d+)', line)}
    issued = sum(hist.get(f'W{i}', 0) for i in range(1, 33))
    lanes, slots = last('proposal_active_lanes', int), last('proposal_lane_slots', int)
    # Locality counters exist only for lav and ldav; they are zero under the other policies.
    refs, uniq = last('proposal_locality_references', int), last('proposal_locality_unique_lines', int)
    perr, cerr = last('proposal_pc_prediction_abs_error', int), last('proposal_neutral_all_prediction_abs_error', int)
    work, policy = run.name.split('_', 1)
    # Each kernel launch adds a fixed latency before any block runs; report cycles net of it.
    launches = re.search(r'^BENCHMARK_PASS .*\bkernels=(\d+)', text, re.M)
    latency = re.search(r'^-gpgpu_kernel_launch_latency\s+(\d+)', (run/'gpgpusim.config').read_text(), re.M)
    cycles = last('gpu_tot_sim_cycle', int)
    active = cycles - int(launches.group(1))*int(latency.group(1)) if launches and latency and cycles else ''
    sched = issued + sum(hist.get(k, 0) for k in ('Stall', 'W0_Idle', 'W0_Scoreboard'))
    rows.append(dict(workload=work, policy=policy,
        cycles=cycles, active_cycles=active, insn=last('gpu_tot_sim_insn', int),
        issued_warp_insn=issued, issue_rate=round(issued/sched, 4) if sched else '',
        issue_lane_ratio=round(sum(i*hist.get(f'W{i}', 0) for i in range(1, 33))/(32*issued), 4) if issued else '',
        resident_lane_ratio=round(lanes/slots, 4) if slots else '',
        line_sharing=round(1 - uniq/refs, 4) if refs else '',
        pc_err_vs_const=round(perr/cerr, 4) if cerr else '',
        Stall=hist.get('Stall', ''), W0_Idle=hist.get('W0_Idle', ''), W0_Scoreboard=hist.get('W0_Scoreboard', ''),
        L1D_misses=last('L1D_total_cache_misses', int), L1D_pending_hits=last('L1D_total_cache_pending_hits', int),
        L1D_miss_rate=last('L1D_total_cache_miss_rate'),
        L1D_resv_fails=last('L1D_total_cache_reservation_fails', int),
        L2_misses=last('L2_total_cache_misses', int), mem_stalls=last('gpgpu_n_stall_shd_mem', int),
        L1_size=','.join(sorted(set(re.findall(r'Reconfigure L1 cache to (\d+KB)', text))))))
gto = {r['workload']: r for r in rows if r['policy'] == 'gto'}
for r in rows:
    g = gto.get(r['workload'])
    r['speedup_vs_gto'] = round(g['cycles']/r['cycles'], 5) if g and r['cycles'] else ''
    r['active_speedup_vs_gto'] = round(g['active_cycles']/r['active_cycles'], 5) if g and g['active_cycles'] and r['active_cycles'] else ''
keys = list(rows[0])
with (out/'summary.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
for work in dict.fromkeys(r['workload'] for r in rows):
    sel = [r for r in rows if r['workload'] == work]
    print('\n' + work)
    for k in keys[1:]:
        print(f'{k:22s}', ' '.join(f'{str(r[k]):>12s}' for r in sel))
    if len({r['insn'] for r in sel}) != 1:
        print('NOTE: instruction counts differ between policies; compare cycles, not IPC.')
print('\nwrote', out/'summary.csv')
PY
