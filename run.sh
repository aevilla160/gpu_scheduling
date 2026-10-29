#!/usr/bin/env bash
set -euo pipefail
bench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
preset="${1:-gto}"
: "${GPGPUSIM_ROOT:?Activate the patched GPGPU-Sim environment first}"
export PTX_SIM_MODE_FUNC=0
config_dir="${H200_CONFIG_DIR:-$GPGPUSIM_ROOT/configs/tested-cfgs/SM90_H200}"
output_dir="${OUTPUT_DIR:-$bench_dir/runs}/$preset"
test -x "$bench_dir/real_bench" || { echo "Build real_bench first" >&2; exit 1; }
test -f "$bench_dir/../scheduler/presets/$preset.config" || { echo "Unknown preset" >&2; exit 1; }
for work in bfs spmv; do
 for stride in 1 32; do
  test ! -e "$output_dir/$work-s$stride" || { echo "Existing output; choose a fresh OUTPUT_DIR" >&2; exit 1; }
 done
done
for work in bfs spmv; do
 file=ca-GrQc.csr
 if [ "$work" = spmv ]; then file=bcsstk17.csr; fi
 test -f "$bench_dir/data/prepared/$file" || { echo "Run download_data.sh first" >&2; exit 1; }
 for stride in 1 32; do
  run_dir="$output_dir/$work-s$stride"
  python3 "$bench_dir/../scheduler/make_run_config.py" --base "$config_dir" --preset "$preset" --out "$run_dir"
  if (cd "$run_dir" && "$bench_dir/real_bench" "$work" "$bench_dir/data/prepared/$file" "$stride" > simulation.log 2>&1); then
   grep -q "^REAL_PASS workload=$work " "$run_dir/simulation.log"
   echo "PASS workload=$work stride=$stride preset=$preset log=$run_dir/simulation.log"
   grep -E '^gpu_tot_(sim_cycle|sim_insn|ipc) =|^REAL_PASS' "$run_dir/simulation.log" | tail -n 4
  else
   status=$?
   tail -n 20 "$run_dir/simulation.log" >&2
   exit "$status"
  fi
 done
done
