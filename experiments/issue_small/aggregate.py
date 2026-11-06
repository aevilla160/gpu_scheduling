# Cross-size summary for one run tag. Run from the directory holding the result folders:
#   python3 aggregate.py            # stock runs, issue_small_n*_b768_*
#   python3 aggregate.py row11      # issue_row11_n*_b768_*
# A = speedup vs GTO on active cycles, which follows the slowest block.
# M = speedup vs GTO on summed scheduler cycles, which follows the mean block time.
import csv, glob, statistics as st, sys
from collections import defaultdict
tag = sys.argv[1] if len(sys.argv) > 1 else 'small'
A, M = defaultdict(list), defaultdict(list)
def sched(r): return sum(int(r[k]) for k in ('Stall', 'W0_Idle', 'W0_Scoreboard', 'issued_warp_insn'))
for f in sorted(glob.glob(f'issue_{tag}_n*_b768_*/summary.csv')):
    rows = list(csv.DictReader(open(f)))
    gto = {r['workload']: r for r in rows if r['policy'] == 'gto'}
    for r in rows:
        g = gto.get(r['workload'])
        if r['policy'] == 'gto' or not g or not r.get('active_cycles') or not r.get('Stall'): continue
        A[r['workload'], r['policy']].append(100*(int(g['active_cycles'])/int(r['active_cycles']) - 1))
        M[r['workload'], r['policy']].append(100*(sched(g)/sched(r) - 1))
def cols(v): return f"{st.mean(v):7.2f}{min(v):7.2f}{max(v):7.2f}{sum(x > 0 for x in v):5d}"
print(f"{'workload':9s}{'policy':12s}{'sizes':>6s} | {'A mean':>7s}{'min':>7s}{'max':>7s}{'pos':>5s} | {'M mean':>7s}{'min':>7s}{'max':>7s}{'pos':>5s}")
for key in sorted(A):
    print(f"{key[0]:9s}{key[1]:12s}{len(A[key]):6d} | {cols(A[key])} | {cols(M[key])}")
