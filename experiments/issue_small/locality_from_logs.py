# Read-only: pulls L1 misses, pending hits, and the patch's locality counters from existing sweep logs.
# Run from the directory that holds the issue_small_n*_b768_* result folders.
import glob, re
from collections import defaultdict
KEYS = ['L1D_total_cache_misses', 'L1D_total_cache_pending_hits', 'proposal_locality_samples',
        'proposal_locality_references', 'proposal_locality_unique_lines', 'proposal_locality_score_sum',
        'proposal_pc_prediction_abs_error', 'proposal_neutral_all_prediction_abs_error']
runs = defaultdict(dict)
for f in glob.glob('issue_small_n*_b768_*/*/sim.log'):
    text = open(f, errors='replace').read()
    v = {}
    for k in KEYS:
        m = re.findall(r'^\s*' + k + r'\s*=\s*(\d+)\s*$', text, re.M)
        v[k] = int(m[-1]) if m else 0
    work, policy = f.split('/')[-2].split('_', 1)
    runs[(work, int(re.search(r'_n(\d+)_b768_', f).group(1)))][policy] = v
others = ['dav_issue', 'lav_issue', 'ldav_issue', 'lrr', 'stall']
print(f"{'workload':8s}{'n':>6s}{'gto_miss':>10s}{'miss_same':>10s}{'gto_pend':>10s}  pending hits % vs GTO "
      + ' '.join(f'{p[:5]:>6s}' for p in others) + '   LAV: sharing  mean_L  pc_err/const')
for (work, n), t in sorted(runs.items()):
    if 'gto' not in t: continue
    g = t['gto']; gm, gp = g[KEYS[0]], g[KEYS[1]]
    same = 'yes' if len({v[KEYS[0]] for v in t.values()}) == 1 else 'NO'
    pend = ' '.join(f"{100*(t[p][KEYS[1]]/gp - 1):6.1f}" if p in t and gp else f"{'':>6s}" for p in others)
    l = t.get('lav_issue', {}); refs, cerr, smp = l.get(KEYS[3]), l.get(KEYS[7]), l.get(KEYS[2])
    lav = (f"{1 - l[KEYS[4]]/refs:9.3f}" if refs else f"{'':>9s}") + (f"{l[KEYS[5]]/smp:8.0f}" if smp else f"{'':>8s}") \
        + (f"{l[KEYS[6]]/cerr:10.3f}" if cerr else '')
    print(f"{work:8s}{n:6d}{gm:10d}{same:>10s}{gp:10d}  {'':22s}{pend}   {'':5s}{lav}")
