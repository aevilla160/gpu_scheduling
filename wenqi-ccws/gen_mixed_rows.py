"""Generate a sparse matrix whose warps differ in cache reuse, in PCSR1 format.

spmv_csr runs one thread per row, so rows 32k..32k+31 form one warp. Each such
group of 32 rows is made either "long" (every row has LONG nonzeros: each thread
streams a long private run and reuses each cache line many times) or "short"
(every row has SHORT nonzeros: little reuse). Long groups are chosen at random
with probability LONG_FRAC, so every thread block mixes both kinds of warps.

Usage: python3 gen_mixed_rows.py OUT.csr [rows=131072] [long=200] [short=8]
                                         [long_frac=0.25] [seed=5764]
"""
import random
import sys


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    a = sys.argv
    out = a[1]
    rows = int(a[2]) if len(a) > 2 else 131072
    long_nnz = int(a[3]) if len(a) > 3 else 200
    short_nnz = int(a[4]) if len(a) > 4 else 8
    long_frac = float(a[5]) if len(a) > 5 else 0.25
    seed = int(a[6]) if len(a) > 6 else 5764
    cols = rows
    if rows % 32 or max(long_nnz, short_nnz) > cols:
        sys.exit('rows must be a multiple of 32 and row length <= rows')
    rng = random.Random(seed)
    groups = rows // 32
    is_long = [rng.random() < long_frac for _ in range(groups)]
    lengths = [long_nnz if is_long[r // 32] else short_nnz for r in range(rows)]
    nnz = sum(lengths)
    if rows > 1048576 or nnz > 67108864:
        sys.exit('exceeds real_bench input limits')
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    with open(out, 'w') as f:
        f.write('PCSR1 %d %d %d\n' % (rows, cols, nnz))
        f.write(' '.join(map(str, offsets)))
        f.write('\n')
        values = []
        for n in lengths:
            f.write(' '.join(map(str, sorted(rng.sample(range(cols), n)))))
            f.write('\n')
            values.append(' '.join('%g' % ((rng.randrange(17) - 8) * 0.25 or 0.25)
                                   for _ in range(n)))
        for line in values:
            f.write(line)
            f.write('\n')
    print('wrote %s rows=%d cols=%d nnz=%d long_warps=%d/%d long=%d short=%d seed=%d'
          % (out, rows, cols, nnz, sum(is_long), groups, long_nnz, short_nnz, seed))


if __name__ == '__main__':
    main()
