"""Generate a synthetic long-row sparse matrix in the PCSR1 format read by real_bench.

Mirrors the SpMV-Scalar input used in DAWS (SHOC random matrix, ~82 nnz/row):
every row has the same number of nonzeros at uniformly random columns, so each
thread of the one-thread-per-row spmv_csr kernel streams a long private run of
values/col_idx and reuses each cache line across many loop iterations.

Usage: python3 gen_long_rows.py OUT.csr [rows] [nnz_per_row] [seed]
"""
import random
import sys


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
    per_row = int(sys.argv[3]) if len(sys.argv) > 3 else 82
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 5764
    cols = rows
    nnz = rows * per_row
    if rows > 1048576 or nnz > 67108864 or per_row > cols:
        sys.exit('exceeds real_bench input limits')
    rng = random.Random(seed)
    with open(out, 'w') as f:
        f.write('PCSR1 %d %d %d\n' % (rows, cols, nnz))
        f.write(' '.join(str(r * per_row) for r in range(rows + 1)))
        f.write('\n')
        values = []
        for r in range(rows):
            row_cols = sorted(rng.sample(range(cols), per_row))
            f.write(' '.join(map(str, row_cols)))
            f.write('\n')
            values.append(' '.join('%g' % ((rng.randrange(17) - 8) * 0.25 or 0.25)
                                   for _ in range(per_row)))
        for line in values:
            f.write(line)
            f.write('\n')
    print('wrote %s rows=%d cols=%d nnz=%d nnz_per_row=%d seed=%d'
          % (out, rows, cols, nnz, per_row, seed))


if __name__ == '__main__':
    main()
