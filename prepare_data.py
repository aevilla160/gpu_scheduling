#!/usr/bin/env python3
import collections
import gzip
import hashlib
import json
import math
import struct
from pathlib import Path

def f32(x):
    value=struct.unpack('<f',struct.pack('<f',x))[0]
    if not math.isfinite(value):raise ValueError('Nonfinite FP32 matrix value')
    return value

def write_csr(path,rows,cols,entries):
    offsets=[0];indices=[];values=[]
    for row in entries:
        for col,value in sorted(row.items()):
            if not 0<=col<cols:raise ValueError('Column out of range')
            indices.append(col);values.append(f32(value))
        offsets.append(len(indices))
    if len(offsets)!=rows+1:raise ValueError('Row count mismatch')
    with path.open('w') as out:
        out.write('PCSR1 %d %d %d\n'%(rows,cols,len(indices)))
        out.write(' '.join(map(str,offsets))+'\n')
        out.write(' '.join(map(str,indices))+'\n')
        out.write(' '.join(format(v,'.9g') for v in values)+'\n')
    return len(indices)

def prepare_graph(raw,out):
    adj=collections.defaultdict(set)
    with gzip.open(raw,'rt') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):continue
            a,b=map(int,line.split());adj[a].add(b);adj[b].add(a)
    # Use the max-degree vertex as root and make its internal ID zero.
    root=min(adj,key=lambda v:(-len(adj[v]),v))
    ids=[root]+sorted(v for v in adj if v!=root)
    lookup={v:i for i,v in enumerate(ids)}
    rows=[{lookup[u]:1.0 for u in adj[v]} for v in ids]
    nnz=write_csr(out,len(ids),len(ids),rows)
    distance={root:0};q=collections.deque([root])
    while q:
        v=q.popleft()
        for u in adj[v]:
            if u not in distance:distance[u]=distance[v]+1;q.append(u)
    return dict(dataset='ca-GrQc',rows=len(ids),columns=len(ids),stored_directed_edges=nnz,
        root_original_id=root,reachable=len(distance),max_distance=max(distance.values()),
        source='https://snap.stanford.edu/data/ca-GrQc.html',
        download='https://snap.stanford.edu/data/ca-GrQc.txt.gz',
        transformation='undirected union, deduplicate, deterministic compact IDs; max-degree root mapped to zero')

def prepare_matrix(raw,out):
    with gzip.open(raw,'rt') as f:
        banner=f.readline().strip().lower().split()
        if banner!=['%%matrixmarket','matrix','coordinate','real','symmetric']:
            raise ValueError('Expected real symmetric coordinate Matrix Market')
        for line in f:
            if not line.startswith('%') and line.strip():break
        n,m,stored=map(int,line.split())
        if n!=m:raise ValueError('Symmetric matrix must be square')
        rows=[{} for _ in range(n)];count=0
        for line in f:
            if not line.strip() or line.startswith('%'):continue
            r,c,v=line.split();r=int(r)-1;c=int(c)-1;v=float(v.replace('D','E'))
            if not (0<=r<n and 0<=c<m and math.isfinite(v)):raise ValueError('Invalid matrix entry')
            rows[r][c]=rows[r].get(c,0.0)+v
            if r!=c:rows[c][r]=rows[c].get(r,0.0)+v
            count+=1
        if count!=stored:raise ValueError('Matrix Market entry count mismatch')
    nnz=write_csr(out,n,m,rows)
    return dict(dataset='bcsstk17',rows=n,columns=m,stored_triangular_entries=stored,expanded_nnz=nnz,
        source='https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc2/bcsstk17.html',
        download='https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/bcsstruc2/bcsstk17.mtx.gz',
        transformation='expand symmetric off-diagonal entries, merge duplicates, sort rows, round values to FP32')

def main():
    root=Path(__file__).resolve().parent
    raw=root/'data/raw';prepared=root/'data/prepared';prepared.mkdir(parents=True,exist_ok=True)
    records=[]
    for infile,outfile,fn in [('ca-GrQc.txt.gz','ca-GrQc.csr',prepare_graph),('bcsstk17.mtx.gz','bcsstk17.csr',prepare_matrix)]:
        src=raw/infile;dst=prepared/outfile;record=fn(src,dst)
        record.update(raw_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
            csr_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),csr_file=outfile)
        records.append(record)
    if records[0]['rows']!=5242 or records[0]['reachable']!=4158:raise ValueError('SNAP metadata mismatch')
    if records[1]['rows']!=10974 or records[1]['expanded_nnz']!=428650:raise ValueError('Matrix metadata mismatch')
    (root/'data/manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
