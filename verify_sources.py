import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description='Check source identity before/after applying the scheduler patch')
p.add_argument('mode',choices=['base','patched'])
p.add_argument('checkout',type=Path)
a=p.parse_args()
m=json.loads((Path(__file__).resolve().parent/'MANIFEST.json').read_text())
failed=[]
for name,expected in m[a.mode+'_sha256'].items():
    path=a.checkout/name
    actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else 'MISSING'
    if actual!=expected:failed.append(name)
if failed:raise SystemExit('SOURCE MISMATCH: '+', '.join(failed))
print('All five source files match:',a.mode)
