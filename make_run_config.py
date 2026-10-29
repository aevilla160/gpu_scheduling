import argparse
import re
import shutil
from pathlib import Path

presets=Path(__file__).resolve().parent/'presets'
p=argparse.ArgumentParser(description='Copy H200 config and select one scheduler preset')
p.add_argument('--base',type=Path,required=True)
p.add_argument('--preset',choices=sorted(x.stem for x in presets.glob('*.config')),required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if not (a.base/'gpgpusim.config').is_file():p.error('Missing base gpgpusim.config')
if a.out.exists():p.error('Output exists; choose a fresh run directory')
shutil.copytree(a.base,a.out)
config=a.out/'gpgpusim.config'
text=config.read_text()
text='\n'.join(line for line in text.splitlines() if not re.match(
    r'^\s*-gpgpu_(scheduler|ld_alpha|locality_predictor|priority_fetch)\s',line))
config.write_text(text+'\n\n'+(presets/(a.preset+'.config')).read_text(),newline='\n')
print(config.resolve())
