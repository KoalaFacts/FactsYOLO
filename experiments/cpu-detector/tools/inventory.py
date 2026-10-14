#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record selected host dependencies; registry lock entries alone include unused backends."""
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
manifest=str(ROOT/"Cargo.toml")
host=next(x.split(': ',1)[1] for x in subprocess.check_output(['rustc','-vV'],text=True).splitlines() if x.startswith('host:'))
common=['--manifest-path',manifest,'--locked','--offline']
metadata=json.loads(subprocess.check_output(['cargo','metadata',*common,'--format-version','1','--filter-platform',host],text=True))
tree=subprocess.check_output(['cargo','tree',*common,'--target',host,'-e','normal,build','--prefix','none','--format','{p}'],text=True)
active=set(re.findall(r'^([\w-]+) v([^\s]+)',tree,re.M))
features={node['id']:node['features'] for node in metadata['resolve']['nodes']}
packages=[]
for package in metadata['packages']:
    if (package['name'],package['version']) not in active or package['source'] is None:
        continue
    packages.append({key:package[key] for key in ['name','version','source','license','repository','rust_version','links']})
    packages[-1]['features']=features[package['id']]
    packages[-1]['build_script']=any('custom-build' in target['kind'] for target in package['targets'])
if any(p['license'] is None for p in packages):
    raise RuntimeError('dependency has no declared license; review required')
packages.sort(key=lambda p:(p['name'],p['version']))
(ROOT/'dependencies.json').write_text(json.dumps({'target':host,'scope':'selected normal/build dependencies; not a binary SBOM','packages':packages},indent=2)+'\n')
print(f'Recorded {len(packages)} selected registry packages for {host}')
