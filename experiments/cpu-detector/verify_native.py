#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise trained TD1 artifacts through real C and managed native consumers."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]

# Ground-truth prefix, registered independently of trained model predictions.
# Seed909's first six examples have background and class1 but no class0.
NATIVE_CASES = 8

def select_native_cases(rows):
    """Select the fixed held-out prefix, failing if any target class is absent."""
    if not isinstance(rows, list) or len(rows) < NATIVE_CASES:
        raise ValueError('missing native held-out predictions')
    selected = rows[:NATIVE_CASES]
    for i, row in enumerate(selected):
        if (not isinstance(row, dict) or type(row.get('index')) is not int
                or row['index'] != i or row.get('expected_class') not in (None, 0, 1)
                or (row.get('expected_class') is not None and type(row['expected_class']) is not int)):
            raise ValueError('invalid native held-out reference ordering/class')
    if {row['expected_class'] for row in selected} != {None, 0, 1}:
        raise ValueError('native held-out inputs must cover background and both classes')
    return selected



def main():
    if len(sys.argv)!=3:
        raise ValueError('usage: verify_native.py <passed-training-directory> <new-evidence-directory>')
    windows=platform.system()=='Windows'
    supported=(platform.system()=='Linux' and platform.machine()=='x86_64') or (windows and platform.machine().lower() in ('amd64','x86_64'))
    if not supported:
        raise ValueError('real C/C# TD1 gate requires Linux x64 or Windows x64')
    rid='win-x64' if windows else 'linux-x64'
    source=Path(sys.argv[1]).resolve()
    destination=Path(sys.argv[2]).resolve()
    report=json.loads((source/'metrics.json').read_text())
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if report.get('passed') is not True or report.get('source_commit')!=head or report.get('steps')!=2000 or report.get('test_seed')!=909:
        raise ValueError('require a passed actual training report for the current source and fixed experiment')
    artifact=source/'model.fytd'
    if artifact.stat().st_size!=51284:
        raise ValueError('invalid fixed TD1 artifact length')
    destination.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['FACTSYOLO_BUILD_ID']=head;env.pop('LD_PRELOAD',None)
    commands=[]

    def run(name,args,cwd=ROOT,stdout_only=False):
        print(f"{name}: {' '.join(map(str,args))}",flush=True)
        started=time.monotonic()
        with (destination/f'{name}.log').open('w') as log:
            result=subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=subprocess.PIPE if stdout_only else log,stderr=log,timeout=900,text=True)
            if stdout_only: log.write(result.stdout)
        commands.append({'name':name,'argv':list(map(str,args)),'cwd':str(cwd),'exit':result.returncode,'seconds':time.monotonic()-started})
        (destination/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if result.returncode!=0:
            raise RuntimeError(f'{name} failed: {destination / (name+".log")}')
        return result.stdout if stdout_only else (destination/f'{name}.log').read_text()

    target=ROOT/'target/td1-native'
    common=['--locked','--offline','--no-default-features','--features','td1','--target-dir',target]
    run('native-build',['cargo','build','-p','facts-yolo','--release',*common])
    release=target/'release'
    shared=release/('facts_yolo.dll' if windows else 'libfacts_yolo.so')
    libraries=[('shared',release/'facts_yolo.dll.lib')] if windows else [('shared',shared),('static',release/'libfacts_yolo.a')]
    expected=set((ROOT/'tests/c/public-symbols.txt').read_text().splitlines())
    for kind,library in libraries:
        file=shared if kind=='shared' else library
        if windows:
            symbols=run(f'exports-{kind}',['dumpbin','/nologo','/exports',file])
            # MSVC may append " = symbol"; the public export is the name column.
            names={match.group(1) for line in symbols.splitlines() if (match:=re.fullmatch(r'\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(fy_\w+)(?:\s+=\s+\S+)?\s*',line))}
        else:
            symbols=run(f'exports-{kind}',['nm','-D' if kind=='shared' else '-g','--defined-only',file])
            names={parts[-1] for line in symbols.splitlines() if len(parts:=line.split())==3 and parts[-2]=='T' and parts[-1].startswith('fy_')}
        if names!=expected:
            raise RuntimeError(f'{kind} public export set differs: {names.symmetric_difference(expected)}')
    layout=run('rust-layout',['cargo','run','-p','facts-yolo','--example','abi_layout','--quiet',*common],stdout_only=True)
    layout_path=destination/'rust-layout.txt';layout_path.write_text(layout)
    cases=destination/'cases';cases.mkdir()
    for i, prediction in enumerate(select_native_cases(report.get('predictions'))):
        if prediction['index']!=i:
            raise ValueError('unexpected reference sample ordering')
        p=prediction['prediction']
        text='0\n' if p is None else '1\n'+' '.join(map(str,[p['class'],p['score'],*p['corners']]))+'\n'
        (cases/f'expected-{i}.txt').write_text(text)
        shutil.copyfile(source/f'sample-{i}.rgb',cases/f'sample-{i}.rgb')
    for kind,library in libraries:
        executable=destination/(f'c-{kind}.exe' if windows else f'c-{kind}')
        if windows:
            # Keep the portable consumer's bounded standard-C file I/O; /W4 /WX still apply.
            run(f'c-build-{kind}',[os.environ.get('CC','cl'),'/nologo','/std:c11','/W4','/WX','/TC','/D_CRT_SECURE_NO_WARNINGS',
                                  '/I'+str(ROOT/'include'),ROOT/'tests/c/td1.c','/Fe:'+str(executable),'/Fo:'+str(destination/f'c-{kind}.obj'),'/link',library])
            shutil.copyfile(shared,destination/shared.name)
        else:
            flags=['-DFY_STATIC'] if kind=='static' else []
            run(f'c-build-{kind}',[os.environ.get('CC','cc'),'-std=c11','-Wall','-Wextra','-Werror',*flags,'-I',ROOT/'include',ROOT/'tests/c/td1.c',library,'-Wl,-rpath,'+str(release),'-ldl','-lpthread','-lm','-o',executable])
        for i in range(NATIVE_CASES):
            run(f'c-{kind}-{i}',[executable,artifact,cases/f'sample-{i}.rgb',cases/f'expected-{i}.txt'])
    managed_root=ROOT/'tests/dotnet'
    sdk=run('dotnet-version',['dotnet','--version'],managed_root,stdout_only=True).strip()
    if sdk!='10.0.401':
        raise RuntimeError(f'expected .NET10.0.401, got {sdk}')
    project=managed_root/'FactsYOLO.Smoke/FactsYOLO.Smoke.csproj'
    for mode in ['jit','aot']:
        output=destination/f'managed-{mode}'
        extra=['-r',rid,'--self-contained','true','-p:PublishAot=true'] if mode=='aot' else []
        run(f'build-{mode}',['dotnet','publish' if mode=='aot' else 'build',project,'-c','Release','-o',output,*extra],managed_root)
        executable=['dotnet',output/'FactsYOLO.Smoke.dll'] if mode=='jit' else [output/('FactsYOLO.Smoke.exe' if windows else 'FactsYOLO.Smoke')]
        run(f'run-{mode}',[*executable,shared,layout_path,'td1',mode,artifact,cases],managed_root)
    summary={'passed':True,'source_commit':head,'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
             'training_backend':report['backend'],'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
             'heldout_samples':NATIVE_CASES,'c_link_modes':[kind for kind,_ in libraries],'managed_execution_modes':['jit','aot'],
             'runtime_identifier':rid,'platform':platform.platform()}
    (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    try:
        main()
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        print(f'FAIL: {error}',file=sys.stderr);sys.exit(1)
