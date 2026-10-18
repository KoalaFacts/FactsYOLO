#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Execute optional net11.0 consumers against the unchanged net10.0 test package."""
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from xml.sax.saxutils import escape

from compat import SETTINGS, prepare_project, preview_command, validate_training_report
from install_sdk import host_rid

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('stable_sdk_verifier', ROOT / 'tests/sdk-install/verify.py')
stable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stable)


def main(feed, training, sdk, evidence):
    commit = stable.clean_commit()
    rid = host_rid()
    metadata = json.loads((feed / 'package.json').read_text())
    if metadata['source_commit'] != commit or metadata['local_test_only'] is not True:
        raise ValueError('package source differs')
    package = feed / metadata['package']
    if package.parent != feed or stable.sha256(package) != metadata['sha256']:
        raise ValueError('package hash differs')
    receipt = json.loads((sdk / 'sdk-installation.json').read_text())
    if receipt['sdk'] != SETTINGS['sdk'] or receipt['runtime'] != SETTINGS['runtime'] or receipt['rid'] != rid or receipt['sha512'] != SETTINGS['assets'][rid]['sha512']:
        raise ValueError('preview SDK installation receipt differs')
    evidence.mkdir(parents=True, exist_ok=False)
    detached = Path(tempfile.mkdtemp(prefix='factsyolo-dotnet11-consumer-')).resolve()
    if detached.is_relative_to(ROOT):
        raise ValueError('consumer must be outside checkout')
    project = detached / 'consumer'
    consumer = prepare_project(ROOT / 'tests/sdk-install/Consumer', project, ROOT)
    ownership_project = prepare_project(ROOT / 'tests/sdk-install/Ownership', detached / 'ownership', ROOT)
    (detached / 'global.json').write_text(json.dumps({'sdk': {'version': SETTINGS['sdk'], 'rollForward': 'disable', 'allowPrerelease': True}}, indent=2) + '\n')
    local_feed = detached / 'feed'; local_feed.mkdir()
    shutil.copyfile(package, local_feed / package.name)
    library = 'facts_yolo.dll' if rid == 'win-x64' else 'libfacts_yolo.so'
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts or '\\' in n or ':' in n for n in names):
            raise ValueError('unsafe package entries')
        assets = detached / 'assets'
        for name in names:
            if name.startswith('buildinfo/') and not name.endswith('/'):
                target = assets / name.removeprefix('buildinfo/')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        for target_rid, target_library in [('linux-x64', 'libfacts_yolo.so'), ('win-x64', 'facts_yolo.dll')]:
            (assets / target_rid / target_library).write_bytes(archive.read(f'runtimes/{target_rid}/native/{target_library}'))
    manifests = stable.verify_assets(assets, commit)
    native_hash = manifests[rid]['files'][library]
    config = detached / 'NuGet.Config'
    config.write_text(f'''<configuration>
<packageSources><clear/><add key="local" value="{escape(str(local_feed), {'"': '&quot;'})}"/><add key="official" value="https://api.nuget.org/v3/index.json"/></packageSources>
<packageSourceMapping><clear/><packageSource key="local"><package pattern="FactsYOLO.LocalTest"/></packageSource><packageSource key="official"><package pattern="*"/></packageSource></packageSourceMapping>
</configuration>''')
    env = os.environ.copy()
    env.update({'DOTNET_ROOT': str(sdk), 'DOTNET_ROOT_X64': str(sdk), 'PATH': str(sdk) + os.pathsep + env['PATH'],
                'DOTNET_CLI_HOME': str(detached / 'dotnet-home'), 'NUGET_PACKAGES': str(detached / 'packages'),
                'NUGET_HTTP_CACHE_PATH': str(detached / 'http-cache'), 'DOTNET_CLI_TELEMETRY_OPTOUT': '1'})
    env.pop('LD_PRELOAD', None)
    if rid == 'linux-x64':
        env['XDG_RUNTIME_DIR'] = tempfile.mkdtemp(prefix='factsyolo-dotnet11-runtime-')
    if Path(env['NUGET_PACKAGES']).exists() or Path(env['NUGET_HTTP_CACHE_PATH']).exists():
        raise ValueError('NuGet caches must start empty')
    dotnet = sdk / ('dotnet.exe' if rid == 'win-x64' else 'dotnet')
    unrelated = detached / 'unrelated'; unrelated.mkdir()
    commands = []

    def run(label, args, cwd=project, expected=0, contains=()):
        args = preview_command(args, dotnet)
        started = time.monotonic()
        result = subprocess.run(list(map(str, args)), cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
        (evidence / (label + '.log')).write_text(result.stdout, encoding='utf-8')
        commands.append({'label': label, 'argv': list(map(str, args)), 'cwd': str(cwd),
                         'exit': result.returncode, 'expected_exit': expected, 'seconds': time.monotonic() - started})
        (evidence / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        print(f'{label}: exit={result.returncode} expected={expected}', flush=True)
        if result.returncode != expected or any(text not in result.stdout for text in contains):
            print(result.stdout[-12000:], flush=True)
            raise RuntimeError(label + ' failed')
        return result.stdout

    if run('sdk-version', [dotnet, '--version']).strip() != SETTINGS['sdk']:
        raise ValueError('preview SDK version differs')
    runtimes = run('sdk-runtimes', [dotnet, '--list-runtimes'])
    if re.findall(r'Microsoft.NETCore.App (\S+)', runtimes) != [SETTINGS['runtime']]:
        raise ValueError('preview runtime differs')
    models = []
    for backend in ['burn', 'candle']:
        source = training / backend
        report = json.loads((source / 'metrics.json').read_text())
        validate_training_report(report, backend, commit)
        cases = detached / backend; cases.mkdir()
        shutil.copyfile(source / 'model.fytd', cases / 'model.fytd')
        for i, prediction in enumerate(report['predictions'][:6]):
            if prediction['index'] != i:
                raise ValueError('reference ordering differs')
            value = prediction['prediction']
            text = '0\n' if value is None else '1\n' + ' '.join(map(str, [value['class'], value['score'], *value['corners']])) + '\n'
            (cases / f'expected-{i}.txt').write_text(text)
            shutil.copyfile(source / f'sample-{i}.rgb', cases / f'sample-{i}.rgb')
        if len(list(cases.glob('sample-*.rgb'))) != 6:
            raise ValueError('six held-out samples required')
        models.append((backend, cases, stable.sha256(cases / 'model.fytd')))
    selections = {}
    for mode in ['jit', 'aot']:
        publish = detached / ('publish-' + mode)
        flags = ['-c', 'Release', '-r', rid, '--self-contained', 'true' if mode == 'aot' else 'false',
                 f'-p:PublishAot={str(mode == "aot").lower()}', '--configfile', config]
        run('publish-' + mode, [dotnet, 'publish', consumer, *flags, '-o', publish, f'-p:SdkTestVersion={metadata["version"]}'])
        restored = json.loads((project / 'obj/project.assets.json').read_text())
        key = 'FactsYOLO.LocalTest/' + metadata['version']
        if restored['libraries'][key]['type'] != 'package' or any(v['type'] == 'project' for v in restored['libraries'].values()):
            raise ValueError('public consumer must install a package')
        if {Path(p).resolve() for p in restored['packageFolders']} != {Path(env['NUGET_PACKAGES']).resolve()}:
            raise ValueError('non-isolated NuGet cache')
        selected = restored['targets'][f'{SETTINGS["framework"]}/{rid}'][key]
        if set(selected.get('compile', {})) != {'lib/net10.0/FactsYOLO.dll'} or set(selected.get('runtime', {})) != {'lib/net10.0/FactsYOLO.dll'} or set(selected.get('native', {})) != {f'runtimes/{rid}/native/{library}'}:
            raise ValueError('preview must consume unchanged net10.0 package assets')
        cached = Path(env['NUGET_PACKAGES']) / 'factsyolo.localtest' / metadata['version'] / package.name.lower()
        if stable.sha256(cached) != metadata['sha256'] or stable.sha256(publish / library) != native_hash:
            raise ValueError('installed package/native hash differs')
        if mode == 'jit':
            runtime = json.loads((publish / 'Consumer.runtimeconfig.json').read_text())['runtimeOptions']
            if runtime['tfm'] != SETTINGS['framework'] or runtime['framework']['version'] != SETTINGS['runtime'] or runtime['rollForward'] != 'Disable':
                raise ValueError('JIT runtime is not pinned to RC2')
        elif 'Microsoft.DotNet.ILCompiler/' + SETTINGS['runtime'] not in restored['libraries']:
            raise ValueError('NativeAOT compiler differs')
        selections[mode] = {'managed_and_native': selected, 'libraries': sorted(restored['libraries'])}
        (evidence / 'asset-selections.json').write_text(json.dumps(selections, indent=2) + '\n')
        execute = [dotnet, '--fx-version', SETTINGS['runtime'], publish / 'Consumer.dll'] if mode == 'jit' else [publish / ('Consumer.exe' if rid == 'win-x64' else 'Consumer')]
        for backend, cases, _ in models:
            run(f'run-{mode}-{backend}', [*execute, cases / 'model.fytd', cases, commit, native_hash], cwd=unrelated,
                contains=[f'MODE {mode}', 'PASS public SDK sample 5', 'PASS public SDK: 32'])
        ownership = evidence / ('ownership-' + mode)
        run('ownership-publish-' + mode, [dotnet, 'publish', ownership_project, *flags, '-o', ownership])
        ownership_command = [dotnet, '--fx-version', SETTINGS['runtime'], ownership / 'Ownership.dll'] if mode == 'jit' else [ownership / ('Ownership.exe' if rid == 'win-x64' else 'Ownership')]
        run('ownership-run-' + mode, [*ownership_command, publish / library, models[0][1] / 'model.fytd', mode], cwd=unrelated,
            contains=[f'PASS {mode} admitted SDK lease survives Dispose', 'exactly one model release'])
        stable.loader_negatives(mode, publish, library, unrelated, evidence, run)
        stable.copy_ownership.verify(mode, rid, evidence, config, run, SETTINGS['framework'], SETTINGS['runtime'])
        stable.session_busy.verify(mode, rid, evidence, config, run, SETTINGS['framework'], SETTINGS['runtime'])
    summary = {'passed': True, 'source_commit': commit, 'source_dirty': False, 'rid': rid,
               'package_sha256': metadata['sha256'], 'native_sha256': native_hash,
               'consumer_sdk': SETTINGS['sdk'], 'consumer_runtime': SETTINGS['runtime'], 'consumer_framework': SETTINGS['framework'],
               'package_framework': 'net10.0', 'sdk_archive_sha512': receipt['sha512'],
               'models': {name: digest for name, _, digest in models}, 'heldout_samples_per_model': 6,
               'modes': ['jit', 'aot'], 'fresh_nuget_cache': True, 'consumer_directory': str(detached),
               'platform': platform.platform(), 'local_test_only': True}
    (evidence / 'sdk-installation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (evidence / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit('usage: verify.py <feed> <passed-training> <verified-dotnet11-root> <new-evidence>')
    main(*(Path(arg).resolve() for arg in sys.argv[1:]))
