# SPDX-License-Identifier: Apache-2.0
"""Source-linked deterministic CopyTo lifetime checks; not package inference."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {'Result.cs', 'Handles.cs', 'Diagnostics.cs', 'NativeTypes.cs', 'ApiTypes.cs'}
PREMATURE_RELEASE = 'Result physically released before admitted CopyTo completed'


def prepare(destination, shortened, framework='net10.0', runtime='10.0.12'):
    source = ROOT / 'tests/sdk-install/CopyOwnership'
    tree = ET.parse(source / 'CopyOwnership.csproj')
    if tree.findtext('.//TargetFramework') != 'net10.0' or tree.findtext('.//RuntimeFrameworkVersion') != '10.0.12':
        raise ValueError('CopyOwnership stable baseline changed')
    expected = {ROOT / 'sdk/dotnet/FactsYOLO' / name for name in SOURCES}
    actual = {(source / item.attrib['Include']).resolve() for item in tree.findall('.//Compile')}
    if actual != {path.resolve() for path in expected} or tree.findall('.//ProjectReference') or tree.findall('.//PackageReference'):
        raise ValueError('CopyOwnership linked source inventory changed')
    inputs = [p for p in source.rglob('*') if p.relative_to(source).parts[0] not in ('bin', 'obj')]
    if any(p.is_symlink() for p in inputs) or {p.relative_to(source).as_posix() for p in inputs if p.is_file() and p.suffix.lower() == '.cs'} != {'Program.cs'}:
        raise ValueError('CopyOwnership implicit source inventory changed')
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / 'Program.cs', destination / 'Program.cs')
    (destination / 'Sdk').mkdir()
    hashes = {}
    for path in expected:
        shutil.copyfile(path, destination / 'Sdk' / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for group in tree.findall('.//ItemGroup'):
        for item in list(group):
            if item.tag == 'Compile':
                group.remove(item)
    tree.find('.//TargetFramework').text = framework
    tree.find('.//RuntimeFrameworkVersion').text = runtime
    project = destination / 'CopyOwnership.csproj'
    tree.write(project, encoding='utf-8', xml_declaration=False)
    result = destination / 'Sdk/Result.cs'
    if shortened:
        text = result.read_text()
        start = text.index('public void CopyTo(Span<Detection> destination)')
        end = text.index('private static void CopyWithLease', start)
        body = text[start:end]
        marker = '        int capacity = required < (nuint)destination.Length ? (int)required : destination.Length;'
        if body.count(marker) != 1 or body.count('using Lease lease = lifetime.Acquire();') != 1:
            raise ValueError('CopyTo lease/count boundary changed; review negative control')
        body = body.replace(marker, '        lease.Dispose(); // Deliberate test-only premature release.\n' + marker)
        result.write_text(text[:start] + body + text[end:])
    return project, {'source_sha256': hashes, 'compiled_result_sha256': hashlib.sha256(result.read_bytes()).hexdigest(),
                     'test_only_shortened_lease': shortened, 'framework': framework, 'runtime': runtime}


def verify(mode, rid, evidence, config, run, framework='net10.0', runtime='10.0.12'):
    for variant, shortened in [('baseline', False), ('short-lease', True)]:
        directory = evidence / f'copy-ownership-{mode}-{variant}'
        project, provenance = prepare(directory / 'source', shortened, framework, runtime)
        (evidence / f'copy-ownership-provenance-{mode}-{variant}.json').write_text(json.dumps(provenance, indent=2) + '\n')
        publish = directory / 'publish'
        run(f'copy-ownership-publish-{mode}-{variant}', ['dotnet', 'publish', project, '-c', 'Release',
            '-r', rid, '-o', publish, '--self-contained', 'true' if mode == 'aot' else 'false',
            f'-p:PublishAot={str(mode == "aot").lower()}', '--configfile', config])
        execute = ['dotnet', '--fx-version', runtime, publish / 'CopyOwnership.dll'] if mode == 'jit' else [publish / ('CopyOwnership.exe' if os.name == 'nt' else 'CopyOwnership')]
        for phase in ['query', 'copy']:
            expected = PREMATURE_RELEASE if shortened else f'PASS {mode} source-linked CopyTo {phase}:'
            run(f'copy-ownership-run-{mode}-{variant}-{phase}', [*execute, mode, phase],
                expected=1 if shortened else 0, contains=[expected])
