# SPDX-License-Identifier: Apache-2.0
"""Deterministic production-Session BUSY checks against a test-only boundary."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {'Session.cs', 'Result.cs', 'Handles.cs', 'Diagnostics.cs', 'NativeTypes.cs', 'ApiTypes.cs'}
FAILURES = {
    'swallow': 'Concurrent Detect did not preserve native BUSY status 6',
    'serialize': 'Concurrent Detect did not complete while the first call remained paused',
    'retry': 'Concurrent Detect retried native BUSY',
}


def prepare(destination, variant, framework='net10.0', runtime='10.0.12'):
    if variant not in ('baseline', *FAILURES):
        raise ValueError('Unknown SessionBusy variant')
    source = ROOT / 'tests/sdk-install/SessionBusy'
    tree = ET.parse(source / 'SessionBusy.csproj')
    if tree.findtext('.//TargetFramework') != 'net10.0' or tree.findtext('.//RuntimeFrameworkVersion') != '10.0.12':
        raise ValueError('SessionBusy stable baseline changed')
    expected = {ROOT / 'sdk/dotnet/FactsYOLO' / name for name in SOURCES}
    actual = {(source / item.attrib['Include']).resolve() for item in tree.findall('.//Compile')}
    if actual != {path.resolve() for path in expected} or tree.findall('.//ProjectReference') or tree.findall('.//PackageReference'):
        raise ValueError('SessionBusy linked source inventory changed')
    inputs = [p for p in source.rglob('*') if p.relative_to(source).parts[0] not in ('bin', 'obj')]
    if any(p.is_symlink() for p in inputs) or {p.relative_to(source).as_posix() for p in inputs if p.is_file() and p.suffix.lower() == '.cs'} != {'Program.cs'}:
        raise ValueError('SessionBusy implicit source inventory changed')
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
    project = destination / 'SessionBusy.csproj'
    tree.write(project, encoding='utf-8', xml_declaration=False)
    session = destination / 'Sdk/Session.cs'
    if variant != 'baseline':
        text = session.read_text()
        start = text.index('    public Result Detect(')
        end = text.index('    public void Dispose()', start)
        body = text[start:end]
        call = 'Native.Detect(lease.Pointer, &image, &native, &raw, &diagnostic)'
        marker = f'                Diagnostics.Check({call}, &diagnostic);'
        lease = '        using Lease lease = lifetime.Acquire();'
        if body.count(marker) != 1 or body.count(lease) != 1:
            raise ValueError('Session Detect boundary changed; review negative controls')
        if variant == 'serialize':
            body = body.replace(lease, '        lock (lifetime) // Deliberate test-only serialization.\n        {\n' + lease)
            close = body.rindex('    }\n')
            body = body[:close] + '        }\n' + body[close:]
        else:
            changed = f'                int status = {call};\n'
            if variant == 'retry':
                changed += f'                if (status == 6) status = {call}; // Deliberate test-only retry.\n'
            checked = 'status == 6 ? 0 : status' if variant == 'swallow' else 'status'
            changed += f'                Diagnostics.Check({checked}, &diagnostic);'
            body = body.replace(marker, changed)
        session.write_text(text[:start] + body + text[end:])
    return project, {'source_sha256': hashes, 'compiled_session_sha256': hashlib.sha256(session.read_bytes()).hexdigest(),
                     'test_program_sha256': hashlib.sha256((destination / 'Program.cs').read_bytes()).hexdigest(),
                     'variant': variant, 'source_linked_only': True, 'framework': framework, 'runtime': runtime}


def verify(mode, rid, evidence, config, run, framework='net10.0', runtime='10.0.12'):
    for variant in ['baseline', *FAILURES]:
        directory = evidence / f'session-busy-{mode}-{variant}'
        project, provenance = prepare(directory / 'source', variant, framework, runtime)
        (evidence / f'session-busy-provenance-{mode}-{variant}.json').write_text(json.dumps(provenance, indent=2) + '\n')
        publish = directory / 'publish'
        run(f'session-busy-publish-{mode}-{variant}', ['dotnet', 'publish', project, '-c', 'Release',
            '-r', rid, '-o', publish, '--self-contained', 'true' if mode == 'aot' else 'false',
            f'-p:PublishAot={str(mode == "aot").lower()}', '--configfile', config])
        execute = ['dotnet', '--fx-version', runtime, publish / 'SessionBusy.dll'] if mode == 'jit' else [publish / ('SessionBusy.exe' if os.name == 'nt' else 'SessionBusy')]
        for profile in ['full', 'truncated'] if variant == 'baseline' else ['full']:
            expected = f'PASS {mode} source-linked Session BUSY {profile}:' if variant == 'baseline' else FAILURES[variant]
            run(f'session-busy-run-{mode}-{variant}-{profile}', [*execute, mode, profile],
                expected=int(variant != 'baseline'), contains=[expected])
