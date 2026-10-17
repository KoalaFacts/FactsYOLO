# SPDX-License-Identifier: Apache-2.0
"""Detached preview consumer preparation."""
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

SETTINGS = json.loads(Path(__file__).with_name('sdk.json').read_text())


def preview_command(args, dotnet):
    command = list(map(str, args))
    # Windows resolves a bare executable before applying the child env PATH.
    if command[0] == 'dotnet':
        command[0] = str(dotnet)
    return command


def validate_training_report(report, backend, commit):
    if report.get('passed') is not True or report.get('source_commit') != commit or report.get('steps') != 2000 or report.get('test_seed') != 909:
        raise ValueError('require passed current-source Burn/Candle training')
    expected = {'burn': 'burn-flex-0.22.0', 'candle': 'candle-cpu-0.11.0'}
    if report.get('backend') != expected[backend]:
        raise ValueError('training backend differs from requested model')


def prepare_project(source, destination, repository):
    project_name = source.name + '.csproj'
    tree = ET.parse(source / project_name)
    framework = tree.find('.//TargetFramework')
    runtime = tree.find('.//RuntimeFrameworkVersion')
    if framework is None or framework.text != 'net10.0' or runtime is None or runtime.text != '10.0.12':
        raise ValueError('stable consumer baseline changed; review preview retargeting')
    framework.text = SETTINGS['framework']
    runtime.text = SETTINGS['runtime']
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / 'Program.cs', destination / 'Program.cs')
    if source.name == 'Ownership':
        expected = {repository / 'sdk/dotnet/FactsYOLO' / name for name in ['Handles.cs', 'Native.cs', 'NativeTypes.cs']}
        actual = {(source / item.attrib['Include']).resolve() for item in tree.findall('.//Compile')}
        if actual != {path.resolve() for path in expected}:
            raise ValueError('ownership source inventory changed')
        (destination / 'Sdk').mkdir()
        for path in expected:
            shutil.copyfile(path, destination / 'Sdk' / path.name)
        for group in tree.findall('.//ItemGroup'):
            for item in list(group):
                if item.tag == 'Compile':
                    group.remove(item)
    elif source.name != 'Consumer' or tree.findall('.//Compile') or tree.findall('.//ProjectReference'):
        raise ValueError('expected package-only public consumer')
    tree.write(destination / project_name, encoding='utf-8', xml_declaration=False)
    return destination / project_name
