# SPDX-License-Identifier: Apache-2.0
"""Verified preview SDK installation."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import stat
import tarfile
import tempfile
import urllib.request
import zipfile

from compat import SETTINGS


def host_rid():
    if platform.machine().lower() not in ('x86_64', 'amd64') or platform.system() not in ('Linux', 'Windows'):
        raise ValueError('preview gate requires actual Linux/Windows x64')
    return 'win-x64' if platform.system() == 'Windows' else 'linux-x64'


def unpack_verified(archive, destination, expected_hash):
    digest = hashlib.sha512()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError('SDK archive SHA-512 differs from pinned official metadata')
    if destination.exists():
        raise FileExistsError(destination)

    def check_paths(names):
        seen = set()
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or str(path) in seen:
                raise ValueError('unsafe SDK archive entry: ' + name)
            seen.add(str(path))

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            check_paths(bundle.namelist())
            if any(stat.S_ISLNK(info.external_attr >> 16) for info in bundle.infolist()):
                raise ValueError('unsafe SDK archive symbolic link')
            destination.mkdir(parents=True)
            bundle.extractall(destination)
    else:
        with tarfile.open(archive) as bundle:
            check_paths(item.name for item in bundle.getmembers())
            destination.mkdir(parents=True)
            bundle.extractall(destination, filter='data')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--archive', type=Path, help='reuse a local archive, still checking the pinned SHA-512')
    args = parser.parse_args()
    rid = host_rid()
    asset = SETTINGS['assets'][rid]
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix='factsyolo-dotnet11-download-') as temp:
        archive = args.archive
        if archive is None:
            archive = Path(temp) / 'sdk-archive'
            with urllib.request.urlopen(asset['url'], timeout=60) as response, archive.open('wb') as out:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    out.write(chunk)
        unpack_verified(archive, destination, asset['sha512'])
        receipt = {'sdk': SETTINGS['sdk'], 'runtime': SETTINGS['runtime'], 'rid': rid,
                   'url': asset['url'], 'sha512': asset['sha512'], 'archive_bytes': archive.stat().st_size}
        (destination / 'sdk-installation.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
