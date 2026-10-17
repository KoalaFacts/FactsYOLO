# SPDX-License-Identifier: Apache-2.0
import fnmatch
import hashlib
import io
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

from compat import prepare_project, preview_command, validate_training_report
from install_sdk import unpack_verified

ROOT = Path(__file__).resolve().parents[2]


class PreviewCompatibility(unittest.TestCase):
    def test_linked_implicit_source_directory_is_not_silently_omitted(self):
        for kind in ['Consumer', 'Ownership']:
            for target_exists in [True, False]:
                with self.subTest(kind=kind, target_exists=target_exists), tempfile.TemporaryDirectory() as temp:
                    source = Path(temp) / kind
                    shutil.copytree(ROOT / 'tests/sdk-install' / kind, source,
                                    ignore=shutil.ignore_patterns('bin', 'obj'))
                    target = Path(temp) / 'shared'
                    if target_exists:
                        target.mkdir()
                        (target / 'Extra.cs').write_text('internal class Extra {}')
                    (source / 'Linked').symlink_to(target, target_is_directory=True)
                    with self.assertRaisesRegex(ValueError, 'implicit source inventory'):
                        prepare_project(source, Path(temp) / 'copy', ROOT)

    def test_extra_implicitly_compiled_source_is_not_silently_omitted(self):
        for kind in ['Consumer', 'Ownership']:
            for extra in ['Extra.cs', 'nested/Extra.cs', 'Extra.CS']:
                with self.subTest(kind=kind, extra=extra), tempfile.TemporaryDirectory() as temp:
                    source = Path(temp) / kind
                    shutil.copytree(ROOT / 'tests/sdk-install' / kind, source,
                                    ignore=shutil.ignore_patterns('bin', 'obj'))
                    added = source / extra; added.parent.mkdir(exist_ok=True)
                    added.write_text('internal class Extra {}')
                    with self.assertRaisesRegex(ValueError, 'implicit source inventory'):
                        prepare_project(source, Path(temp) / 'copy', ROOT)

    def test_generated_bin_and_obj_sources_are_not_consumer_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'Consumer'
            shutil.copytree(ROOT / 'tests/sdk-install/Consumer', source,
                            ignore=shutil.ignore_patterns('bin', 'obj'))
            for generated in ['bin', 'obj']:
                (source / generated).mkdir()
                (source / generated / 'Generated.cs').write_text('generated output')
            prepare_project(source, Path(temp) / 'copy', ROOT)
            self.assertFalse((Path(temp) / 'copy/obj').exists())

    def test_package_and_helper_changes_trigger_preview_workflow(self):
        workflow = (ROOT / '.github/workflows/dotnet11-compat.yml').read_text()
        patterns = re.findall(r"^      - '([^']+)'$", workflow, re.MULTILINE)
        inputs = ['sdk/dotnet/FactsYOLO/Model.cs', 'sdk/dotnet/build_package.py',
                  'sdk/dotnet/native_assets.py', 'sdk/dotnet/global.json',
                  'tests/sdk-install/test_loader_fixtures.py', 'tests/sdk-install/test_packaging.py',
                  'crates/facts-yolo/src/lib.rs', 'include/facts_yolo.h',
                  'experiments/cpu-detector/model/src/lib.rs', 'tests/c/public-symbols.txt',
                  'tests/compat/td1_v1_vectors.rs', 'xtask/Cargo.toml',
                  'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.gitattributes', 'LICENSE', 'NOTICE']
        for name in inputs:
            with self.subTest(path=name):
                self.assertTrue(any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns), name)
        self.assertFalse(any(fnmatch.fnmatchcase('docs/unrelated.md', pattern) for pattern in patterns))

    def test_loader_dotnet_uses_verified_absolute_executable(self):
        installed = Path('/verified/sdk/dotnet.exe')
        self.assertEqual(preview_command(['dotnet', 'Consumer.dll'], installed), [str(installed), 'Consumer.dll'])
        self.assertEqual(preview_command([installed, '--version'], installed), [str(installed), '--version'])
        self.assertEqual(preview_command(['Consumer.exe', 'model'], installed), ['Consumer.exe', 'model'])

    def test_copied_burn_report_cannot_satisfy_candle_training(self):
        report = {'passed': True, 'source_commit': 'head', 'steps': 2000, 'test_seed': 909, 'backend': 'burn-flex-0.22.0'}
        validate_training_report(report, 'burn', 'head')
        with self.assertRaisesRegex(ValueError, 'backend'):
            validate_training_report(report, 'candle', 'head')
        report['backend'] = 'candle-cpu-0.11.0'
        validate_training_report(report, 'candle', 'head')
        with self.assertRaisesRegex(ValueError, 'backend'):
            validate_training_report(report, 'burn', 'head')

    def test_public_consumer_keeps_package_reference_and_program(self):
        with tempfile.TemporaryDirectory() as temp:
            source = ROOT / 'tests/sdk-install/Consumer'
            dest = Path(temp) / 'consumer'
            prepare_project(source, dest, ROOT)
            self.assertTrue((dest / 'Consumer.csproj').is_file())
            tree = ET.parse(dest / 'Consumer.csproj')
            self.assertEqual(tree.findtext('.//TargetFramework'), 'net11.0')
            self.assertEqual(tree.findtext('.//RuntimeFrameworkVersion'), '11.0.0-rc.2.26504.105')
            self.assertEqual(tree.findtext('.//RollForward'), 'Disable')
            self.assertEqual(tree.findtext('.//TreatWarningsAsErrors'), 'true')
            self.assertEqual((source / 'Program.cs').read_bytes(), (dest / 'Program.cs').read_bytes())
            self.assertEqual(len(tree.findall('.//PackageReference')), 1)
            self.assertFalse(tree.findall('.//ProjectReference'))
            self.assertFalse(tree.findall('.//Compile'))

    def test_ownership_probe_copies_exact_existing_lifetime_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source = ROOT / 'tests/sdk-install/Ownership'
            dest = Path(temp) / 'ownership'
            prepare_project(source, dest, ROOT)
            self.assertTrue((dest / 'Ownership.csproj').is_file())
            self.assertEqual((source / 'Program.cs').read_bytes(), (dest / 'Program.cs').read_bytes())
            for name in ['Handles.cs', 'Native.cs', 'NativeTypes.cs']:
                self.assertEqual((dest / 'Sdk' / name).read_bytes(), (ROOT / 'sdk/dotnet/FactsYOLO' / name).read_bytes())
            self.assertFalse(ET.parse(dest / 'Ownership.csproj').findall('.//Compile'))

    def test_changed_baseline_framework_is_not_silently_retargeted(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'Consumer'; source.mkdir()
            original = (ROOT / 'tests/sdk-install/Consumer/Consumer.csproj').read_text()
            (source / 'Consumer.csproj').write_text(original.replace('net10.0', 'net9.0'))
            (source / 'Program.cs').write_text('')
            with self.assertRaisesRegex(ValueError, 'baseline'):
                prepare_project(source, Path(temp) / 'copy', ROOT)


class VerifiedArchive(unittest.TestCase):
    def make_archive(self, root, kind, name='sdk/version.txt'):
        path = root / ('sdk.zip' if kind == 'zip' else 'sdk.tar.gz')
        if kind == 'zip':
            with zipfile.ZipFile(path, 'w') as out:
                out.writestr(name, b'pinned sdk archive fixture')
        else:
            with tarfile.open(path, 'w:gz') as out:
                info = tarfile.TarInfo(name); info.size = 26
                out.addfile(info, io.BytesIO(b'pinned sdk archive fixture'))
        return path, hashlib.sha512(path.read_bytes()).hexdigest()

    def test_verified_zip_and_tar_extract_real_payload(self):
        for kind in ['zip', 'tar']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); archive, digest = self.make_archive(root, kind)
                dest = root / 'installed'
                unpack_verified(archive, dest, digest)
                self.assertTrue((dest / 'sdk/version.txt').is_file())
                self.assertEqual((dest / 'sdk/version.txt').read_bytes(), b'pinned sdk archive fixture')

    def test_wrong_digest_rejected_before_writing_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); archive, _ = self.make_archive(root, 'zip')
            dest = root / 'installed'
            with self.assertRaisesRegex(ValueError, 'SHA-512'):
                unpack_verified(archive, dest, '0' * 128)
            self.assertFalse(dest.exists())

    def test_traversal_rejected_for_both_archive_formats(self):
        for kind in ['zip', 'tar']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); archive, digest = self.make_archive(root, kind, '../escape')
                with self.assertRaisesRegex(ValueError, 'unsafe'):
                    unpack_verified(archive, root / 'installed', digest)
                self.assertFalse((root / 'escape').exists())

    def test_existing_sdk_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); archive, digest = self.make_archive(root, 'zip')
            dest = root / 'installed'; dest.mkdir()
            with self.assertRaises(FileExistsError):
                unpack_verified(archive, dest, digest)


if __name__ == '__main__':
    unittest.main()
