"""Archive the selected BIG IDEAs 1.1.3 modalities without changing old data.

Keeps an independent versioned copy of CGM, HR, food and demographics. All
payloads must match the official HTTPS checksum index; unchanged bytes are
copied locally. This script never integrates sources into the training corpus.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def index(path):
    output = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        if len(digest) != 64 or '..' in Path(name).parts or Path(name).is_absolute():
            raise ValueError('Invalid official checksum entry')
        output[name] = digest
    return output


def fetch(url, path, expected=None):
    if path.exists():
        if expected and sha(path) != expected:
            raise ValueError(f'Preserved mismatched existing target: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    subprocess.run(['curl.exe', '--noproxy', '*', '--proxy', '', '--fail',
                    '--location', '--silent', '--show-error', '--retry', '3',
                    '--connect-timeout', '20', '--max-time', '600',
                    '--continue-at', '-', '--output', str(part), url], check=True)
    if expected and sha(part) != expected:
        raise ValueError(f'Official checksum mismatch: {path}')
    part.rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    assert root.name == 'CGM_FM' and (root / 'AGENTS.md').is_file()
    base = root / 'datasets/BIG_IDEAs/1.1.3/results'
    audit = root / 'reports/audits/bigideas_1_1_3_2026-11-09'
    old_manifest = root / 'datasets/BIG_IDEAs/1.1.2/results/source_manifest.json'
    old_linkage = root / 'reports/audits/multimodal_download_2026-11-09/bigideas/existing_cgm_linkage.json'
    old_rows = json.loads(old_manifest.read_text(encoding='utf-8'))['files']
    old_rows += json.loads(old_linkage.read_text(encoding='utf-8'))['files']
    old_paths = {r['official_path']: root / r['local_path'] for r in old_rows}

    before_path = audit / 'preservation_before.json'
    if not before_path.exists():
        protected = set()
        for directory in ['datasets/BIG_IDEAs', 'data',
                          'code/CGM-JEPA/Dataset_Open',
                          'reports/audits/multimodal_pairing_2026-11-09/bigideas',
                          'reports/audits/multimodal_download_2026-11-09/bigideas']:
            for path in (root / directory).rglob('*'):
                if path.is_file() and not path.is_relative_to(base.parent):
                    protected.add(path)
        write_json(before_path, [{'path': p.relative_to(root).as_posix(),
                                 'bytes': p.stat().st_size, 'sha256': sha(p)}
                                for p in sorted(protected)])

    url = 'https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3/'
    page = 'https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/'
    fetch(url + 'SHA256SUMS.txt', base / 'metadata/SHA256SUMS.txt')
    entries = index(base / 'metadata/SHA256SUMS.txt')
    old_entries = index(old_manifest.parent / 'metadata/SHA256SUMS.txt')
    fetch(page, base / 'metadata/source_page.html')
    fetch(url + 'LICENSE.txt', base / 'metadata/LICENSE.txt', entries['LICENSE.txt'])
    diff = [{'path': p, 'old_sha256': old_entries.get(p), 'new_sha256': entries.get(p),
             'change': 'added' if p not in old_entries else 'removed' if p not in entries else 'changed'}
            for p in sorted(set(entries) | set(old_entries)) if entries.get(p) != old_entries.get(p)]
    write_json(audit / 'official_version_diff.json', {
        'old_version': '1.1.2', 'new_version': '1.1.3',
        'old_entries': len(old_entries), 'new_entries': len(entries), 'changes': diff})
    wanted = [f'{s:03d}/{kind}_{s:03d}.csv' for s in range(1, 17)
              for kind in ['Dexcom', 'HR', 'Food_Log']] + ['Demographics.csv']
    assert len(wanted) == 49 and all(p in entries for p in wanted)

    def acquire(name):
        target = base / 'source' / name
        expected = entries[name]
        old = old_paths.get(name)
        unchanged = old is not None and old.is_file() and sha(old) == expected
        action = 'copied_verified_old_file' if unchanged else 'downloaded_official_new_file'
        if target.exists():
            if sha(target) != expected:
                raise ValueError(f'Existing target mismatch: {target}')
        elif unchanged:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, target)
        else:
            fetch(url + name, target, expected)
        assert sha(target) == expected
        modality = ('glucose' if '/Dexcom_' in name else 'heart_rate' if '/HR_' in name
                    else 'food_log' if '/Food_Log_' in name else 'demographics')
        return {'official_path': name, 'url': url + name,
                'subject': name.split('/')[0] if '/' in name else None,
                'modality': modality, 'local_path': target.relative_to(root).as_posix(),
                'bytes': target.stat().st_size, 'official_sha256': expected,
                'local_sha256': sha(target), 'official_sha256_verified': True,
                'action': action,
                'prior_local_path': old.relative_to(root).as_posix() if old else None}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(acquire, wanted))
    manifest = {'dataset': 'BIG IDEAs', 'version': '1.1.3', 'source_page': page,
                'doi': '10.13026/aw6y-fc44', 'license': 'Open Data Commons Attribution License v1.0',
                'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'scope': '16 CGM, 16 HR, 16 Food_Log and demographics; no ACC/BVP/EDA/TEMP/IBI.',
                'network_route': 'direct_explicit_no_proxy', 'files': rows,
                'selected_files': len(rows), 'selected_bytes': sum(r['bytes'] for r in rows),
                'copied_unchanged_files': sum(r['action'].startswith('copied') for r in rows),
                'downloaded_changed_files': sum(r['action'].startswith('downloaded') for r in rows),
                'metadata': [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                              'sha256': sha(p)} for p in sorted((base / 'metadata').iterdir())],
                'training_changed': False, 'old_version_preserved': True,
                'checksum_limit': 'Verified against official HTTPS SHA256SUMS; index has no independent signature.'}
    write_json(base / 'source_manifest.json', manifest)
    write_json(audit / 'source_manifest.json', manifest)
    before = json.loads(before_path.read_text(encoding='utf-8'))
    failed = [r['path'] for r in before if not (root / r['path']).is_file()
              or (root / r['path']).stat().st_size != r['bytes']
              or sha(root / r['path']) != r['sha256']]
    preservation = {'checked_files': len(before), 'unchanged_files': len(before) - len(failed),
                    'changed_or_missing': failed, 'training_changed': False}
    write_json(audit / 'preservation_check.json', preservation)
    assert not failed, failed
    print(json.dumps({'files': len(rows), 'bytes': manifest['selected_bytes'],
                      'copied': manifest['copied_unchanged_files'],
                      'downloaded': manifest['downloaded_changed_files'],
                      'official_changes': diff, 'preservation': preservation}, ensure_ascii=False))


if __name__ == '__main__':
    main()
