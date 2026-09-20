"""Обновить зеркало Obsidian из канонического курса; --check ничего не пишет.

Файлы вне manifest не удаляются. При удалении исходника его неизменённая
копия удаляется; изменённая вручную копия сохраняется с предупреждением.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = '.course-sync-manifest.json'
GENERATED_TREES = tuple(f'Уроки/M{i}/' for i in range(1, 8)) + ('Уроки/Кейсы/', 'lib/', 'images/')
NAVIGATION = 'Навигация по курсу.md'
PLAN = 'Уроки/plan-курса.md'


def generated_path(relative):
    """Manifest may name only generated files inside the vault."""
    path = PurePosixPath(relative)
    return (not path.is_absolute() and '..' not in path.parts and
            (relative in (NAVIGATION, PLAN) or relative.startswith(GENERATED_TREES)))


def source_trees(root=ROOT):
    """Canonical folders and their generated locations inside the vault."""
    vault = root / 'obsidian-vault'
    mappings = [(root / 'course/modules' / f'M{i}', vault / 'Уроки' / f'M{i}') for i in range(1, 8)]
    mappings += [(root / 'course/cases', vault / 'Уроки/Кейсы'),
                 (root / 'course/lib', vault / 'lib'),
                 (root / 'course/images', vault / 'images')]
    return mappings


def pairs(root=ROOT):
    """Sources and destinations, including practice assets and shared helpers."""
    for source, destination in source_trees(root):
        if not source.is_dir():
            raise FileNotFoundError(f'Нет канонического каталога: {source}')
        for src in sorted(source.rglob('*')):
            if src.is_file() and '__pycache__' not in src.parts and src.suffix != '.pyc':
                yield src, destination / src.relative_to(source)
    yield root / 'course/NAVIGATION.md', root / 'obsidian-vault' / NAVIGATION
    yield root / 'course/plan.md', root / 'obsidian-vault' / PLAN


def markdown_content(source, destination, mirrored_targets=()):
    """Resolve links in the mirror, falling back to the canonical source."""
    def rewrite(match):
        wrapped = match.group(1).startswith('<')
        target = match.group(1).strip('<>')
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith(('#', '/')):
            return match.group(0)
        local, separator, anchor = target.partition('#')
        resolved = (source.parent / unquote(local)).resolve()
        for canonical, mirrored in mirrored_targets:
            if resolved == canonical or canonical in resolved.parents:
                resolved = mirrored / resolved.relative_to(canonical)
                break
        relative = Path(os.path.relpath(resolved, destination.parent)).as_posix()
        if separator:
            relative += '#' + anchor
        # CommonMark requires angle brackets around link destinations with spaces.
        wrapped = wrapped or any(char.isspace() for char in relative)
        return '](' + ('<' + relative + '>' if wrapped else relative) + ')'
    return re.sub(r'\]\((<[^>]+>|[^\n)]+)\)', rewrite, source.read_text()).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def synchronize(root=ROOT, check=False):
    vault = root / 'obsidian-vault'
    manifest_path = vault / MANIFEST
    previous = json.loads(manifest_path.read_text())['files'] if manifest_path.exists() else {}
    if any(not generated_path(path) for path in previous):
        raise ValueError('Manifest содержит путь вне генерируемой части хранилища')
    expected = {}
    mirrored_targets = source_trees(root) + [(root / 'course/NAVIGATION.md', vault / NAVIGATION),
                                           (root / 'course/plan.md', vault / PLAN)]
    for source, destination in pairs(root):
        if source.suffix == '.md':
            # The main navigation intentionally links to the canonical course.
            targets = () if source.name == 'NAVIGATION.md' else mirrored_targets
            content = markdown_content(source, destination, targets)
        else:
            content = source.read_bytes()
        expected[destination.relative_to(vault).as_posix()] = content
    changed, stale, preserved = [], [], []
    for relative, content in expected.items():
        destination = vault / relative
        if not destination.is_file() or destination.read_bytes() != content:
            changed.append(relative)
            if not check:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
    for relative in previous.keys() - expected.keys():
        destination = vault / relative
        if destination.is_file():
            if digest(destination.read_bytes()) != previous[relative]:
                preserved.append(relative)
                continue
            stale.append(relative)
            if not check:
                destination.unlink()
                parent = destination.parent
                while parent != vault and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
    files = {relative: digest(content) for relative, content in sorted(expected.items())}
    # Keep tracking a manually changed stale copy until someone resolves it.
    files.update({relative: previous[relative] for relative in preserved})
    manifest = json.dumps({'version': 1, 'files': files}, ensure_ascii=False, indent=2) + '\n'
    manifest_changed = not manifest_path.exists() or manifest_path.read_text() != manifest
    if not check:
        vault.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest)
    return {'changed': changed, 'stale': stale, 'preserved': preserved, 'manifest_changed': manifest_changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = synchronize(check=args.check)
    print(f"Файлы: обновить {len(result['changed'])}, удалить устаревшие {len(result['stale'])}"
          if args.check else f"Обновлено {len(result['changed'])}, удалено устаревших {len(result['stale'])}")
    if result['preserved']:
        print('Сохранены изменённые вручную копии удалённых исходников:')
        print('\n'.join(result['preserved']))
    if (args.check and any(result.values())) or result['preserved']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
