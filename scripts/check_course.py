"""Проверка структуры, синтаксиса, ссылок и (опционально) всех демонстраций."""
from pathlib import Path
import argparse
import ast
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.avif', '.ico', '.bmp', '.tif', '.tiff'}
# These are a dated audit and archived authoring prompts, not current instructions.
HISTORICAL = ('docs/archive/', 'docs/maintenance/COURSE_REVIEW.md')


def visible(path, root):
    return not any(part.startswith('.') or part == '__pycache__' for part in path.relative_to(root).parts)


def without_code(text):
    return re.sub(r'```.*?```|~~~.*?~~~', '', text, flags=re.S)


def check(root=ROOT):
    errors = []
    for folder in ('course', 'skills', 'scripts', 'tests'):
        for path in (root / folder).rglob('*.py'):
            if visible(path, root):
                try:
                    ast.parse(path.read_text(), filename=str(path))
                except SyntaxError as exc:
                    errors.append(str(exc))
    markdown = list(root.glob('*.md'))
    assets = list(root.iterdir())
    for folder in ('course', 'skills', 'obsidian-vault', 'docs', 'scripts'):
        markdown.extend((root / folder).rglob('*.md'))
        assets.extend((root / folder).rglob('*'))
    for path in assets:
        if path.is_file() and visible(path, root) and path.suffix.lower() in IMAGE_SUFFIXES:
            if 'images' not in path.relative_to(root).parts[:-1]:
                errors.append(f'{path.relative_to(root)}: изображение должно лежать в каталоге images/')
    vault = root / 'obsidian-vault'
    vault_files = [path for path in vault.rglob('*') if path.is_file() and visible(path, root)]
    wiki_names = {path.name for path in vault_files} | {path.stem for path in vault_files}
    for path in markdown:
        relative = path.relative_to(root).as_posix()
        if not visible(path, root) or relative.startswith(HISTORICAL):
            continue
        content = without_code(path.read_text())
        for match in re.finditer(r'\]\((<[^>]+>|[^\n)]+)\)', content):
            target = match.group(1).strip().strip('<>')
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                continue
            target = unquote(target.split('#')[0])
            if not (path.parent / target).exists():
                errors.append(f'{relative}: {target}')
        if vault in path.parents:
            for match in re.finditer(r'\[\[([^\]|]+)', content):
                target = match.group(1).split('#')[0]
                if target and target not in wiki_names and not (vault / target).exists() and not (vault / (target + '.md')).exists():
                    errors.append(f'{relative}: [[{target}]]')
    for old in ('course/content', 'course/visuals'):
        if (root / old).exists():
            errors.append(f'{old}: устаревший каталог; используются course/modules и images/')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-practices', action='store_true')
    args = parser.parse_args()
    errors = check()
    if errors:
        print('\n'.join(errors))
        raise SystemExit(1)
    print('Структура, синтаксис Python, локальные Markdown- и Obsidian-ссылки: OK')
    if args.run_practices:
        practices = sorted((ROOT / 'course/modules').glob('M*/practice/practice_*.py'))
        if not practices:
            raise SystemExit('Не найдены практики course/modules/M*/practice/practice_*.py')
        logs = Path(tempfile.mkdtemp(prefix='stats-ab-validation-'))
        env = os.environ.copy()
        env['MPLCONFIGDIR'] = str(logs / 'matplotlib')
        env['XDG_CACHE_HOME'] = str(logs / 'cache')
        report = {'python': sys.version, 'packages': {}, 'practices': [], 'logs': str(logs)}
        report_path = ROOT / 'build/validation-results.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        for name in ('numpy', 'scipy', 'pandas', 'matplotlib', 'statsmodels', 'pymc', 'arviz'):
            try:
                report['packages'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                report['packages'][name] = None
        for path in practices:
            start = time.monotonic()
            with (logs / (path.stem + '.log')).open('w') as output:
                try:
                    result = subprocess.run([sys.executable, str(path)], cwd=ROOT, stdout=output,
                                            stderr=subprocess.STDOUT, env=env, timeout=1800)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = 'timeout'
            report['practices'].append({'path': str(path.relative_to(ROOT)), 'exit_code': code,
                                       'seconds': round(time.monotonic() - start, 2)})
            print(path.stem, 'OK' if code == 0 else f'FAIL ({code})', flush=True)
            if code != 0:
                errors.append(str(path))
            report_path.write_text(json.dumps(report, indent=2))
        print('Отчёт:', report_path)
        print('Журналы:', logs)
        if errors:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
