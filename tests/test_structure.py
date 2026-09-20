"""Protect relocatable links and safe regeneration of the Obsidian mirror."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync = load_script('sync_course')
checker = load_script('check_course')


@pytest.fixture
def project(tmp_path):
    for i in range(1, 8):
        (tmp_path / f'course/modules/M{i}').mkdir(parents=True)
    for folder in ('course/lib', 'course/images', 'course/cases', 'docs', 'obsidian-vault/Концепции'):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / 'course/lib/foundations.py').write_text('ANSWER = 42\n')
    (tmp_path / 'course/PREREQUISITES.md').write_text('# Prerequisites\n')
    (tmp_path / 'course/cases/case.md').write_text('# Case\n')
    (tmp_path / 'course/modules/M1/lesson.md').write_text(
        '# Урок\n[База](../../PREREQUISITES.md)\n[Кейс](../../cases/case.md)\n'
    )
    (tmp_path / 'course/images/route.png').write_bytes(b'figure')
    (tmp_path / 'docs/setup.md').write_text('# Setup\n')
    (tmp_path / 'course/plan.md').write_text('# План\n[Урок](modules/M1/lesson.md)\n')
    (tmp_path / 'course/NAVIGATION.md').write_text(
        '[Урок](modules/M1/lesson.md#first)\n'
        '[Настройка](../docs/setup.md)\n'
        '![Маршрут](images/route.png)\n'
        '[Сайт](https://example.org) [Здесь](#top)\n'
    )
    return tmp_path


def test_mirror_keeps_helpers_assets_and_navigation_targets(project):
    sync.synchronize(project)
    vault = project / 'obsidian-vault'
    assert (vault / 'lib/foundations.py').read_text() == 'ANSWER = 42\n'
    assert (vault / 'images/route.png').read_bytes() == b'figure'
    text = (vault / sync.NAVIGATION).read_text()
    assert '(../course/modules/M1/lesson.md#first)' in text
    assert '(../docs/setup.md)' in text
    assert '(../course/images/route.png)' in text
    assert '(https://example.org) [Здесь](#top)' in text
    lesson = (vault / 'Уроки/M1/lesson.md').read_text()
    assert '(../../../course/PREREQUISITES.md)' in lesson
    assert '(../Кейсы/case.md)' in lesson
    assert checker.check(project) == []
    assert not any(sync.synchronize(project, check=True).values())


def test_removed_source_removes_only_tracked_copy_and_preserves_personal_notes(project):
    vault = project / 'obsidian-vault'
    sync.synchronize(project)
    personal = vault / 'Уроки/M1/мои заметки.md'
    personal.write_text('Мой разбор\n')
    (project / 'course/modules/M1/lesson.md').unlink()
    result = sync.synchronize(project)
    assert result['stale'] == ['Уроки/M1/lesson.md']
    assert not (vault / 'Уроки/M1/lesson.md').exists()
    assert personal.read_text() == 'Мой разбор\n'


def test_check_reports_stale_and_differences_without_writing(project):
    sync.synchronize(project)
    source = project / 'course/modules/M1/lesson.md'
    source.rename(source.with_name('new.md'))
    image = project / 'obsidian-vault/images/route.png'
    image.write_bytes(b'old figure')
    manifest = project / 'obsidian-vault' / sync.MANIFEST
    before = manifest.read_bytes()
    result = sync.synchronize(project, check=True)
    assert 'Уроки/M1/new.md' in result['changed']
    assert 'images/route.png' in result['changed']
    assert result['stale'] == ['Уроки/M1/lesson.md']
    assert (project / 'obsidian-vault/Уроки/M1/lesson.md').exists()
    assert not (project / 'obsidian-vault/Уроки/M1/new.md').exists()
    assert image.read_bytes() == b'old figure'
    assert manifest.read_bytes() == before


def test_manually_changed_stale_copy_is_preserved(project):
    sync.synchronize(project)
    copy = project / 'obsidian-vault/Уроки/M1/lesson.md'
    copy.write_text('Мои правки в копии\n')
    (project / 'course/modules/M1/lesson.md').unlink()
    result = sync.synchronize(project)
    assert result['preserved'] == ['Уроки/M1/lesson.md']
    assert copy.read_text() == 'Мои правки в копии\n'
    assert sync.synchronize(project, check=True)['preserved']


@pytest.mark.parametrize('unsafe', ['../private.md', '/tmp/private.md', 'Концепции/личное.md'])
def test_manifest_cannot_delete_outside_generated_scope(project, unsafe):
    (project / 'obsidian-vault' / sync.MANIFEST).write_text(json.dumps({'files': {unsafe: 'hash'}}))
    with pytest.raises(ValueError, match='вне генерируемой'):
        sync.synchronize(project)


def test_validation_finds_stray_images_and_broken_links(project):
    sync.synchronize(project)
    (project / 'course/modules/M1/stray.png').write_bytes(b'figure')
    (project / 'README.md').write_text('[Сломанная ссылка](course/missing.md)\n')
    errors = checker.check(project)
    assert any('stray.png: изображение должно' in error for error in errors)
    assert any('README.md: course/missing.md' in error for error in errors)


def test_navigation_link_to_spaced_filename_is_valid_commonmark(project):
    source = project / 'course/modules/M1/README.md'
    source.write_text('[Маршрут](../../NAVIGATION.md#start)\n')
    sync.synchronize(project)
    mirrored = project / 'obsidian-vault/Уроки/M1/README.md'
    assert mirrored.read_text() == '[Маршрут](<../../Навигация по курсу.md#start>)\n'
    assert checker.check(project) == []


def test_plan_is_updated_from_canonical_source_with_working_mirror_links(project):
    sync.synchronize(project)
    source = project / 'course/plan.md'
    source.write_text('# Новый план\n[Урок](modules/M1/lesson.md)\n')
    assert sync.PLAN in sync.synchronize(project, check=True)['changed']
    sync.synchronize(project)
    mirrored = project / 'obsidian-vault' / sync.PLAN
    assert mirrored.read_text() == '# Новый план\n[Урок](M1/lesson.md)\n'
    assert checker.check(project) == []
