#!/usr/bin/env python3
"""Транскрипция курса Глеба Михайлова: mp4 -> wav (afconvert) -> faster-whisper."""
import subprocess, re, sys, time
from pathlib import Path
from faster_whisper import WhisperModel

ROOT = Path('/Users/user/Desktop/курсы/статистика и АБ/[SW.BAND] [Глеб Михайлов] АБ-тесты с Глебом Михайловым (2023)')
OUT = Path('/Users/user/.zcode/workspace/default/materials/transcripts')
TMP = Path('/tmp/gm_audio')
LOG = OUT / '_progress.log'
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

def log(msg):
    with open(LOG, 'a') as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

# группируем mp4 по урокам (папкам)
lessons = {}
for mp4 in sorted(ROOT.rglob('*.mp4')):
    rel = mp4.relative_to(ROOT)
    lesson_dir = rel.parent
    lessons.setdefault(lesson_dir, []).append(mp4)

def part_key(p):
    m = re.search(r'(\d+)\.(\d+)\.(\d+)\.mp4$', p.name)
    return (int(m.group(3)) if m else 999, p.name) if m else (999, p.name)

log(f"start: {len(lessons)} lessons, {sum(len(v) for v in lessons.values())} videos")
model = WhisperModel("small", device="cpu", compute_type="int8")
log("model loaded")

done = 0
for lesson_dir, files in lessons.items():
    name = re.sub(r'^\[SW\.BAND\]\s*', '', lesson_dir.name)
    name = re.sub(r'[^\w\.\- ]', '', name).strip().replace('. ', ' ').rstrip('.')
    safe = re.sub(r'\s+', '-', name) or 'untitled'
    out_md = OUT / f"ГМ-{safe}.md"
    if out_md.exists():
        done += 1
        continue
    parts = sorted(files, key=part_key)
    texts = []
    for mp4 in parts:
        wav = TMP / (mp4.stem + '.wav')
        try:
            r = subprocess.run(['afconvert', '-f', 'WAVE', '-d', 'LEI16@16000', '-c', '1',
                                str(mp4), str(wav)], capture_output=True, timeout=300)
            if r.returncode != 0 or not wav.exists():
                log(f"AFCONVERT FAIL {mp4.name}: {r.stderr.decode()[:200]}")
                continue
            segments, info = model.transcribe(str(wav), language="ru", vad_filter=True)
            t = " ".join(s.text.strip() for s in segments)
            texts.append(f"## {mp4.stem}\n\n{t}\n")
            log(f"ok {lesson_dir.name}/{mp4.name} ({info.duration:.0f}s)")
        except Exception as e:
            log(f"ERROR {mp4.name}: {e}")
        finally:
            wav.unlink(missing_ok=True)
    if texts:
        out_md.write_text(
            f"# Глеб Михайлов — {name}\n\n> Автотранскрипция (faster-whisper small), "
            f"{len(parts)} частей. Исходники: `…/{lesson_dir}`\n\n" + "\n".join(texts),
            encoding='utf-8')
    done += 1
    log(f"lesson done {done}/{len(lessons)}: {out_md.name}")

log("ALL DONE")
