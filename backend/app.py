import os
import io
import time
import math
import random
from typing import List, Optional, Tuple, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ──────────────────────────────────────────────────────────────────────────────
# 프로젝트 경로들
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_harmonizator")
PATTERN_DIR = os.path.join(BASE_DIR, "accomp_patterns")
CHORD_CSV = os.path.join(BASE_DIR, "chord.CSV")
PROG_CSV = os.path.join(BASE_DIR, "prog16.CSV")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# harmonizator_keyed.py import (수정 없이 사용)
# ──────────────────────────────────────────────────────────────────────────────
import sys
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    import harmonizator_keyed as HK
except Exception as e:
    raise RuntimeError(f"Failed to import harmonizator_keyed.py: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI & CORS
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="HarmonyCamp Backend", version="0.1.0")

# 필요 시 환경변수로 추가 오리진 설정 가능 (쉼표로 구분)
extra_origins = os.environ.get("CORS_ORIGINS", "").strip()
ALLOW_ORIGINS = ["https://golbingoblin.github.io", "http://localhost:5500", "http://127.0.0.1:5500"]
if extra_origins:
    ALLOW_ORIGINS += [o.strip() for o in extra_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 생성물(static) 서빙 (다운로드 링크 제공)
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

# ──────────────────────────────────────────────────────────────────────────────
# 전역 로드(앱 시작 시 1회)
# ──────────────────────────────────────────────────────────────────────────────
try:
    PC_NAMES, CHORD_TABLE = HK.load_chord_table(CHORD_CSV)
    PROGRESSIONS = HK.load_prog16(PROG_CSV)
except Exception as e:
    # CSV 없으면 서버는 떠 있지만 analyze 시 에러 반환
    PC_NAMES, CHORD_TABLE, PROGRESSIONS = [], {}, []
    print("[WARN] Failed to load chord/prog CSVs at startup:", e)

# ──────────────────────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────────────────────
def _pick_random_mid_in(folder: str) -> Optional[str]:
    cands = []
    for ext in ("*.mid", "*.midi"):
        cands.extend([os.path.join(folder, x) for x in os.listdir(folder) if x.lower().endswith(ext[1:])])
    return random.choice(cands) if cands else None

def _limit_melody_to_bars(notes: List[HK.Note], max_bars: int, beats_per_bar: float = HK.BAR_BEATS) -> List[HK.Note]:
    if max_bars <= 0:
        return notes
    max_beats = max_bars * beats_per_bar
    out: List[HK.Note] = []
    for n in notes:
        if n.start_beat >= max_beats:
            continue
        end = min(n.end_beat, max_beats)
        if end - n.start_beat > HK.EPS_BEAT:
            out.append(HK.Note(n.start_beat, end, n.pitch, n.velocity))
    return out

def _make_segments_from_progression_labels(p: HK.Progression, block_index: int) -> List[dict]:
    """프론트에서 코드 변환지점 시각화용. 16분 그리드 기준 세그먼트 → 비트 시간."""
    segs = HK.chord_segments(p.seq)
    out = []
    for st_step, en_step, tok in segs:
        st_b = block_index * HK.BLOCK_BEATS + st_step * HK.SIXTEENTH
        en_b = block_index * HK.BLOCK_BEATS + en_step * HK.SIXTEENTH
        out.append({"start_beat": st_b, "end_beat": en_b, "token": tok or ""})
    return out

# ──────────────────────────────────────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/list-genres")
def list_genres():
    if not os.path.isdir(PATTERN_DIR):
        return {"genres": []}
    genres = []
    for d in sorted(os.listdir(PATTERN_DIR)):
        full = os.path.join(PATTERN_DIR, d)
        if not os.path.isdir(full):
            continue
        if os.path.isdir(os.path.join(full, "bass")) and os.path.isdir(os.path.join(full, "comping")):
            genres.append(d)
    return {"genres": genres}

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    genre: str = Form(...),
    bpm: Optional[float] = Form(None),
    max_bars: int = Form(16),
):
    # 사전 체크
    if not CHORD_TABLE or not PROGRESSIONS:
        raise HTTPException(status_code=500, detail="chord.CSV / prog16.CSV not loaded on server.")

    # 패턴 로드
    gdir = os.path.join(PATTERN_DIR, genre)
    if not os.path.isdir(gdir):
        raise HTTPException(status_code=400, detail=f"Genre '{genre}' not found.")

    bass_dir = os.path.join(gdir, "bass")
    comp_dir = os.path.join(gdir, "comping")
    if not (os.path.isdir(bass_dir) and os.path.isdir(comp_dir)):
        raise HTTPException(status_code=400, detail=f"Genre '{genre}' must have 'bass' and 'comping' subfolders.")

    bass_mid = _pick_random_mid_in(bass_dir)
    comp_mid = _pick_random_mid_in(comp_dir)
    if not (bass_mid and comp_mid):
        raise HTTPException(status_code=400, detail=f"No pattern MIDI in '{genre}/bass' or '{genre}/comping'.")

    pat_bass = HK.extract_pattern_rhythm(bass_mid)
    pat_comp = HK.extract_pattern_rhythm(comp_mid)

    # 업로드 MIDI 읽기
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    # mido는 파일 경로가 필요 없고 메모리로도 가능하지만,
    # 기존 HK 함수는 파일 경로 기반이므로 임시파일로 저장 후 사용
    ts = time.strftime("%Y%m%d_%H%M%S")
    tmp_in = os.path.join(OUTPUT_DIR, f"uploaded_{ts}.mid")
    with open(tmp_in, "wb") as f:
        f.write(raw)

    # 멜로디 노트 추출
    melody, total_beats, first_tempo_us = HK.melody_notes_from_midifile(tmp_in)
    if not melody:
        raise HTTPException(status_code=400, detail="No melody notes found in uploaded MIDI.")

    # 마디 제한
    melody = _limit_melody_to_bars(melody, max_bars=max_bars, beats_per_bar=HK.BAR_BEATS)
    if not melody:
        raise HTTPException(status_code=400, detail="Melody truncated to 0 beats by max_bars limit.")

    # BPM 결정 (업로드 파일 템포 > 폼값)
    if bpm is None:
        bpm = HK.mido.tempo2bpm(first_tempo_us) if first_tempo_us else 120.0
    bpm = float(max(20.0, min(240.0, bpm)))

    # 키 추정 (프롬프트 없음)
    root_pc, mode = HK.guess_key_from_melody(melody)
    offset = HK.semitone_offset_for_key(root_pc, mode)   # C/Am → 원키 (+)
    semi_to_C = (12 - offset) % 12                       # melody → C/Am
    melody_C = HK.transpose_notes(melody, semi_to_C)

    key_label = f"Auto: {HK.NAMES_SHARP[root_pc]} {mode} (offset +{offset})"

    # 진행 선택 (C/Am에서)
    chosen_in_C = HK.choose_progressions_for_blocks(melody_C, PROGRESSIONS, CHORD_TABLE, PC_NAMES)

    # (중요) 진행 토널맵을 C에서 해석 → 숫자 이조로 원키에 맞춤(이후 CSV 의존 X)
    resolved_C = [HK.resolve_progression_from_C(p, CHORD_TABLE) for p in chosen_in_C]
    resolved_key = [HK.transpose_resolved_progression(rp, offset) for rp in resolved_C]

    # 리보이싱 & 렌더
    rendered_out, chosen_names = HK.assemble_from_resolved_progressions(melody, resolved_key, pat_bass, pat_comp)

    # MIDI 저장 (원키 리보이싱 결과 + 원본 멜로디)
    out_mid = os.path.join(OUTPUT_DIR, f"harmony_{ts}.mid")
    HK.export_mid(rendered_out, melody, bpm, out_mid, monitor_channel=HK.CH_MELODY_DEFAULT)

    # 디스플레이용 코드 구간(라벨): 오직 표시용으로만 CSV 라벨 이조 사용
    # (사운드는 숫자 이조 결과에 기반)
    # 한 블록=8마디=BLOCK_BEATS
    label_progs = [HK.transpose_progression_labels(p, offset, CHORD_TABLE) for p in chosen_in_C]
    segments = []
    for i, lp in enumerate(label_progs):
        segments += _make_segments_from_progression_labels(lp, i)

    # 총 길이(비트) – 블록 수 * BLOCK_BEATS
    duration_beats = len(resolved_key) * HK.BLOCK_BEATS

    # 다운로드 URL
    midi_url = f"/files/{os.path.basename(out_mid)}"

    return JSONResponse({
        "ok": True,
        "bpm": bpm,
        "key_label": key_label,
        "offset": offset,
        "duration_beats": duration_beats,
        "midi_url": midi_url,
        "chosen_progressions": chosen_names,   # 디버그/로그용
        "segments": segments,                  # 코드 변경 위치(시각화용)
        "genre": genre,
        "patterns": {
            "bass": os.path.basename(bass_mid),
            "comping": os.path.basename(comp_mid),
        },
    })
