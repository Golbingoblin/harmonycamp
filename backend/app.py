# -*- coding: utf-8 -*-
# FastAPI backend that wraps harmonizator_keyed.py without modifying it.
import os, io, time, math, shutil, glob, re
from typing import List, Optional, Dict, Tuple
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

import mido
import pandas as pd

# ── import the existing script as a module ────────────────────────────────────
import importlib.util, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("hz", os.path.join(HERE, "harmonizator_keyed.py"))
hz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hz)  # type: ignore

OUTPUT_DIR = os.path.join(HERE, "output_harmonizator")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CORS ─────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ── load chord/prog at startup ───────────────────────────────────────────────
CSV_CHORD = os.path.join(HERE, "chord.CSV")
CSV_PROG  = os.path.join(HERE, "prog16.CSV")

try:
    PC_NAMES, CHORD_TABLE = hz.load_chord_table(CSV_CHORD)
    PROGS = hz.load_prog16(CSV_PROG)
except Exception as e:
    print("CSV load failed:", e)
    PC_NAMES, CHORD_TABLE, PROGS = hz.PC_NAMES_DEFAULT, {}, []

# ── genre listing (reads folders) ────────────────────────────────────────────
def list_genres() -> List[str]:
    base = os.path.join(HERE, "accomp_patterns")
    if not os.path.isdir(base): return []
    return sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

def pick_random_mid_in(folder: str) -> Optional[str]:
    cand = sorted(glob.glob(os.path.join(folder, '*.mid')) + glob.glob(os.path.join(folder, '*.midi')))
    return cand[0] if cand else None  # deterministic: first

def extract_patterns_for_genre(genre: str) -> Tuple[hz.PatternRhythm, hz.PatternRhythm]:
    folder_bass = os.path.join(HERE, "accomp_patterns", genre, "bass")
    folder_comp = os.path.join(HERE, "accomp_patterns", genre, "comping")
    if not (os.path.isdir(folder_bass) and os.path.isdir(folder_comp)):
        raise RuntimeError("accomp_patterns/<genre>/{bass,comping} not found")
    pb = pick_random_mid_in(folder_bass); pc = pick_random_mid_in(folder_comp)
    if not (pb and pc): raise RuntimeError("No pattern MIDIs in bass/comping folders.")
    return hz.extract_pattern_rhythm(pb), hz.extract_pattern_rhythm(pc)

# ── chord-name transpose rule (no csv re-lookup) ────────────────────────────
#  - We keep original token from PROG (e.g., 'Cmaj7/3') and rotate its root by +semi.
#  - Preference: keys with sharps → sharp names; flat keys → flat names. No double-accidentals.
NAME2PC = hz.NAME2PC
N_SHARP = hz.NAMES_SHARP
N_FLAT  = hz.NAMES_FLAT
ROOT_RE = re.compile(r'^([A-G](?:#|b)?)(.*)$')

SHARP_KEYS = {0,7,2,9,4,11,6}     # C,G,D,A,E,B,F#    (pc numbers)
FLAT_KEYS  = {5,10,3,8,1,6}       # F,Bb,Eb,Ab,Db,Gb

def split_base_inv(tok: str) -> Tuple[str, Optional[int]]:
    s = (tok or "").strip()
    if not s: return "", None
    if '/' in s:
        b, inv = s.split('/', 1)
        try: invn = int(inv)
        except: invn = None
        return b.strip(), invn
    return s, None

def transpose_token_rule(tok: str, semi: int, prefer_flat: bool=False) -> str:
    base, inv = split_base_inv(tok)
    m = ROOT_RE.match(base or "")
    if not m: return tok
    root, suffix = m.group(1), m.group(2)
    pc = NAME2PC.get(root)
    if pc is None: return tok
    npc = (pc + semi) % 12
    name = (N_FLAT if prefer_flat else N_SHARP)[npc] + (suffix or "")
    return f"{name}/{inv}" if inv else name

# ── build chord segments from progression seq (128 steps per 8 bars) ────────
def segments_from_seq(seq: List[Optional[str]]) -> List[Tuple[int,int,Optional[str]]]:
    segs = []
    cur, st = seq[0], 0
    for i in range(1, 128):
        if seq[i] != cur:
            segs.append((st, i, cur)); st = i; cur = seq[i]
    segs.append((st, 128, cur))
    return segs

# ── JSON models ──────────────────────────────────────────────────────────────
# AnalyzeResponse에 두 필드 추가
class AnalyzeResponse(BaseModel):
    bpm: float
    total_beats: float
    bars_shown: int
    bar_beats: float
    chords: list[dict]
    melody: list[dict]
    comping: list[dict]
    bass: list[dict]
    download_url: str
    key_label: str            # ← 추가
    chosen: list[str]         # ← 추가 (블록별 선택 진행 이름)

# ── endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/list-genres")
def api_genres():
    return {"genres": list_genres()}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    genre: str = Form("basic"),
    bpm_override: Optional[float] = Form(None),
    max_bars: int = Form(16),
    key_mode: str = Form("detect"),  # 'detect' | 'c' | 'manual'
    manual_key: Optional[str] = Form(None),  # e.g., "Gb major"
):
    # save upload to temp
    ts = time.strftime("%Y%m%d_%H%M%S")
    tmp_path = os.path.join(OUTPUT_DIR, f"_upload_{ts}.mid")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # melody & tempo
    melody, total_beats, first_tempo = hz.melody_notes_from_midifile(tmp_path)
    if not melody:
        raise RuntimeError("No melody notes found (non-drum).")
    bpm = bpm_override or (mido.tempo2bpm(first_tempo) if first_tempo else 120.0)

    # key offset
    if key_mode == "c":
        offset, key_label = 0, "C collection (no transpose)"
    elif key_mode == "manual" and manual_key:
        parsed = hz.parse_key_string(manual_key)
        if not parsed: raise RuntimeError(f"Cannot parse manual key: {manual_key}")
        root_pc, mode = parsed
        offset = hz.semitone_offset_for_key(root_pc, mode)
        key_label = f"Manual: {hz.NAMES_SHARP[root_pc]} {mode} (+{offset})"
    else:
        root_pc, mode = hz.guess_key_from_melody(melody)
        offset = hz.semitone_offset_for_key(root_pc, mode)
        key_label = f"Auto: {hz.NAMES_SHARP[root_pc]} {mode} (+{offset})"

    # melody -> C
    semi_to_C = (12 - offset) % 12
    melody_C = hz.transpose_notes(melody, semi_to_C)

    # choose progression per block in C
    chosen_C = hz.choose_progressions_for_blocks(melody_C, PROGS, CHORD_TABLE, PC_NAMES)

    # resolve tone-maps in C; transpose numerically to original key
    resolved_C   = [hz.resolve_progression_from_C(p, CHORD_TABLE) for p in chosen_C]
    resolved_key = [hz.transpose_resolved_progression(rp, offset) for rp in resolved_C]

    # patterns
    pat_bass, pat_comp = extract_patterns_for_genre(genre)

    # re-voice in ORIGINAL key
    rendered, chosen_names = hz.assemble_from_resolved_progressions(melody, resolved_key, pat_bass, pat_comp)

    # export MIDI (for download)
    out_mid = os.path.join(OUTPUT_DIR, f'harmonized_{os.path.basename(file.filename)}_{ts}.mid')
    hz.export_mid(rendered, melody, bpm, out_mid, hz.CH_MELODY_DEFAULT)

    # prepare JSON tracks
    def evs_by_role(role: str) -> List[Dict]:
        out = []
        for ev_on in [e for e in rendered.events if e.type=="on" and e.role==role]:
            off = next((x for x in rendered.events if x.type=="off" and x.role==role and x.time_beats>=ev_on.time_beats and x.note==ev_on.note), None)
            if off and off.time_beats>ev_on.time_beats:
                out.append({"start": ev_on.time_beats, "end": off.time_beats, "pitch": ev_on.note, "velocity": ev_on.velocity})
        return out

    comp = evs_by_role("comp")
    bass = evs_by_role("bass")
    melj = [{"start": n.start_beat, "end": n.end_beat, "pitch": n.pitch, "velocity": n.velocity} for n in melody]

    # chord lane (transpose C-labels by +offset, avoid double-accidentals pragmatically)
    prefer_flat = (offset in FLAT_KEYS)
    chords_json: List[Dict] = []
    BAR_BEATS = hz.BAR_BEATS
    SIX = hz.SIXTEENTH
    bars_to_show = max(1, min(max_bars, int(math.ceil(rendered.total_beats / BAR_BEATS))))
    # build labels per block from original PROG tokens
    for iblock, p in enumerate(chosen_C):
        segs = segments_from_seq(p.seq)   # in 16th steps of 8 bars
        base0 = iblock * (8.0 * BAR_BEATS)
        for st, en, tok in segs:
            if tok:
                lab = transpose_token_rule(tok, offset, prefer_flat=prefer_flat)
                st_b = base0 + st * SIX
                en_b = base0 + en * SIX
                chords_json.append({"start_beats": st_b, "end_beats": en_b, "label": lab})

    # trim to UI length
    hard_end = bars_to_show * BAR_BEATS
    def trim_notes(arr):
        out=[]
        for a in arr:
            st, en = a["start"], a["end"]
            if st >= hard_end: continue
            en = min(en, hard_end)
            if en - st > 1e-6:
                b = dict(a); b["end"] = en; out.append(b)
        return out

    melj = trim_notes(melj)
    comp = trim_notes(comp)
    bass = trim_notes(bass)
    chords_json = [c for c in chords_json if c["start_beats"] < hard_end]

    return AnalyzeResponse(
        bpm=float(bpm),
        total_beats=float(rendered.total_beats),
        bars_shown=int(bars_to_show),
        bar_beats=float(BAR_BEATS),
        chords=chords_json,
        melody=melj,
        comping=comp,
        bass=bass,
        download_url=f"/download/{os.path.basename(out_mid)}",
        key_label=key_label,           # ← 추가
        chosen=chosen_names,           # ← 추가
    )

@app.get("/download/{fname}")
def dl(fname: str):
    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.isfile(path):
        return JSONResponse({"error":"file not found"}, status_code=404)
    return FileResponse(path, media_type="audio/midi", filename=fname)
