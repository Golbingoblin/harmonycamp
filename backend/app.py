# app.py — HarmonyCamp backend (FastAPI)
# - Endpoints: /health, /genres, /list-genres (alias), /harmonize, /analyze (alias)
# - Uses existing harmonizator_keyed.py WITHOUT modifying it
# - Returns: key_label, chosen_progressions, midi_url (and optional audio_url stub)

import os
import io
import uuid
import glob
import random
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ===== Import user's existing engine (no modification) =====
# harmonizator_keyed.py must be in the same directory or importable
from harmonizator_keyed import (
    # data types
    Note, PatternRhythm, Progression, ChordInfo, RenderedSong,
    # loaders
    load_chord_table, load_prog16,
    # midi & analysis
    melody_notes_from_midifile,
    guess_key_from_melody, semitone_offset_for_key,
    transpose_notes,
    choose_progressions_for_blocks,
    resolve_progression_from_C, transpose_resolved_progression,
    assemble_from_resolved_progressions,
    export_mid,
    extract_pattern_rhythm,
    pick_random_mid_in,
    # constants
    NAMES_SHARP, BLOCK_BEATS, BAR_BEATS, CH_MELODY_DEFAULT
)

# ===== App =====
app = FastAPI(title="HarmonyCamp Backend", version="1.0.0")

# Allow CORS (GitHub Pages 등 외부 오리진)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요시 ["https://golbingoblin.github.io"] 로 좁히세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
os.makedirs(PUBLIC_DIR, exist_ok=True)

# Static mount for generated files
app.mount("/files", StaticFiles(directory=PUBLIC_DIR), name="files")

# ====== cache: load chord/prog once ======
PC_NAMES, CHORD_TABLE = load_chord_table(os.path.join(BASE_DIR, "chord.CSV"))
PROGS = load_prog16(os.path.join(BASE_DIR, "prog16.CSV"))

def list_genres() -> List[str]:
    root = os.path.join(BASE_DIR, "accomp_patterns")
    if not os.path.isdir(root):
        return []
    out = []
    for d in sorted(os.listdir(root)):
        gpath = os.path.join(root, d)
        if not os.path.isdir(gpath):
            continue
        if os.path.isdir(os.path.join(gpath, "bass")) and os.path.isdir(os.path.join(gpath, "comping")):
            out.append(d)
    return out

# ====== Utils ======
def choose_patterns_for_genre(genre: str) -> tuple[PatternRhythm, PatternRhythm, str]:
    """Pick one bass & one comping pattern MIDI from genre folders; fallback to 'basic'."""
    def pick_from(folder: str) -> Optional[str]:
        cand = sorted(glob.glob(os.path.join(folder, "*.mid")) + glob.glob(os.path.join(folder, "*.midi")))
        return random.choice(cand) if cand else None

    base = os.path.join(BASE_DIR, "accomp_patterns")
    gdir = os.path.join(base, genre)
    if not os.path.isdir(gdir):
        # fallback to 'basic'
        gdir = os.path.join(base, "basic")
        genre = "basic"

    bass_dir = os.path.join(gdir, "bass")
    comp_dir = os.path.join(gdir, "comping")

    b_path = pick_from(bass_dir)
    c_path = pick_from(comp_dir)

    # final fallbacks
    if not b_path:
        b_path = pick_from(os.path.join(base, "basic", "bass"))
    if not c_path:
        c_path = pick_from(os.path.join(base, "basic", "comping"))

    if not b_path or not c_path:
        raise RuntimeError("No pattern MIDIs found under accomp_patterns.")

    pat_bass = extract_pattern_rhythm(b_path)
    pat_comp = extract_pattern_rhythm(c_path)
    return pat_bass, pat_comp, genre

def make_abs_url(request: Request, rel_path: str) -> str:
    base = str(request.base_url).rstrip("/")
    rel = rel_path if rel_path.startswith("/") else "/" + rel_path
    return base + rel

# ====== Endpoints ======
@app.get("/health")
def health():
    return PlainTextResponse("ok true")

@app.get("/genres")
def genres():
    gs = list_genres()
    if not gs:
        gs = ["basic"]  # last resort
    return {"genres": gs}

# alias for older frontends
@app.get("/list-genres")
def list_genres_alias():
    return genres()

@app.post("/harmonize")
async def harmonize(
    request: Request,
    midi_file: UploadFile = File(...),
    bpm: float = Form(120.0),
    genre: str = Form("basic"),
):
    """
    Pipeline (non-interactive):
      1) read melody MIDI
      2) auto-detect key from melody
      3) transpose melody -> C/Am (selection only)
      4) choose progressions per block in C/Am
      5) resolve tone-maps in C (csv used once)
      6) transpose maps back to original key
      7) re-voice accompaniment in ORIGINAL key
      8) export single MIDI file and return URL + metadata
    """
    # save upload to bytes -> temp file
    data = await midi_file.read()
    tmp_in = os.path.join(PUBLIC_DIR, f"upload_{uuid.uuid4().hex}.mid")
    with open(tmp_in, "wb") as f:
        f.write(data)

    # 1) melody
    melody, total_beats, first_tempo = melody_notes_from_midifile(tmp_in)
    if not melody:
        return JSONResponse(status_code=400, content={"detail": "No melody notes detected in MIDI (non-drum)."})

    # 2) key detect
    root_pc, mode = guess_key_from_melody(melody)
    offset = semitone_offset_for_key(root_pc, mode)  # C/Am -> orig (+)
    semi_to_C = (12 - offset) % 12                   # melody -> C/Am
    key_label = f"Auto: {NAMES_SHARP[root_pc]} {mode} (offset +{offset})"

    # 3) transpose melody to C for selection
    melody_C = transpose_notes(melody, semi_to_C)

    # 4) choose in C
    chosen_in_C = choose_progressions_for_blocks(melody_C, PROGS, CHORD_TABLE, PC_NAMES)

    # 5) resolve in C
    resolved_C   = [resolve_progression_from_C(p, CHORD_TABLE) for p in chosen_in_C]

    # 6) transpose maps to original key
    resolved_key = [transpose_resolved_progression(rp, offset) for rp in resolved_C]

    # pick patterns for genre
    try:
        pat_bass, pat_comp, genre_used = choose_patterns_for_genre(genre)
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": f"Pattern error: {e}"})

    # 7) re-voice & render
    rendered_out, chosen_names = assemble_from_resolved_progressions(melody, resolved_key, pat_bass, pat_comp)

    # BPM handling
    if (not bpm) or bpm <= 0:
        from mido import tempo2bpm
        bpm = float(tempo2bpm(first_tempo)) if first_tempo else 120.0

    # 8) export
    out_name = f"harmonized_{uuid.uuid4().hex}.mid"
    out_path = os.path.join(PUBLIC_DIR, out_name)
    export_mid(rendered_out, melody, bpm, out_path, monitor_channel=CH_MELODY_DEFAULT)

    midi_url = make_abs_url(request, f"/files/{out_name}")

    return {
        "ok": True,
        "key_label": key_label,
        "chosen_progressions": chosen_names,
        "midi_url": midi_url,
        # "audio_url": None,  # (선택) 사운드폰트 렌더 후 제공하고 싶다면 여기에 절대 URL 넣기
        "genre_used": genre_used,
        "total_beats": rendered_out.total_beats,
        "bpm": bpm,
    }

# alias for older frontends
@app.post("/analyze")
async def analyze_alias(
    request: Request,
    midi_file: UploadFile = File(...),
    bpm: float = Form(120.0),
    genre: str = Form("basic"),
):
    return await harmonize(request, midi_file, bpm, genre)
