#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harmonizator v0.5.0 — key-invariant selection, numeric-map transpose, settings menu/persist
- Selection in C/Am → resolve deg→pc map ONCE in C → rotate numerically to original key (no post-lookup)
- Re-voice in ORIGINAL key (Bass lowest; Comping highest; cap=A4=69; include T/L)
- Real-time capture: fix Windows port crash (no nested open_output)
- Settings persisted to settings_harmonizator.json (MIDI IN/OUT, monitor on/off, channel, double, loudness, BPM)
"""

import os
import sys
import math
import time
import glob
import json
import random
import threading
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import pandas as pd
import mido
import re

try:
    import msvcrt  # Windows only
except Exception:
    msvcrt = None

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
WEIGHTS = {**{str(k): 32 for k in range(1, 7)}, '7': 24, 'T': 8, 'L': 8, 'S': 4, 'A': 2}
PC_NAMES_DEFAULT = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']

CH_COMPING = 1   # CH2
CH_BASS    = 2   # CH3
CH_MELODY_DEFAULT = 0  # CH1
CH_CLICK   = 9   # CH10

NOTE_RANGE_LOW  = 40  # E2
NOTE_RANGE_HIGH = 69  # A4

BLOCK_BEATS = 32.0
BAR_BEATS   = 4.0
SIXTEENTH   = 0.25

CLIP_AT_BLOCK_BOUNDARY = True
SAFETY_PANIC_MS        = 500
EPS_BEAT = 1e-5

CLICK_WEAK = 77
CLICK_STR  = 76
CLICK_VEL  = 127
CLICK_STACK_STRONG = 8
CLICK_STACK_WEAK   = 8

OUTPUT_DIR = './output_harmonizator'
SETTINGS_PATH = './settings_harmonizator.json'

# These will be overwritten from settings on run
LOUDNESS_PCT_MELODY  = 100
LOUDNESS_PCT_COMPING = 100
LOUDNESS_PCT_BASS    = 100

mido.set_backend('mido.backends.rtmidi')

# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Note:
    start_beat: float
    end_beat: float
    pitch: int
    velocity: int

@dataclass
class PatternEvent:
    onset_beats: float
    dur_beats: float
    velocity: int = 100

@dataclass
class PatternRhythm:
    events: List[PatternEvent]
    total_beats: float

@dataclass
class Progression:
    name: str
    seq: List[Optional[str]]  # len 128 (16th grid), carries forward

@dataclass
class ChordInfo:
    labels: Dict[str, str]     # pc_name -> {1..7,T,L,S,A}
    pc_order: List[str]
    deg_to_pc: Dict[str, int]  # '1'..'7','T','L','S','A' -> 0..11

# Runtime chord independent of csv
@dataclass
class ResolvedChord:
    deg_to_pc: Dict[str, int]      # numeric PC map (absolute 0..11)
    inv: Optional[int]             # 1..7 or None
    display_name: str              # for logs (may be synthetic)

@dataclass
class ResolvedProgression:
    name: str
    seq: List[Optional[ResolvedChord]]

@dataclass
class MidiEvent:
    time_beats: float
    type: str   # 'on' | 'off' | 'panic'
    channel: int
    note: int
    velocity: int = 0
    role: str = ""  # 'mel' | 'comp' | 'bass' | ''

@dataclass
class RenderedSong:
    events: List[MidiEvent]
    total_beats: float

# ──────────────────────────────────────────────────────────────────────────────
# Settings (persist)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "midi_in": None,             # device name str or None
    "midi_out": None,            # device name str or None
    "enable_monitor": True,
    "monitor_channel": 1,        # 1..16 (avoid 10); internally we use 0-based
    "double_melody": False,
    "loudness_melody": 100,
    "loudness_comping": 100,
    "loudness_bass": 100,
    "bpm_capture": 120.0
}

def load_settings() -> dict:
    s = DEFAULT_SETTINGS.copy()
    if os.path.isfile(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            s.update({k: data.get(k, v) for k, v in DEFAULT_SETTINGS.items()})
        except Exception as e:
            print("Warning: failed to load settings, using defaults:", e)
    return s

def save_settings(s: dict):
    try:
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Warning: failed to save settings:", e)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ev_sort_key(ev: MidiEvent):
    order = 0 if ev.type == 'panic' else (1 if ev.type == 'off' else 2)
    return (ev.time_beats, order)

def _scale_vel(v: int, pct: int) -> int:
    x = int(round(v * max(0, pct) / 100.0))
    return max(1, min(127, x))

def choose_from_list(title: str, options: List[str], allow_quit: bool=False) -> int:
    print(f"\n== {title} ==")
    for i, s in enumerate(options):
        print(f"  [{i}] {s}")
    if allow_quit:
        print("  [q] Quit")
    while True:
        r = input("Select index: ").strip().lower()
        if allow_quit and r == 'q':
            return -1
        if r.isdigit():
            idx = int(r)
            if 0 <= idx < len(options):
                return idx
        print("Invalid selection. Try again.")

def prompt_yes_no(msg: str, default: bool=True) -> bool:
    d = 'Y/n' if default else 'y/N'
    while True:
        r = input(f"{msg} ({d}): ").strip().lower()
        if not r:
            return default
        if r in ('y','yes'):
            return True
        if r in ('n','no'):
            return False
        print("Please answer y/n.")

def list_mid_files(cwd: str='.') -> List[str]:
    return sorted(glob.glob(os.path.join(cwd, '*.mid')) + glob.glob(os.path.join(cwd, '*.midi')))

def pick_random_mid_in(folder: str) -> Optional[str]:
    cand = sorted(glob.glob(os.path.join(folder, '*.mid')) + glob.glob(os.path.join(folder, '*.midi')))
    return random.choice(cand) if cand else None

# ──────────────────────────────────────────────────────────────────────────────
# CSV loaders
# ──────────────────────────────────────────────────────────────────────────────
def load_chord_table(path: str='./chord.CSV', encoding: str='cp949') -> Tuple[List[str], Dict[str, ChordInfo]]:
    raw = pd.read_csv(path, header=None, encoding=encoding)
    pc_names = [str(x).strip() for x in raw.iloc[0, 1:13].tolist()]
    if len(pc_names) != 12:
        raise ValueError('chord.CSV: header B:M must contain 12 pitch-class names')

    table: Dict[str, ChordInfo] = {}
    for i in range(2, len(raw)):
        name_cell = raw.iloc[i, 0]
        name = str(name_cell).strip() if pd.notna(name_cell) else ''
        if not name or name.lower() == 'nan':
            continue
        row = raw.iloc[i, 1:13]
        labels: Dict[str, str] = {}
        deg_to_pc: Dict[str, int] = {}
        for pc_idx, mark in enumerate(row.tolist()):
            if pd.isna(mark):
                continue
            m = str(mark).strip()
            if not m:
                continue
            pc_name = pc_names[pc_idx]
            labels[pc_name] = m
            if m.isdigit():
                k = int(m)
                if 1 <= k <= 7 and (str(k) not in deg_to_pc):
                    deg_to_pc[str(k)] = pc_idx
            elif m in ('T','L','S','A'):
                if m not in deg_to_pc:
                    deg_to_pc[m] = pc_idx
        table[name] = ChordInfo(labels=labels, pc_order=pc_names, deg_to_pc=deg_to_pc)
    return pc_names, table

def load_prog16(path: str='./prog16.CSV', encoding: str='cp949') -> List[Progression]:
    raw = pd.read_csv(path, header=None, encoding=encoding)
    progs: List[Progression] = []
    for i in range(2, len(raw)):
        name_cell = raw.iloc[i, 0]
        name = str(name_cell).strip() if pd.notna(name_cell) else f'Row{i}'
        row = raw.iloc[i, 1:130]  # 128 cols
        if row.isna().all():
            continue
        seq: List[Optional[str]] = []
        prev: Optional[str] = None
        for cell in row.tolist():
            s = '' if (cell is None or (isinstance(cell, float) and math.isnan(cell))) else str(cell).strip()
            if s:
                prev = s
            seq.append(prev)
        if len(seq) < 128:
            seq += [seq[-1] if seq else None] * (128 - len(seq))
        progs.append(Progression(name=name, seq=seq))
    return progs

# ──────────────────────────────────────────────────────────────────────────────
# Key handling
# ──────────────────────────────────────────────────────────────────────────────
NAMES_SHARP = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
NAMES_FLAT  = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
NAME2PC = {
    'C':0,'B#':0,
    'C#':1,'Db':1,'C♯':1,'D♭':1,
    'D':2,
    'D#':3,'Eb':3,'D♯':3,'E♭':3,
    'E':4,'Fb':4,'F♭':4,
    'F':5,'E#':5,'E♯':5,
    'F#':6,'Gb':6,'F♯':6,'G♭':6,
    'G':7,
    'G#':8,'Ab':8,'G♯':8,'A♭':8,
    'A':9,
    'A#':10,'Bb':10,'A♯':10,'B♭':10,
    'B':11,'Cb':11,'C♭':11,
}

MAJ_COLL = [{(r+i) % 12 for i in (0,2,4,5,7,9,11)} for r in range(12)]
AEOL_TONIC_PC = 9  # A

def _pc_hist_from_melody(notes: List[Note]) -> list[float]:
    h = [0.0]*12
    for n in notes:
        h[n.pitch % 12] += max(0.0, n.end_beat - n.start_beat)
    return h

def _argmax(xs): return max(range(len(xs)), key=lambda i: xs[i])

def guess_key_from_melody(notes: List[Note]) -> tuple[int,str]:
    h = _pc_hist_from_melody(notes)
    scores = [sum(h[i] for i in MAJ_COLL[r]) for r in range(12)]
    root_pc = _argmax(scores)
    maj_bias = h[(root_pc + 4) % 12] + 0.5*h[(root_pc + 11) % 12]
    min_bias = h[(root_pc + (AEOL_TONIC_PC - 0)) % 12] + h[(root_pc + 3) % 12]
    mode = 'major' if maj_bias >= min_bias else 'minor'
    return root_pc, mode

def parse_key_string(s: str) -> tuple[int,str] | None:
    z = s.strip().replace('M','maj').replace('min','minor').lower()
    z = z.replace('♯','#').replace('♭','b')
    m = re.match(r'^([a-g](?:#|b)?)(?:\s*(maj|major|m|mi|minor))?$', z)
    if not m: return None
    root = m.group(1).upper()
    mode = m.group(2) or ''
    pc = NAME2PC.get(root)
    if pc is None: return None
    if mode in ('','maj','major'):
        return pc, 'major'
    return pc, 'minor'

def semitone_offset_for_key(root_pc: int, mode: str) -> int:
    anchor = 0 if mode == 'major' else 9  # C or A
    return (root_pc - anchor) % 12

def choose_key_offset(notes: List[Note]) -> tuple[int, str]:
    options = ["Treat as C-collection (no transpose)", "Detect key from melody", "Manual key (e.g., Eb major, F# minor)"]
    choice = choose_from_list("Key handling", options, allow_quit=False)
    if choice == 0:
        return 0, "C collection (no transpose)"
    if choice == 1:
        root_pc, mode = guess_key_from_melody(notes)
        off = semitone_offset_for_key(root_pc, mode)
        return off, f"Auto: {NAMES_SHARP[root_pc]} {mode} (offset +{off})"
    while True:
        s = input("Enter key (e.g., E major / F# minor / Eb maj / Am): ").strip()
        res = parse_key_string(s)
        if res:
            root_pc, mode = res
            off = semitone_offset_for_key(root_pc, mode)
            return off, f"Manual: {NAMES_SHARP[root_pc]} {mode} (offset +{off})"
        print("Could not parse. Try again.")

# ──────────────────────────────────────────────────────────────────────────────
# Melody MIDI → notes
# ──────────────────────────────────────────────────────────────────────────────
def melody_notes_from_midifile(path: str) -> Tuple[List[Note], float, Optional[int]]:
    mid = mido.MidiFile(path)
    tpq = mid.ticks_per_beat
    abs_ticks = 0
    notes_active: Dict[Tuple[int,int], Tuple[float,int]] = {}
    notes: List[Note] = []
    first_tempo_us = None

    for msg in mido.merge_tracks(mid.tracks):
        abs_ticks += msg.time
        if msg.is_meta:
            if msg.type == 'set_tempo' and first_tempo_us is None:
                first_tempo_us = msg.tempo
            continue
        if not hasattr(msg, 'channel') or msg.channel == 9:
            continue
        beat_time = abs_ticks / tpq
        if msg.type == 'note_on' and msg.velocity > 0:
            notes_active[(msg.channel, msg.note)] = (beat_time, msg.velocity)
        elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in notes_active:
                st, vel = notes_active.pop(key)
                dur = beat_time - st
                if dur > EPS_BEAT:
                    notes.append(Note(st, beat_time, msg.note, vel))

    total_beats = max((n.end_beat for n in notes), default=0.0)
    return notes, total_beats, first_tempo_us

# ──────────────────────────────────────────────────────────────────────────────
# Scoring (in C/Am)
# ──────────────────────────────────────────────────────────────────────────────
def parse_chord_token(tok: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    if tok is None:
        return None, None
    s = tok.strip()
    if not s:
        return None, None
    if '/' in s:
        base, inv = s.split('/', 1)
        try:
            n = int(inv)
        except:
            n = None
        return base.strip(), n
    return s, None

def chord_segments(seq: List[Optional[str]]) -> List[Tuple[int, int, Optional[str]]]:
    segs: List[Tuple[int,int,Optional[str]]] = []
    cur = seq[0]
    seg_start = 0
    for i in range(1, 128):
        if seq[i] != cur:
            segs.append((seg_start, i, cur))
            seg_start = i
            cur = seq[i]
    segs.append((seg_start, 128, cur))
    return segs

def note_pc_name(note: Note, pc_names: List[str]) -> str:
    return pc_names[note.pitch % 12]

def weight_for_label(lbl: Optional[str]) -> int:
    if not lbl:
        return 0
    return WEIGHTS.get(lbl, 0)

def score_progression_for_block(notes: List[Note], prog: Progression, chord_table: Dict[str, ChordInfo], pc_names: List[str]) -> float:
    segs = chord_segments(prog.seq)
    total = 0.0
    for st_step, en_step, tok in segs:
        base, _inv = parse_chord_token(tok)
        if base is None:
            continue
        chord = chord_table.get(base)
        if not chord:
            continue
        st_b = st_step * SIXTEENTH
        en_b = en_step * SIXTEENTH
        for n in notes:
            overlap = max(0.0, min(en_b, n.end_beat) - max(st_b, n.start_beat))
            if overlap <= 0:
                continue
            lbl = chord.labels.get(note_pc_name(n, pc_names))
            w = weight_for_label(lbl)
            if w:
                total += w * overlap
    return total

# ──────────────────────────────────────────────────────────────────────────────
# Pattern rhythm
# ──────────────────────────────────────────────────────────────────────────────
def extract_pattern_rhythm(path: str) -> PatternRhythm:
    mid = mido.MidiFile(path)
    tpq = mid.ticks_per_beat
    abs_ticks = 0
    notes_active: Dict[Tuple[int,int], Tuple[float, int]] = {}
    events: Dict[float, Tuple[float, int]] = {}

    for msg in mido.merge_tracks(mid.tracks):
        abs_ticks += msg.time
        if msg.is_meta or not hasattr(msg, 'channel'):
            continue
        beat = abs_ticks / tpq
        if msg.type == 'note_on' and msg.velocity > 0:
            notes_active[(msg.channel, msg.note)] = (beat, msg.velocity)
        elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in notes_active:
                st, vel = notes_active.pop(key)
                dur = max(0.0, beat - st)
                if dur > EPS_BEAT:
                    prev = events.get(st)
                    if prev is None:
                        events[st] = (dur, vel)
                    else:
                        prev_dur, prev_vel = prev
                        events[st] = (max(prev_dur, dur), max(prev_vel, vel))

    if not events:
        return PatternRhythm(events=[PatternEvent(0.0, 4.0, 100)], total_beats=4.0)

    onsets = sorted(events.keys())
    evs = [PatternEvent(o, events[o][0], events[o][1]) for o in onsets]
    total = max(o + d for o, d in ((e.onset_beats, e.dur_beats) for e in evs))
    if 3.9 <= total <= 4.1:
        total = 4.0
    return PatternRhythm(events=evs, total_beats=total)

# ──────────────────────────────────────────────────────────────────────────────
# Transpose helpers (notes)
# ──────────────────────────────────────────────────────────────────────────────
def transpose_notes(notes: List[Note], semi: int) -> List[Note]:
    if semi % 12 == 0:
        return [Note(n.start_beat, n.end_beat, n.pitch, n.velocity) for n in notes]
    out: List[Note] = []
    for n in notes:
        p = max(0, min(127, n.pitch + semi))
        out.append(Note(n.start_beat, n.end_beat, p, n.velocity))
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Resolve in C, then numeric rotate (no post-lookup)
# ──────────────────────────────────────────────────────────────────────────────
_ROOT_RE = re.compile(r'^([A-G](?:#|b)?)(.*)$')

def _split_base_and_inv(tok: str) -> Tuple[str, Optional[int]]:
    s = tok.strip()
    if not s: return "", None
    if '/' in s:
        base, inv = s.split('/', 1)
        try:
            n = int(inv)
        except:
            n = None
        return base.strip(), n
    return s, None

def resolve_progression_from_C(p: Progression, chord_table: Dict[str, ChordInfo]) -> ResolvedProgression:
    seq_res: List[Optional[ResolvedChord]] = []
    for tok in p.seq:
        if tok is None:
            seq_res.append(None); continue
        base, inv = _split_base_and_inv(tok)
        chd = chord_table.get(base)
        if not chd:
            seq_res.append(None)
            continue
        seq_res.append(ResolvedChord(deg_to_pc=dict(chd.deg_to_pc), inv=inv, display_name=base))
    return ResolvedProgression(name=p.name, seq=seq_res)

def transpose_resolved_progression(rp: ResolvedProgression, semi: int) -> ResolvedProgression:
    def rot(pc: int) -> int: return (pc + semi) % 12
    seq2: List[Optional[ResolvedChord]] = []
    for rc in rp.seq:
        if rc is None:
            seq2.append(None); continue
        deg_map = {k: rot(v) for k, v in rc.deg_to_pc.items()}
        seq2.append(ResolvedChord(deg_to_pc=deg_map, inv=rc.inv, display_name=rc.display_name))
    return ResolvedProgression(name=f"{rp.name} (+{semi})", seq=seq2)

# ──────────────────────────────────────────────────────────────────────────────
# Voicing (numeric map)
# ──────────────────────────────────────────────────────────────────────────────
def nearest_pitch_leq(pc: int, max_pitch: int) -> int:
    k = (max_pitch - pc) // 12
    return pc + 12 * int(k)

def lowest_pitch_geq(pc: int, min_pitch: int) -> int:
    k = math.ceil((min_pitch - pc) / 12)
    return pc + 12 * int(k)

def comping_voicing_from_map(deg_to_pc: Dict[str,int], bass_deg: Optional[int],
                             low_limit: int=NOTE_RANGE_LOW, high_limit: int=NOTE_RANGE_HIGH) -> List[int]:
    notes: List[int] = []
    bass_key = str(bass_deg) if bass_deg is not None else None
    for deg in ['1','2','3','4','5','6','7','T','L']:
        if bass_key is not None and deg == bass_key:
            continue
        if deg not in deg_to_pc:
            continue
        pc = deg_to_pc[deg]
        p = nearest_pitch_leq(pc, high_limit)
        if p < low_limit:
            continue
        notes.append(p)
    return sorted(set(notes))

def bass_pitch_from_map(deg_to_pc: Dict[str,int], inversion: Optional[int],
                        low_limit: int=NOTE_RANGE_LOW, high_limit: int=NOTE_RANGE_HIGH) -> Tuple[Optional[int], Optional[int]]:
    deg = inversion if (isinstance(inversion, int) and 1 <= inversion <= 7) else 1
    key = str(deg)
    if key not in deg_to_pc:
        return None, None
    pc = deg_to_pc[key]
    p = lowest_pitch_geq(pc, low_limit)
    while p > high_limit:
        p -= 12
    if p < low_limit or p > high_limit:
        return None, None
    return p, deg

# ──────────────────────────────────────────────────────────────────────────────
# Accompaniment synthesis (from resolved progressions)
# ──────────────────────────────────────────────────────────────────────────────
def build_accompaniment_for_block_resolved(
    prog: ResolvedProgression,
    pat_bass: PatternRhythm,
    pat_comp: PatternRhythm,
) -> List[MidiEvent]:
    events: List[MidiEvent] = []

    seq = prog.seq
    bar_starts = [i * BAR_BEATS for i in range(8)]

    change_steps: List[int] = [0]
    last = seq[0]
    for i in range(1, 128):
        if seq[i] != last:
            change_steps.append(i)
            last = seq[i]
    triggers = sorted(set(bar_starts + [st * SIXTEENTH for st in change_steps] + [BLOCK_BEATS]))

    def chord_at(beat_t: float) -> Tuple[Optional[Dict[str,int]], Optional[int]]:
        step = int(round(beat_t / SIXTEENTH))
        step = max(0, min(127, step))
        rc = seq[step]
        if rc is None:
            return None, None
        return rc.deg_to_pc, rc.inv

    for k in range(len(triggers) - 1):
        t0 = triggers[k]
        t1 = triggers[k + 1]

        step = int(round(t0 / SIXTEENTH))
        mod = step % 4
        ext = 0.0
        if mod == 1:   ext = 0.75
        elif mod == 2: ext = 0.50
        elif mod == 3: ext = 0.25

        events.append(MidiEvent(t0, 'panic', CH_COMPING, 0, role='comp'))
        events.append(MidiEvent(t0, 'panic', CH_BASS,    0, role='bass'))

        deg_map, inv = chord_at(t0)
        if not deg_map:
            events.append(MidiEvent(t1, 'panic', CH_COMPING, 0, role='comp'))
            events.append(MidiEvent(t1, 'panic', CH_BASS,    0, role='bass'))
            continue

        # Bass (lowest possible)
        b_pitch, b_deg = bass_pitch_from_map(deg_map, inv)
        if b_pitch is not None:
            for i, ev in enumerate(pat_bass.events):
                base_onset = t0 + ev.onset_beats
                if i == 0:
                    onset_i = base_onset
                    dur_i   = ev.dur_beats + ext
                else:
                    onset_i = base_onset + ext
                    dur_i   = ev.dur_beats
                if onset_i >= t1:
                    continue
                off_t = min(onset_i + dur_i, t1)
                if off_t - onset_i > EPS_BEAT:
                    vel_i = max(1, min(127, getattr(ev, 'velocity', 100)))
                    events.append(MidiEvent(onset_i, 'on',  CH_BASS, b_pitch, vel_i, role='bass'))
                    events.append(MidiEvent(off_t,  'off', CH_BASS, b_pitch, 0,    role='bass'))

        # Comping (highest first)
        voicing = comping_voicing_from_map(deg_map, bass_deg=b_deg)
        if voicing:
            for i, ev in enumerate(pat_comp.events):
                base_onset = t0 + ev.onset_beats
                if i == 0:
                    onset_i = base_onset
                    dur_i   = ev.dur_beats + ext
                else:
                    onset_i = base_onset + ext
                    dur_i   = ev.dur_beats
                if onset_i >= t1:
                    continue
                off_t = min(onset_i + dur_i, t1)
                if off_t - onset_i > EPS_BEAT:
                    vel_i = max(1, min(127, getattr(ev, 'velocity', 100)))
                    for p in voicing:
                        events.append(MidiEvent(onset_i, 'on',  CH_COMPING, p, vel_i, role='comp'))
                        events.append(MidiEvent(off_t,  'off', CH_COMPING, p, 0,    role='comp'))

        events.append(MidiEvent(t1, 'panic', CH_COMPING, 0, role='comp'))
        events.append(MidiEvent(t1, 'panic', CH_BASS,    0, role='bass'))

    if CLIP_AT_BLOCK_BOUNDARY:
        events.append(MidiEvent(BLOCK_BEATS, 'panic', CH_COMPING, 0, role='comp'))
        events.append(MidiEvent(BLOCK_BEATS, 'panic', CH_BASS,    0, role='bass'))

    events.sort(key=_ev_sort_key)
    return events

# ──────────────────────────────────────────────────────────────────────────────
# Build (select in C → resolve in C → rotate numeric → re-voice)
# ──────────────────────────────────────────────────────────────────────────────
def pick_best_progression_for_block(notes: List[Note], progs: List[Progression],
                                    chord_table: Dict[str, ChordInfo], pc_names: List[str]) -> Progression:
    best = None
    best_score = -1e-18
    ties: List[Progression] = []
    for p in progs:
        s = score_progression_for_block(notes, p, chord_table, pc_names)
        if s > best_score:
            best, best_score = p, s
            ties = [p]
        elif abs(s - best_score) < 1e-9:
            ties.append(p)
    if len(ties) > 1:
        best = random.choice(ties)
    return best

def split_notes_into_blocks(notes: List[Note]) -> List[List[Note]]:
    if not notes:
        return []
    last = max(n.end_beat for n in notes)
    n_blocks = int(math.ceil(last / BLOCK_BEATS))
    blocks: List[List[Note]] = [[] for _ in range(n_blocks)]
    for n in notes:
        b0 = int(n.start_beat // BLOCK_BEATS)
        b1 = int((n.end_beat - 1e-9) // BLOCK_BEATS)
        for bi in range(b0, b1 + 1):
            st = max(0.0, n.start_beat - bi * BLOCK_BEATS)
            en = min(BLOCK_BEATS, n.end_beat - bi * BLOCK_BEATS)
            if en - st > EPS_BEAT:
                blocks[bi].append(Note(st, en, n.pitch, n.velocity))
    return blocks

def choose_progressions_for_blocks(notes_tr: List[Note], progs: List[Progression],
                                   chord_table: Dict[str, ChordInfo], pc_names: List[str]) -> List[Progression]:
    blocks = split_notes_into_blocks(notes_tr)
    chosen: List[Progression] = []
    for blk in blocks:
        chosen.append(pick_best_progression_for_block(blk, progs, chord_table, pc_names))
    return chosen

def assemble_from_resolved_progressions(
    melody_orig: List[Note],
    resolved_list: List[ResolvedProgression],
    pat_bass: PatternRhythm,
    pat_comp: PatternRhythm
) -> Tuple[RenderedSong, List[str]]:
    blocks = split_notes_into_blocks(melody_orig)
    all_events: List[MidiEvent] = []
    names: List[str] = []
    for bi, rp in enumerate(resolved_list):
        names.append(rp.name)
        evs = build_accompaniment_for_block_resolved(rp, pat_bass, pat_comp)
        for e in evs:
            all_events.append(MidiEvent(e.time_beats + bi * BLOCK_BEATS, e.type, e.channel, e.note, e.velocity, role=e.role))
    total_beats = len(resolved_list) * BLOCK_BEATS
    all_events.sort(key=_ev_sort_key)
    return RenderedSong(all_events, total_beats), names

# ──────────────────────────────────────────────────────────────────────────────
# Playback & export
# ──────────────────────────────────────────────────────────────────────────────
class Player:
    def __init__(self, outport: mido.ports.BaseOutput, bpm: float, monitor_channel: Optional[int]=None):
        self.out = outport
        self.bpm = bpm
        self.sec_per_beat = 60.0 / bpm
        self.monitor_channel = monitor_channel
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def send_panic_channel(self, ch: int):
        self.out.send(mido.Message('control_change', channel=ch, control=123, value=0))
        self.out.send(mido.Message('control_change', channel=ch, control=120, value=0))

    def send_panic_all(self):
        for ch in range(16):
            self.send_panic_channel(ch)

    def _play_loop(self, rendered: RenderedSong):
        t0 = time.time()
        for ev in rendered.events:
            if self._stop_flag.is_set():
                break
            tgt = t0 + ev.time_beats * self.sec_per_beat
            if (dt := tgt - time.time()) > 0:
                while (tgt - time.time()) > 0:
                    if self._stop_flag.is_set():
                        break
                    time.sleep(0.02)
                if self._stop_flag.is_set():
                    break

            if ev.type == 'panic':
                self.send_panic_channel(ev.channel)
            elif ev.type == 'on':
                pct = 100
                if ev.role == 'mel':
                    pct = LOUDNESS_PCT_MELODY
                elif ev.role == 'comp':
                    pct = LOUDNESS_PCT_COMPING
                elif ev.role == 'bass':
                    pct = LOUDNESS_PCT_BASS
                v = _scale_vel(ev.velocity, pct)
                self.out.send(mido.Message('note_on', channel=ev.channel, note=ev.note, velocity=v))
            elif ev.type == 'off':
                self.out.send(mido.Message('note_off', channel=ev.channel, note=ev.note, velocity=0))

        time.sleep(SAFETY_PANIC_MS / 1000.0)
        self.send_panic_all()

    def play_events_async(self, rendered: RenderedSong):
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._play_loop, args=(rendered,), daemon=True)
        self._thread.start()
        return self._thread

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.send_panic_all()

def export_mid(rendered: RenderedSong, melody: List[Note], bpm: float, path: str, monitor_channel: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tpq = 480
    mid = mido.MidiFile(ticks_per_beat=tpq)

    tempo_tr = mido.MidiTrack(); mid.tracks.append(tempo_tr)
    tempo_tr.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))

    def beat_to_ticks(b):
        return int(round(b * tpq))

    # Melody
    mel_tr = mido.MidiTrack(); mid.tracks.append(mel_tr)
    mel_events = []
    for n in melody:
        mel_events.append(('off', n.end_beat, n.pitch, 0))
        mel_events.append(('on',  n.start_beat, n.pitch, n.velocity))
    mel_events.sort(key=lambda x: (x[1], 0 if x[0] == 'off' else 1))
    t_last = 0
    for typ, b, pitch, vel in mel_events:
        t = beat_to_ticks(b); delta = t - t_last; t_last = t
        if typ == 'on':
            mel_tr.append(mido.Message('note_on', channel=monitor_channel, note=pitch, velocity=vel, time=max(0, delta)))
        else:
            mel_tr.append(mido.Message('note_off', channel=monitor_channel, note=pitch, velocity=0, time=max(0, delta)))

    # Accompaniment
    acc_tr = mido.MidiTrack(); mid.tracks.append(acc_tr)
    t_last = 0
    for ev in rendered.events:
        t = beat_to_ticks(ev.time_beats); delta = t - t_last; t_last = t
        if ev.type == 'panic':
            acc_tr.append(mido.Message('control_change', channel=ev.channel, control=123, value=0, time=max(0, delta)))
            acc_tr.append(mido.Message('control_change', channel=ev.channel, control=120, value=0, time=0))
        elif ev.type == 'on':
            acc_tr.append(mido.Message('note_on', channel=ev.channel, note=ev.note, velocity=ev.velocity, time=max(0, delta)))
        elif ev.type == 'off':
            acc_tr.append(mido.Message('note_off', channel=ev.channel, note=ev.note, velocity=0, time=max(0, delta)))
    mid.save(path)

def melody_to_events(melody: List[Note], channel: int, double_up: bool=False) -> List[MidiEvent]:
    evs: List[MidiEvent] = []
    for n in melody:
        evs.append(MidiEvent(n.start_beat, 'on',  channel, n.pitch, n.velocity, role='mel'))
        evs.append(MidiEvent(n.end_beat,   'off', channel, n.pitch, 0,          role='mel'))
        if double_up:
            hi = min(127, n.pitch + 12)
            evs.append(MidiEvent(n.start_beat, 'on',  channel, hi, n.velocity, role='mel'))
            evs.append(MidiEvent(n.end_beat,   'off', channel, hi, 0,          role='mel'))
    evs.sort(key=_ev_sort_key)
    return evs

# ──────────────────────────────────────────────────────────────────────────────
# Real-time capture (metronome + monitoring)
# ──────────────────────────────────────────────────────────────────────────────
class RealTimeCapture:
    def __init__(self, in_name: str, out: mido.ports.BaseOutput, bpm: float, monitor_channel: Optional[int]=None):
        self.in_name = in_name
        self.out = out
        self.bpm = bpm
        self.sec_per_beat = 60.0 / bpm
        self.monitor_channel = monitor_channel
        self.stop_flag = threading.Event()
        self.started = threading.Event()
        self.last_note_time: Optional[float] = None
        self.notes_active: Dict[int, Tuple[float,int]] = {}
        self.notes: List[Note] = []
        self.start_wall: Optional[float] = None

        self._owns_met = False
        try:
            self.out_met = mido.open_output("Microsoft GS Wavetable Synth 0")
            self._owns_met = True
        except Exception as e:
            # Fallback to main out if GS not available
            self.out_met = self.out

    def metronome_thread(self):
        beat_idx = 0
        while not self.stop_flag.is_set():
            if self.started.is_set() and self.start_wall is not None:
                now = time.time()
                elapsed = now - self.start_wall
                next_beat = math.floor(elapsed / self.sec_per_beat) + 1
                tgt = self.start_wall + next_beat * self.sec_per_beat
                time.sleep(max(0, tgt - now))
            else:
                time.sleep(self.sec_per_beat)
            beat_idx += 1
            e = (beat_idx % 4)
            note = CLICK_STR if e == 1 else CLICK_WEAK
            layers = CLICK_STACK_STRONG if e == 1 else CLICK_STACK_WEAK
            for _ in range(max(1, layers)):
                self.out_met.send(mido.Message('note_on', channel=CH_CLICK, note=note, velocity=CLICK_VEL))
                self.out_met.send(mido.Message('note_off', channel=CH_CLICK, note=note, velocity=0))

    def input_thread(self):
        try:
            inp = mido.open_input(self.in_name)
        except Exception as e:
            print(f"Failed to open MIDI IN '{self.in_name}': {e}")
            return
        with inp:
            while not self.stop_flag.is_set():
                for msg in inp.iter_pending():
                    if msg.type in ('note_on', 'note_off'):
                        now = time.time()
                        if self.start_wall is None and msg.type == 'note_on' and msg.velocity > 0:
                            self.start_wall = now
                            self.started.set()
                        if self.start_wall is None:
                            continue
                        beat_time = (now - self.start_wall) / self.sec_per_beat
                        if self.monitor_channel is not None and hasattr(msg, 'note'):
                            if msg.type == 'note_on' and msg.velocity > 0:
                                self.out.send(mido.Message('note_on', channel=self.monitor_channel, note=msg.note, velocity=msg.velocity))
                            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                                self.out.send(mido.Message('note_off', channel=self.monitor_channel, note=msg.note, velocity=0))
                        if msg.type == 'note_on' and msg.velocity > 0:
                            self.notes_active[msg.note] = (beat_time, msg.velocity)
                            self.last_note_time = beat_time
                        else:
                            if msg.note in self.notes_active:
                                st, vel = self.notes_active.pop(msg.note)
                                dur = beat_time - st
                                if dur > EPS_BEAT:
                                    self.notes.append(Note(st, beat_time, msg.note, vel))
                                self.last_note_time = beat_time
                time.sleep(0.002)

    def run_until_idle_2bars(self) -> List[Note]:
        t_met = threading.Thread(target=self.metronome_thread, daemon=True)
        t_inp = threading.Thread(target=self.input_thread, daemon=True)
        t_met.start(); t_inp.start()
        print("\n▶ Real-time capture started. Play your melody. First note = start of bar 1.")
        try:
            while True:
                time.sleep(0.05)
                if not self.started.is_set() or self.last_note_time is None:
                    continue
                now_beats = (time.time() - self.start_wall) / self.sec_per_beat
                if now_beats - self.last_note_time >= 8.0:
                    break
        finally:
            self.stop_flag.set()
            t_inp.join(timeout=2.0)
            t_met.join(timeout=2.0)
        try:
            self.out_met.send(mido.Message('control_change', channel=CH_CLICK, control=123, value=0))
            self.out_met.send(mido.Message('control_change', channel=CH_CLICK, control=120, value=0))
        except Exception:
            pass
        if self._owns_met:
            try: self.out_met.close()
            except Exception: pass
        if not self.notes:
            return []
        t0 = min(n.start_beat for n in self.notes)
        return [Note(n.start_beat - t0, n.end_beat - t0, n.pitch, n.velocity) for n in self.notes]

# ──────────────────────────────────────────────────────────────────────────────
# Settings menu
# ──────────────────────────────────────────────────────────────────────────────
def settings_menu(settings: dict):
    while True:
        print("\n== Settings ==")
        print(f"  [0] MIDI OUT device : {settings.get('midi_out')}")
        print(f"  [1] MIDI IN device  : {settings.get('midi_in')}")
        print(f"  [2] Input monitoring: {'On' if settings.get('enable_monitor') else 'Off'}")
        print(f"  [3] Monitor channel : {settings.get('monitor_channel')}")
        print(f"  [4] Double melody   : {'On' if settings.get('double_melody') else 'Off'}")
        print(f"  [5] Loudness (M/C/B): {settings.get('loudness_melody')}/{settings.get('loudness_comping')}/{settings.get('loudness_bass')}")
        print(f"  [6] Capture BPM     : {settings.get('bpm_capture')}")
        print( "  [q] Back (save)")
        sel = input("Select index: ").strip().lower()
        if sel == 'q':
            save_settings(settings)
            print("Settings saved.")
            return
        if not sel.isdigit():
            print("Invalid"); continue
        idx = int(sel)

        if idx == 0:
            names = mido.get_output_names()
            if not names:
                print("No MIDI output devices found.")
            else:
                i = choose_from_list("Select MIDI OUT device", names, allow_quit=True)
                if i >= 0:
                    settings['midi_out'] = names[i]
        elif idx == 1:
            names = mido.get_input_names()
            if not names:
                print("No MIDI input devices found.")
            else:
                i = choose_from_list("Select MIDI IN device", names, allow_quit=True)
                if i >= 0:
                    settings['midi_in'] = names[i]
        elif idx == 2:
            settings['enable_monitor'] = prompt_yes_no("Enable INPUT monitoring?", settings.get('enable_monitor', True))
        elif idx == 3:
            while True:
                try:
                    ch = int(input("Monitor channel (1-16, avoid 10): ").strip() or str(settings.get('monitor_channel', 1)))
                    if 1 <= ch <= 16 and ch != 10:
                        settings['monitor_channel'] = ch; break
                except: pass
                print("Enter 1..16 (not 10).")
        elif idx == 4:
            settings['double_melody'] = prompt_yes_no("Double melody one octave up?", settings.get('double_melody', False))
        elif idx == 5:
            def ask_int(prompt, cur):
                try:
                    v = int(input(f"{prompt} [0-200] (current {cur}): ").strip() or str(cur))
                    return max(0, min(200, v))
                except:
                    return cur
            settings['loudness_melody']  = ask_int("Melody loudness %", settings.get('loudness_melody', 100))
            settings['loudness_comping'] = ask_int("Comping loudness %", settings.get('loudness_comping', 100))
            settings['loudness_bass']    = ask_int("Bass loudness %",    settings.get('loudness_bass', 100))
        elif idx == 6:
            try:
                v = float(input("Capture BPM [20-300]: ").strip() or str(settings.get('bpm_capture', 120.0)))
                if 20.0 <= v <= 300.0:
                    settings['bpm_capture'] = v
                else:
                    print("Out of range.")
            except:
                print("Invalid number.")
        else:
            print("Invalid index.")

# ──────────────────────────────────────────────────────────────────────────────
# Run flows (no repeated prompts for saved settings)
# ──────────────────────────────────────────────────────────────────────────────
def run_from_midifile(settings: dict, progs: List[Progression], chord_table_raw: Dict[str, ChordInfo], pc_names: List[str]):
    # Patterns
    genres = [d for d in sorted(os.listdir('./accomp_patterns')) if os.path.isdir(os.path.join('./accomp_patterns', d))] if os.path.isdir('./accomp_patterns') else []
    if not genres:
        print("No genres found under ./accomp_patterns. Create e.g. ./accomp_patterns/Pop/{bass,comping}")
        return
    g_idx = choose_from_list("Select genre", genres, allow_quit=True)
    if g_idx < 0: return
    genre = genres[g_idx]
    folder_bass = os.path.join('accomp_patterns', genre, 'bass')
    folder_comp = os.path.join('accomp_patterns', genre, 'comping')
    if not os.path.isdir(folder_bass) or not os.path.isdir(folder_comp):
        print("Selected genre folders missing bass/comping subdirs."); return

    pat_bass_path = pick_random_mid_in(folder_bass)
    pat_comp_path = pick_random_mid_in(folder_comp)
    if not pat_bass_path or not pat_comp_path:
        print("No pattern MIDIs in bass/comping folders."); return
    pat_bass = extract_pattern_rhythm(pat_bass_path)
    pat_comp = extract_pattern_rhythm(pat_comp_path)
    print(f"Pattern (bass): {os.path.basename(pat_bass_path)} | events={len(pat_bass.events)} | total_beats={pat_bass.total_beats}")
    print(f"Pattern (comp): {os.path.basename(pat_comp_path)} | events={len(pat_comp.events)} | total_beats={pat_comp.total_beats}")

    files = list_mid_files('.')
    if not files:
        print("No .mid files in current folder. Put your melody MIDI here.")
        return
    f_idx = choose_from_list("Select melody MIDI file", [os.path.basename(x) for x in files], allow_quit=True)
    if f_idx < 0: return
    path = files[f_idx]
    melody, total_beats, first_tempo = melody_notes_from_midifile(path)
    if not melody:
        print("No melody notes found (non-drum)."); return
    bpm = mido.tempo2bpm(first_tempo) if first_tempo else 120.0
    print(f"Melody total {total_beats:.2f} beats. Using BPM={bpm:.2f} for playback/export.")

    # Key handling
    offset, key_label = choose_key_offset(melody)
    semi_to_C = (12 - offset) % 12
    melody_C  = transpose_notes(melody, semi_to_C)
    print(f"[Key] {key_label} | melody_to_C=+{semi_to_C} semitones")

    # Selection in C → resolve in C → rotate numeric to original
    chosen_in_C = choose_progressions_for_blocks(melody_C, progs, chord_table_raw, pc_names)
    resolved_C   = [resolve_progression_from_C(p, chord_table_raw) for p in chosen_in_C]
    resolved_key = [transpose_resolved_progression(rp, offset) for rp in resolved_C]

    rendered_out, chosen_names = assemble_from_resolved_progressions(melody, resolved_key, pat_bass, pat_comp)
    print("Chosen progressions per 8-bar block:")
    for i, name in enumerate(chosen_names):
        print(f"  block {i+1}: {name}")

    # Loudness from settings
    global LOUDNESS_PCT_MELODY, LOUDNESS_PCT_COMPING, LOUDNESS_PCT_BASS
    LOUDNESS_PCT_MELODY  = settings.get('loudness_melody', 100)
    LOUDNESS_PCT_COMPING = settings.get('loudness_comping', 100)
    LOUDNESS_PCT_BASS    = settings.get('loudness_bass', 100)

    # Playback
    mel_ch = (settings.get('monitor_channel', 1) - 1) if settings.get('enable_monitor', True) else CH_MELODY_DEFAULT
    double_flag = settings.get('double_melody', False)
    mel_evs = melody_to_events(melody, mel_ch, double_up=double_flag)
    combined = RenderedSong(events=sorted(rendered_out.events + mel_evs, key=_ev_sort_key),
                            total_beats=rendered_out.total_beats)

    out_name = settings.get('midi_out')
    if not out_name:
        print("No MIDI OUT configured. Set it in Settings."); return
    try:
        with mido.open_output(out_name) as out:
            # timbres
            out.send(mido.Message('program_change', channel=mel_ch, program=71))
            out.send(mido.Message('program_change', channel=CH_COMPING, program=0))
            out.send(mido.Message('program_change', channel=CH_BASS, program=0))

            ts = time.strftime('%Y%m%d_%H%M%S')
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            outpath = os.path.join(OUTPUT_DIR, f'harmonized_{os.path.basename(path)}_{ts}.mid')
            export_mid(rendered_out, melody, bpm, outpath, monitor_channel=mel_ch)
            print(f"Saved: {outpath}")

            player = Player(out, bpm=bpm, monitor_channel=None)
            t = player.play_events_async(combined)
            print("Playing... (Press ENTER / Space / Q to stop early)")
            if msvcrt is not None:
                while t.is_alive():
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch in ('\r', '\n', ' ', 'q', 'Q'):
                            player.stop(); break
                    time.sleep(0.03)
            else:
                t.join()
    except Exception as e:
        print("Failed to open MIDI OUT for playback:", e)

def run_realtime(settings: dict, progs: List[Progression], chord_table_raw: Dict[str, ChordInfo], pc_names: List[str]):
    # Patterns
    genres = [d for d in sorted(os.listdir('./accomp_patterns')) if os.path.isdir(os.path.join('./accomp_patterns', d))] if os.path.isdir('./accomp_patterns') else []
    if not genres:
        print("No genres found under ./accomp_patterns. Create e.g. ./accomp_patterns/Pop/{bass,comping}")
        return
    g_idx = choose_from_list("Select genre", genres, allow_quit=True)
    if g_idx < 0: return
    genre = genres[g_idx]
    folder_bass = os.path.join('accomp_patterns', genre, 'bass')
    folder_comp = os.path.join('accomp_patterns', genre, 'comping')
    if not os.path.isdir(folder_bass) or not os.path.isdir(folder_comp):
        print("Selected genre folders missing bass/comping subdirs."); return

    pat_bass_path = pick_random_mid_in(folder_bass)
    pat_comp_path = pick_random_mid_in(folder_comp)
    if not pat_bass_path or not pat_comp_path:
        print("No pattern MIDIs in bass/comping folders."); return
    pat_bass = extract_pattern_rhythm(pat_bass_path)
    pat_comp = extract_pattern_rhythm(pat_comp_path)
    print(f"Pattern (bass): {os.path.basename(pat_bass_path)} | events={len(pat_bass.events)} | total_beats={pat_bass.total_beats}")
    print(f"Pattern (comp): {os.path.basename(pat_comp_path)} | events={len(pat_comp.events)} | total_beats={pat_comp.total_beats}")

    in_name  = settings.get('midi_in')
    out_name = settings.get('midi_out')
    if not in_name:
        names = mido.get_input_names()
        if not names:
            print("No MIDI input devices found."); return
        i = choose_from_list("Select MIDI IN device", names, allow_quit=True)
        if i < 0: return
        in_name = names[i]; settings['midi_in'] = in_name; save_settings(settings)
    if not out_name:
        names = mido.get_output_names()
        if not names:
            print("No MIDI output devices found."); return
        i = choose_from_list("Select MIDI OUT device", names, allow_quit=True)
        if i < 0: return
        out_name = names[i]; settings['midi_out'] = out_name; save_settings(settings)

    bpm = float(settings.get('bpm_capture', 120.0))
    mon_ch_0based = (settings.get('monitor_channel', 1) - 1)
    mon_channel_for_echo = mon_ch_0based if settings.get('enable_monitor', True) else None

    # CAPTURE PHASE — open output ONCE for metronome/echo, then close
    try:
        with mido.open_output(out_name) as out_cap:
            if mon_channel_for_echo is not None:
                out_cap.send(mido.Message('program_change', channel=mon_channel_for_echo, program=71))
            rtc = RealTimeCapture(in_name, out_cap, bpm=bpm, monitor_channel=mon_channel_for_echo)
            melody = rtc.run_until_idle_2bars()
    except Exception as e:
        print("Failed during capture (MIDI OUT open):", e); return

    if not melody:
        print("No notes captured."); return
    total_beats = max(n.end_beat for n in melody)
    print(f"Captured melody: {len(melody)} notes, {total_beats:.2f} beats")

    # Key selection
    offset, key_label = choose_key_offset(melody)
    semi_to_C = (12 - offset) % 12
    melody_C  = transpose_notes(melody, semi_to_C)
    print(f"[Key] {key_label} | melody_to_C=+{semi_to_C} semitones")

    chosen_in_C = choose_progressions_for_blocks(melody_C, progs, chord_table_raw, pc_names)
    resolved_C   = [resolve_progression_from_C(p, chord_table_raw) for p in chosen_in_C]
    resolved_key = [transpose_resolved_progression(rp, offset) for rp in resolved_C]
    rendered_out, chosen_names = assemble_from_resolved_progressions(melody, resolved_key, pat_bass, pat_comp)

    print("Chosen progressions per 8-bar block:")
    for i, name in enumerate(chosen_names):
        print(f"  block {i+1}: {name}")

    # Loudness from settings
    global LOUDNESS_PCT_MELODY, LOUDNESS_PCT_COMPING, LOUDNESS_PCT_BASS
    LOUDNESS_PCT_MELODY  = settings.get('loudness_melody', 100)
    LOUDNESS_PCT_COMPING = settings.get('loudness_comping', 100)
    LOUDNESS_PCT_BASS    = settings.get('loudness_bass', 100)

    # Playback phase — open output again AFTER capture is fully closed
    double_flag = settings.get('double_melody', False)
    mel_ch_for_playback = mon_ch_0based if settings.get('enable_monitor', True) else CH_MELODY_DEFAULT
    mel_evs = melody_to_events(melody, mel_ch_for_playback, double_up=double_flag)
    combined = RenderedSong(events=sorted(rendered_out.events + mel_evs, key=_ev_sort_key),
                            total_beats=rendered_out.total_beats)

    ts = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, f'harmonized_realtime_{ts}.mid')
    export_mid(rendered_out, melody, bpm, outpath, monitor_channel=mel_ch_for_playback)
    print(f"Saved: {outpath}")

    try:
        with mido.open_output(out_name) as out_play:
            # timbres
            out_play.send(mido.Message('program_change', channel=mel_ch_for_playback, program=71))
            out_play.send(mido.Message('program_change', channel=CH_COMPING, program=0))
            out_play.send(mido.Message('program_change', channel=CH_BASS, program=0))

            player = Player(out_play, bpm=bpm, monitor_channel=None)
            t = player.play_events_async(combined)
            print("Playing... (Press ENTER / Space / Q to stop early)")
            if msvcrt is not None:
                while t.is_alive():
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch in ('\r', '\n', ' ', 'q', 'Q'):
                            player.stop(); break
                    time.sleep(0.03)
            else:
                t.join()
    except Exception as e:
        print("Failed to open MIDI OUT for playback:", e)

# ──────────────────────────────────────────────────────────────────────────────
# Main menu
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Harmonizator v0.5.0 — key-invariant, numeric-map transpose, settings persist")
    try:
        pc_names, chord_table_raw = load_chord_table('./chord.CSV')
        progs = load_prog16('./prog16.CSV')
    except Exception as e:
        print("Failed to load chord/prog tables:", e); return
    print(f"Loaded chord table: {len(chord_table_raw)} chords; progressions: {len(progs)}")

    settings = load_settings()

    while True:
        print("\n== Main Menu ==")
        print("  [0] Run: From MIDI file")
        print("  [1] Run: Real-time capture")
        print("  [2] Settings")
        print("  [q] Quit")
        r = input("Select index: ").strip().lower()
        if r == 'q':
            print("Bye."); break
        if not r.isdigit():
            print("Invalid."); continue
        i = int(r)
        if i == 0:
            run_from_midifile(settings, progs, chord_table_raw, pc_names)
        elif i == 1:
            run_realtime(settings, progs, chord_table_raw, pc_names)
        elif i == 2:
            settings_menu(settings)
        else:
            print("Invalid index.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
