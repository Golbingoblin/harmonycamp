const BACKEND_BASE =
  window.location.hostname.includes('localhost') || window.location.hostname.includes('127.0.0.1')
    ? 'http://127.0.0.1:8000'
    : 'https://harmonycamp.onrender.com';

// HarmonyCamp frontend – static, uses SoundFont in browser
const $ = (sel) => document.querySelector(sel);
const log = (x) => { const el=$("#log"); el.textContent += x+"\n"; el.scrollTop=el.scrollHeight; };

let API_BASE = $("#apiBase").value;
$("#apiBase").addEventListener("change", e => API_BASE = e.target.value);

// simple ping
$("#btnPing").onclick = async () => {
  try {
    const r = await fetch(`${API_BASE}/health`); $("#pingStatus").textContent = r.ok ? "OK" : "NG";
  } catch { $("#pingStatus").textContent = "ERR"; }
};

// get genres
async function loadGenres(){
  try {
    const r = await fetch(`${API_BASE}/list-genres`); const j = await r.json();
    const g = $("#genre"); g.innerHTML = ""; (j.genres||["basic"]).forEach(s=>{
      const o=document.createElement("option"); o.value=o.textContent=s; g.appendChild(o);
    });
  } catch (e) { log("list-genres error: "+e); }
}
loadGenres();

// state from backend
let LAST = null;  // {bpm,bar_beats,bars_shown, chords, melody, comping, bass, download_url, key_label?, progression_names?}

// canvas drawing (very simple piano-roll)
// lanes: bar grid, chord rail, melody, comping, bass
function drawTimeline(){
  const cvs = $("#canvas"), ctx = cvs.getContext("2d");
  const W = cvs.width, H = cvs.height;
  ctx.clearRect(0,0,W,H);
  ctx.font = "12px sans-serif";

  if(!LAST){
    ctx.fillStyle="#222"; ctx.fillText("Upload a MIDI and click Harmonize.", 20, 20);
    return;
  }

  const barBeats = LAST.bar_beats;
  const bars = LAST.bars_shown;
  const beatsTotal = Math.max(1, barBeats * bars);

  // layout rows
  const ROWS = [
    {name:"bars",   y: 20,  h: 40},
    {name:"chord",  y: 70,  h: 40},
    {name:"melody", y: 120, h: 70, color:"#2f80ed"},
    {name:"comp",   y: 200, h: 70, color:"#10b981"},
    {name:"bass",   y: 280, h: 70, color:"#ef4444"},
  ];
  const leftPad = 60, rightPad = 20;
  const innerW = W - leftPad - rightPad;

  const xOf = (beats) => leftPad + innerW * (beats / beatsTotal);

  // bar grid
  ctx.strokeStyle="#ddd"; ctx.lineWidth=1;
  for(let b=0;b<=bars;b++){
    const x = xOf(b*barBeats);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    ctx.fillStyle="#666"; ctx.fillText(`${b+1}`, x+3, ROWS[0].y+12);
  }
  // beat ticks
  ctx.strokeStyle="#f1f1f1";
  for(let k=0;k<beatsTotal;k++){
    const x = xOf(k);
    ctx.beginPath(); ctx.moveTo(x, ROWS[0].y); ctx.lineTo(x, H); ctx.stroke();
  }

  // chord rail
  (LAST.chords||[]).forEach(c=>{
    const x0=xOf(c.start_beats), x1=xOf(c.end_beats);
    ctx.fillStyle="#fff4cc"; ctx.strokeStyle="#eab308";
    ctx.fillRect(x0, ROWS[1].y, x1-x0, ROWS[1].h-10);
    ctx.strokeRect(x0, ROWS[1].y, x1-x0, ROWS[1].h-10);
    ctx.fillStyle="#000"; ctx.fillText(c.label, x0+4, ROWS[1].y+18);
  });

  // note lanes (rough line segments by pitch)
  function drawNotes(notes, row, col){
    if(!notes) return;
    const pMin=36, pMax=84; // normalize to visual range
    ctx.strokeStyle=col; ctx.lineWidth=3; ctx.globalAlpha=0.85;
    notes.forEach(n=>{
      const x0=xOf(n.start), x1=xOf(n.end);
      const y=row.y + row.h - ((n.pitch-pMin)/(pMax-pMin)) * (row.h-10) - 5;
      ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke();
    });
    ctx.globalAlpha=1;
  }
  drawNotes(LAST.melody, ROWS[2], ROWS[2].color);
  drawNotes(LAST.comping, ROWS[3], ROWS[3].color);
  drawNotes(LAST.bass,   ROWS[4], ROWS[4].color);

  // left captions
  ctx.fillStyle="#333"; ctx.fillText("Bars", 10, ROWS[0].y+18);
  ctx.fillText("Chord",10, ROWS[1].y+18);
  ctx.fillText("Mel",  10, ROWS[2].y+18);
  ctx.fillText("Comp", 10, ROWS[3].y+18);
  ctx.fillText("Bass", 10, ROWS[4].y+18);

  // top-right key display (also mirrors into #keyLabel if present)
  const keyTxt = LAST.key_label || LAST.key || (LAST.key_detected && LAST.key_detected.label) || "";
  if (keyTxt){
    ctx.fillStyle = "#111";
    ctx.fillText(`Key: ${keyTxt}`, W - 120, 16);
  }
}

// request harmonize
$("#btnHarmonize").onclick = async () => {
  const f = $("#midiFile").files[0]; if(!f){ alert("MIDI 파일을 선택하세요"); return; }
  const fd = new FormData();
  fd.append("file", f);
  fd.append("genre", $("#genre").value);
  fd.append("bpm_override", $("#tempo").value);
  fd.append("max_bars", $("#maxBars").value);
  fd.append("key_mode", $("#keyMode").value);
  fd.append("manual_key", $("#manualKey").value || "");

  $("#btnDownload").style.display="none";
  log("Uploading…");
  const r = await fetch(`${API_BASE}/analyze`, { method:"POST", body: fd });
  if(!r.ok){ const t=await r.text(); log("analyze error: "+t); alert("Analyze 실패"); return; }
  LAST = await r.json();

  // --- UI reflects key & chosen progressions ---
  const keyTxt = LAST.key_label || LAST.key || (LAST.key_detected && LAST.key_detected.label) || "(unknown)";
  const keyEl = $("#keyLabel"); if (keyEl) keyEl.textContent = keyTxt;
  log(`Analyze OK. BPM=${LAST.bpm} bars=${LAST.bars_shown}  Key=${keyTxt}`);

  // Try to find progression names array in several common shapes
  const progNames =
    LAST.progression_names ||
    (LAST.debug && LAST.debug.progression_names) ||
    LAST.chosen ||
    LAST.progs ||
    null;

  if (Array.isArray(progNames) && progNames.length){
    log("Chosen progressions per block:");
    progNames.forEach((nm, i)=> log(`  block ${i+1}: ${nm}`));
  } else {
    log("(No progression name array found in response)");
  }

  drawTimeline();
  const a = $("#btnDownload");
  if (LAST.download_url){
    a.href = `${API_BASE}${LAST.download_url}`; a.style.display="inline-block";
  } else {
    a.style.display = "none";
  }
};

// ── playback with SoundFont (WebAudio) ──────────────────────────────────────
const AudioContext = window.AudioContext || window.webkitAudioContext;
const ctx = new AudioContext();

const masterGain = ctx.createGain(); masterGain.gain.value = +$("#volMaster").value; masterGain.connect(ctx.destination);
const gMel = ctx.createGain(), gComp = ctx.createGain(), gBass = ctx.createGain();
gMel.connect(masterGain); gComp.connect(masterGain); gBass.connect(masterGain);
// metronome: separate from master (not affected)
const gMet = ctx.createGain(); gMet.gain.value = 0.7; gMet.connect(ctx.destination);

// sliders & mute
$("#volMaster").oninput = e => masterGain.gain.value = +e.target.value;
$("#volMel").oninput = e => gMel.gain.value = $("#muteMel").checked ? 0 : +e.target.value;
$("#volComp").oninput = e => gComp.gain.value = $("#muteComp").checked ? 0 : +e.target.value;
$("#volBass").oninput = e => gBass.gain.value = $("#muteBass").checked ? 0 : +e.target.value;
$("#muteMel").onchange = () => $("#volMel").dispatchEvent(new Event("input"));
$("#muteComp").onchange = () => $("#volComp").dispatchEvent(new Event("input"));
$("#muteBass").onchange = () => $("#volBass").dispatchEvent(new Event("input"));

// instruments via CDN (quick path)
let instMel=null, instComp=null, instBass=null;
async function ensureInstruments(){
  if(!instMel)  { instMel  = await window.Soundfont.instrument(ctx, 'acoustic_grand_piano'); instMel.connect(gMel); }
  if(!instComp) { instComp = await window.Soundfont.instrument(ctx, 'acoustic_grand_piano'); instComp.connect(gComp); }
  if(!instBass) { instBass = await window.Soundfont.instrument(ctx, 'acoustic_bass');        instBass.connect(gBass); }
}

// scheduling
let playing = false;
function scheduleNote(inst, when, midi, dur, vel=0.9){
  // soundfont-player handles its own envelopes; instrument already routed to part gain
  inst.play(midi, when, {gain: vel, duration: dur, adsr: [0.002,0.02,0.7,0.05]});
}

function stopAll(){
  try{ instMel && instMel.stop(); }catch{}
  try{ instComp && instComp.stop(); }catch{}
  try{ instBass && instBass.stop(); }catch{}
  playing = false;
}

function scheduleMetronome(startTime, bpm, bars, barBeats){
  if(!$("#metronome").checked) return;
  const secPerBeat = 60/bpm;
  for(let b=0;b<bars*barBeats;b++){
    const t = startTime + b*secPerBeat;
    const osc = ctx.createOscillator(); const g = ctx.createGain();
    const isDown = (b % barBeats)===0;
    g.gain.setValueAtTime(isDown?0.25:0.12, t);
    g.gain.exponentialRampToValueAtTime(0.0001, t+0.04);
    osc.frequency.value = isDown?1400:900;
    osc.connect(g); g.connect(gMet);
    osc.start(t); osc.stop(t+0.05);
  }
}

$("#btnPlay").onclick = async () => {
  if(!LAST){ alert("먼저 Harmonize 하세요."); return; }
  await ctx.resume();            // unlock audio (autoplay policies)
  await ensureInstruments();     // lazy-load SF2s
  const bpm = +$("#tempo").value || LAST.bpm;
  const secPerBeat = 60/bpm;
  const start = ctx.currentTime + 0.1;

  const bars = LAST.bars_shown, barBeats = LAST.bar_beats;
  scheduleMetronome(start, bpm, bars, barBeats);

  function sched(arr, inst){
    (arr||[]).forEach(n=>{
      const t = start + n.start*secPerBeat;
      const d = Math.max(0.02, (n.end-n.start)*secPerBeat);
      const v = Math.min(1, (n.velocity||100)/100);
      scheduleNote(inst, t, n.pitch, d, v);
    });
  }
  if (!$("#muteMel").checked)  sched(LAST.melody, instMel);
  if (!$("#muteComp").checked) sched(LAST.comping, instComp);
  if (!$("#muteBass").checked) sched(LAST.bass,   instBass);

  playing = true;
};

$("#btnStop").onclick = stopAll;

// redraw on resize
window.addEventListener("resize", drawTimeline);
