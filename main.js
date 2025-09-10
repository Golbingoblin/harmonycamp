/* main.js — HarmonyCamp frontend glue (fixed backend)
 * - Backend 고정: https://harmonycamp.onrender.com  (localStorage로 덮어쓸 수 있음)
 * - 장르 로드(/list-genres → 실패시 /genres)
 * - /harmonize 호출
 * - Key 라벨 표시, 선택된 코드 진행 로그 표시
 * - 응답에 midi_url, audio_url 있으면 다운로드/재생 연결
 */

(function () {
  'use strict';

  // ====== DOM Helpers ======
  const $ = (sel) => document.querySelector(sel);

  // Optional elements (없어도 동작)
  const elFile     = $('#midiFile')    || $('#file') || null;
  const elTempo    = $('#tempo')       || $('#bpm')  || null;
  const elGenre    = $('#genre')       || $('#genreSelect') || null;
  const elKeyLabel = $('#keyLabel')    || null;
  const elStatus   = $('#status')      || null;
  const elBtnHarm  = $('#btnHarmonize')|| $('#harmonizeBtn') || null;
  const elDnld     = $('#btnDownload') || $('#downloadBtn') || null;
  const elBackend  = $('#backendLabel')|| null; // 현재 API 표시용
  const elAudio    = $('#player')      || null; // <audio id="player">면 사용

  // 로그 영역이 없으면 생성
  let elLog = $('#debugLog');
  if (!elLog) {
    elLog = document.createElement('pre');
    elLog.id = 'debugLog';
    elLog.style.whiteSpace = 'pre-wrap';
    elLog.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace';
    elLog.style.fontSize = '12px';
    elLog.style.padding = '8px';
    elLog.style.border = '1px solid #ddd';
    elLog.style.borderRadius = '6px';
    elLog.style.background = '#fafafa';
    elLog.style.maxHeight = '240px';
    elLog.style.overflow = 'auto';
    document.body.appendChild(elLog);
  }

  const log = (...msgs) => {
    const line = msgs.map(x => typeof x === 'object' ? JSON.stringify(x) : String(x)).join(' ');
    elLog.textContent += (line + '\n');
    elLog.scrollTop = elLog.scrollHeight;
    console.debug('[HarmonyCamp]', ...msgs);
  };
  const setStatus = (s) => { if (elStatus) elStatus.textContent = s; log(s); };

  // ====== BACKEND 설정 (자동탐지 제거, 고정 + 수동 저장 지원) ======
  const DEFAULT_BACKEND = "https://harmonycamp.onrender.com"; // 배포 백엔드
  let API_BASE = (localStorage.getItem("backendBase") || DEFAULT_BACKEND).replace(/\/+$/, "");

  // 페이지에 수동설정 UI가 있는 경우(<input id="backendInput">, <button id="backendSave">)
  function wireBackendSettings() {
    const elBackendInput = $('#backendInput');
    const elBackendSave  = $('#backendSave');
    if (elBackendInput) elBackendInput.value = API_BASE;
    if (elBackend) elBackend.textContent = API_BASE;

    if (elBackendInput && elBackendSave) {
      elBackendSave.addEventListener('click', () => {
        const v = (elBackendInput.value || '').trim().replace(/\/+$/, '');
        if (!v) { alert('백엔드 URL을 입력하세요.'); return; }
        API_BASE = v;
        localStorage.setItem('backendBase', API_BASE);
        if (elBackend) elBackend.textContent = API_BASE;
        setStatus('Backend saved: ' + API_BASE);
        loadGenres().catch(()=>{});
      });
    }
  }

  // ====== HTTP ======
  async function getJSON(url) {
    const r = await fetch(url, { method: 'GET' });
    if (!r.ok) throw new Error(`GET ${url} -> ${r.status}`);
    // /health 처럼 text일 수 있으니 시도
    const ct = r.headers.get('content-type') || '';
    if (!ct.includes('application/json')) {
      const t = await r.text();
      try { return JSON.parse(t); } catch { return { ok: t }; }
    }
    return r.json();
  }

  async function postForm(url, formData) {
    const r = await fetch(url, { method: 'POST', body: formData });
    if (!r.ok) {
      let detail = '';
      try { detail = await r.text(); } catch {}
      throw new Error(`POST ${url} -> ${r.status} ${detail}`);
    }
    return r.json();
  }

  // ====== Genres ======
  async function loadGenres() {
    const sel = elGenre;
    if (!sel) { log('genre select not found; skip'); return; }
    sel.innerHTML = '<option value="">(로딩중…)</option>';

    // 우선 /list-genres (기존 백엔드) → 실패시 /genres
    try {
      let list = [];
      try {
        const data1 = await getJSON(API_BASE + '/list-genres');
        list = data1.genres || data1 || [];
      } catch {
        const data2 = await getJSON(API_BASE + '/genres');
        list = Array.isArray(data2) ? data2 : (data2.genres || []);
      }
      if (!list.length) throw new Error('empty genres');

      sel.innerHTML = '';
      for (const g of list) {
        const opt = document.createElement('option');
        opt.value = g; opt.textContent = g;
        sel.appendChild(opt);
      }
      if (!sel.value && sel.options.length) sel.value = sel.options[0].value;
      log('Genres loaded:', list);
    } catch (e) {
      log('Failed to load genres, using fallback "basic":', e.message);
      sel.innerHTML = '<option value="basic">basic</option>';
    }
  }

  // ====== Harmonize ======
  async function doHarmonize() {
    if (!elFile || !elFile.files || !elFile.files[0]) {
      alert('MIDI 파일을 선택하세요.');
      return;
    }
    const midiFile = elFile.files[0];
    const bpm = (elTempo && elTempo.value) ? String(elTempo.value) : '120';
    const genre = (elGenre && elGenre.value) ? elGenre.value : 'basic';

    const fd = new FormData();
    fd.append('midi_file', midiFile, midiFile.name);
    fd.append('bpm', bpm);
    fd.append('genre', genre);

    setStatus('Harmonizing… (업로드/처리 중)');

    try {
      const res = await postForm(API_BASE + '/harmonize', fd);
      // 기대 포맷:
      // { ok:true, key_label:"Auto: C major", chosen_progressions:["..."], midi_url:"...", audio_url?: "...", total_beats?: number }

      if (res.key_label && elKeyLabel) {
        elKeyLabel.textContent = res.key_label;
      }
      if (Array.isArray(res.chosen_progressions) && res.chosen_progressions.length) {
        log('— Selected progressions (per 8-bar block) —');
        res.chosen_progressions.forEach((name, idx) => log(`  Block ${idx + 1}: ${name}`));
      } else {
        log('No chosen_progressions in response (or empty).');
      }

      if (elDnld && res.midi_url) {
        elDnld.href = res.midi_url;
        elDnld.download = (midiFile.name.replace(/\.(mid|midi)$/i, '') || 'harmonized') + '_HC.mid';
        elDnld.removeAttribute('disabled');
      }

      if (elAudio && res.audio_url) {
        try {
          elAudio.src = res.audio_url;
          await elAudio.play();
          log('Audio started via <audio> element.');
        } catch (e) {
          log('Audio element play failed:', e.message);
        }
      }

      setStatus('완료: Harmonized.');
    } catch (e) {
      log('Harmonize failed:', e.message);
      setStatus('오류: Harmonize 실패. 콘솔/로그 확인.');
      alert('Harmonize 실패: ' + e.message);
    }
  }

  // ====== Wire UI ======
  function wireUI() {
    if (elBtnHarm) {
      elBtnHarm.addEventListener('click', (e) => { e.preventDefault(); doHarmonize(); });
    }
    if (elFile) {
      elFile.addEventListener('change', () => {
        if (elFile.files && elFile.files[0]) log('Selected file:', elFile.files[0].name);
      });
    }
  }

  // ====== Boot ======
  window.addEventListener('DOMContentLoaded', async () => {
    wireBackendSettings();
    wireUI();
    if (elBackend) elBackend.textContent = API_BASE;
    setStatus('Backend 연결: ' + API_BASE);
    await loadGenres();
  });
})();
