/* main.js — HarmonyCamp frontend glue (final)
 * - Auto-detect backend API base
 * - Load genres
 * - Send /harmonize
 * - Show key label and chosen progression log
 * - Keep existing playback logic intact; optional <audio id="player"> support if audio_url returned
 */

(function () {
  // ====== DOM Helpers ======
  const $ = (sel) => document.querySelector(sel);

  // Optional elements (존재하지 않아도 동작)
  const elFile     = $('#midiFile')    || $('#file') || null;
  const elTempo    = $('#tempo')       || $('#bpm')  || null;
  const elGenre    = $('#genre')       || $('#genreSelect') || null;
  const elKeyLabel = $('#keyLabel')    || null;
  const elStatus   = $('#status')      || null;
  const elBtnHarm  = $('#btnHarmonize')|| $('#harmonizeBtn') || null;
  const elBtnPlay  = $('#btnPlay')     || null;
  const elBtnStop  = $('#btnStop')     || null;
  const elDnld     = $('#btnDownload') || $('#downloadBtn') || null;
  const elBackend  = $('#backendLabel')|| null; // 있으면 현재 API 표시
  const elAudio    = $('#player')      || null; // <audio id="player"> 있으면 사용

  // 없으면 만들어 붙임(페이지 하단)
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
    elLog.style.maxHeight = '220px';
    elLog.style.overflow = 'auto';
    document.body.appendChild(elLog);
  }
  const log = (...msgs) => {
    const line = msgs.map(x => typeof x === 'object' ? JSON.stringify(x) : String(x)).join(' ');
    elLog.textContent += (line + '\n');
    elLog.scrollTop = elLog.scrollHeight;
    console.debug('[HC]', ...msgs);
  };

  const setStatus = (s) => { if (elStatus) elStatus.textContent = s; log(s); };

  // ====== Backend auto-detect ======
  const FALLBACK_RENDER_URL = 'https://harmonycamp-backend.onrender.com'; // 배포한 Render URL로 바꾸면 더 빨라짐(선택)
  const HEALTH_PATH = '/health';

  async function ping(base) {
    try {
      const url = base.replace(/\/+$/, '') + HEALTH_PATH;
      const r = await fetch(url, { method: 'GET', mode: 'cors' });
      if (!r.ok) return false;
      const text = await r.text();
      return text.trim().toLowerCase().includes('ok');
    } catch (e) {
      return false;
    }
  }

  async function autoDetectBackend() {
    // 1) 수동 지정(전역/로컬 저장)
    if (window.HARMONY_BACKEND) {
      if (await ping(window.HARMONY_BACKEND)) return window.HARMONY_BACKEND.replace(/\/+$/, '');
    }
    const saved = localStorage.getItem('backendBase');
    if (saved && await ping(saved)) return saved.replace(/\/+$/, '');

    // 2) 개발/로컬
    const host = location.hostname;
    if (/^(localhost|127\.0\.0\.1)$/i.test(host)) {
      const local = 'http://127.0.0.1:8000';
      if (await ping(local)) return local;
    }

    // 3) 동일 오리진 프록시(/api) — GitHub Pages는 보통 불가하지만 혹시 모름
    const sameOriginApi = location.origin.replace(/\/+$/, '') + '/api';
    if (await ping(sameOriginApi)) return sameOriginApi;

    // 4) Render 기본값(수정 가능)
    if (await ping(FALLBACK_RENDER_URL)) return FALLBACK_RENDER_URL.replace(/\/+$/, '');

    return null;
  }

  let API_BASE = null;

  // ====== API ======
  async function getJSON(url) {
    const r = await fetch(url, { method: 'GET' });
    if (!r.ok) throw new Error(`GET ${url} -> ${r.status}`);
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
    try {
      const data = await getJSON(API_BASE + '/genres');
      const list = Array.isArray(data) ? data : (data.genres || []);
      if (!list.length) throw new Error('empty genres');
      sel.innerHTML = '';
      for (const g of list) {
        const opt = document.createElement('option');
        opt.value = g; opt.textContent = g;
        sel.appendChild(opt);
      }
      log('Genres loaded:', list);
    } catch (e) {
      log('Failed to load /genres, using fallback:', e.message);
      sel.innerHTML = '<option value="basic">basic</option>';
    }
  }

  // ====== Harmonize ======
  async function doHarmonize() {
    if (!API_BASE) {
      alert('Backend API not detected. Check your Render URL or run backend locally.');
      return;
    }
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

    setStatus('Harmonizing… (업로드 중)');

    try {
      const res = await postForm(API_BASE + '/harmonize', fd);

      // Expect: { ok, key_label, chosen_progressions, midi_url, audio_url?, total_beats? }
      if (res.key_label && elKeyLabel) {
        elKeyLabel.textContent = res.key_label;
      }
      // 선택된 코드 진행 로그
      if (Array.isArray(res.chosen_progressions) && res.chosen_progressions.length) {
        log('— Selected progressions (per 8-bar block) —');
        res.chosen_progressions.forEach((name, idx) => {
          log(`  Block ${idx + 1}: ${name}`);
        });
      } else {
        log('No chosen_progressions in response (or empty).');
      }

      // 다운로드 버튼 연결
      if (elDnld && res.midi_url) {
        elDnld.href = res.midi_url;
        elDnld.download = (midiFile.name.replace(/\.(mid|midi)$/i, '') || 'harmonized') + '_HC.mid';
        elDnld.removeAttribute('disabled');
      }

      // 오디오 재생(선택) — <audio id="player">가 있을 때만
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
      elBtnHarm.addEventListener('click', (e) => {
        e.preventDefault();
        doHarmonize();
      });
    }

    // 파일 선택시 표시
    if (elFile) {
      elFile.addEventListener('change', () => {
        if (elFile.files && elFile.files[0]) {
          log('Selected file:', elFile.files[0].name);
        }
      });
    }

    // 백엔드 수동 설정 UI(옵션)
    // index.html에 <input id="backendInput">, <button id="backendSave">가 있으면 저장 지원
    const elBackendInput = $('#backendInput');
    const elBackendSave  = $('#backendSave');
    if (elBackendInput && elBackendSave) {
      elBackendSave.addEventListener('click', async () => {
        const v = (elBackendInput.value || '').trim();
        if (!v) { alert('빈 값입니다.'); return; }
        if (await ping(v)) {
          API_BASE = v.replace(/\/+$/, '');
          localStorage.setItem('backendBase', API_BASE);
          if (elBackend) elBackend.textContent = API_BASE;
          setStatus('Backend saved: ' + API_BASE);
          await loadGenres();
        } else {
          alert('해당 주소로 /health 확인 실패: ' + v);
        }
      });
    }
  }

  // ====== Boot ======
  window.addEventListener('DOMContentLoaded', async () => {
    wireUI();
    setStatus('Backend 탐지 중…');
    API_BASE = await autoDetectBackend();

    if (!API_BASE) {
      setStatus('Backend 탐지 실패. 상단/설정에서 백엔드 URL 직접 입력 후 저장하세요.');
      log('Tip) Render로 배포했다면 예: https://<your-app>.onrender.com');
    } else {
      if (elBackend) elBackend.textContent = API_BASE;
      setStatus('Backend 연결: ' + API_BASE);
      await loadGenres();
    }
  });
})();
