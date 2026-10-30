(() => {
  'use strict';

  // ---------- Elements ----------
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('overlay');
  const menuBtn = document.getElementById('menuBtn');
  const newChatBtn = document.getElementById('newChatBtn');
  const exportBtn = document.getElementById('exportBtn');
  const chatList = document.getElementById('chatList');
  const scrollArea = document.getElementById('scrollArea');
  const emptyState = document.getElementById('emptyState');
  const messagesEl = document.getElementById('messages');
  const inputEl = document.getElementById('input');
  const sendBtn = document.getElementById('sendBtn');
  const sendIcon = document.getElementById('sendIcon');
  const stopIcon = document.getElementById('stopIcon');
  const micBtn = document.getElementById('micBtn');
  const ctxMenu = document.getElementById('ctxMenu');
  const ctxDelete = document.getElementById('ctxDelete');
  const attachBtn = document.getElementById('attachBtn');
  const fileInput = document.getElementById('fileInput');
  const attachChips = document.getElementById('attachChips');
  const attachMenu = document.getElementById('attachMenu');
  const docsStrip = document.getElementById('docsStrip');
  const webSearchModeEl = document.getElementById('webSearchMode');

  // ---------- Persistence ----------
  const STORE_KEY = 'astra_chats_v1';
  const MODEL_MODE_KEY = 'astra_model_mode';
  const WEB_SEARCH_KEY = 'astra_web_search';
  const MAX_TITLE = 34;
  const PLACEHOLDER = 'Message Astra AI...';

  // ---------- State ----------
  let chats = [];
  let activeId = null;
  let nextId = 1;
  let busy = false;
  let controller = null;
  let speakingBtn = null;
  let transcribing = false;
  let hintTimer = null;
  let pendingFiles = [];  // {id, name, kind: 'image'|'doc', mime?, data?, text?, thumb?, loading?}

  // ---------- Attachments ----------
  // Groq qwen3.8-27b vision limit: max 3 images, ~4MB each (see console.groq.com/docs/vision)
  const MAX_FILES = 5;
  const MAX_IMAGES = 3;
  const MAX_IMG_BYTES = 4 * 1024 * 1024;
  let fileIdSeq = 1;

  function renderChips() {
    if (!pendingFiles.length) {
      attachChips.style.display = 'none';
      attachChips.replaceChildren();
      return;
    }
    attachChips.style.display = '';
    attachChips.replaceChildren();

    pendingFiles.forEach(f => {
      const chip = document.createElement('div');
      chip.className = 'attach-chip' + (f.loading ? ' loading' : '');

      if (f.kind === 'image' && f.thumb) {
        const img = document.createElement('img');
        img.src = f.thumb;
        img.alt = f.name;
        chip.appendChild(img);
      } else {
        const icon = document.createElement('span');
        icon.textContent = f.kind === 'doc' ? '📄' : '📎';
        chip.appendChild(icon);
      }

      const name = document.createElement('span');
      name.className = 'chip-name';
      name.textContent = f.loading ? 'Loading…' : f.name;
      name.title = f.name;
      chip.appendChild(name);

      const remove = document.createElement('button');
      remove.className = 'chip-remove';
      remove.textContent = '×';
      remove.title = 'Remove';
      remove.addEventListener('click', () => {
        pendingFiles = pendingFiles.filter(x => x.id !== f.id);
        renderChips();
      });
      chip.appendChild(remove);

      attachChips.appendChild(chip);
    });
  }

  // ---------- Docs strip (RAG — per-conversation uploaded documents) ----------
  function renderDocsStrip() {
    const chat = activeChat();
    const docs = (chat && chat.docs) || [];
    if (!docs.length) {
      docsStrip.style.display = 'none';
      docsStrip.replaceChildren();
      return;
    }
    docsStrip.style.display = '';
    docsStrip.replaceChildren();
    docs.forEach(d => {
      const chip = document.createElement('div');
      chip.className = 'doc-chip' + (d.loading ? ' loading' : '');
      const icon = document.createElement('span');
      icon.textContent = '📄';
      chip.appendChild(icon);
      const name = document.createElement('span');
      name.className = 'chip-name';
      name.textContent = d.loading ? 'Indexing…' : d.name;
      name.title = d.name + (d.chunks ? ' · ' + d.chunks + ' chunks' : '');
      chip.appendChild(name);
      if (!d.loading && d.chunks) {
        const meta = document.createElement('span');
        meta.className = 'doc-meta';
        meta.textContent = d.chunks + ' chunks';
        chip.appendChild(meta);
      }
      const remove = document.createElement('button');
      remove.className = 'chip-remove';
      remove.textContent = '×';
      remove.title = 'Remove document';
      remove.disabled = !!d.loading;
      remove.addEventListener('click', () => removeDoc(d.doc_id));
      chip.appendChild(remove);
      docsStrip.appendChild(chip);
    });
  }

  function removeDoc(docId) {
    const chat = activeChat();
    if (!chat) return;
    const doc = (chat.docs || []).find(d => d.doc_id === docId);
    chat.docs = (chat.docs || []).filter(d => d.doc_id !== docId);
    renderDocsStrip();
    updateSendEnabled();
    save();
    if (doc) {
      fetch('/api/docs/' + encodeURIComponent(docId) +
        '?conversation_id=' + encodeURIComponent(chat.convId),
        { method: 'DELETE' }).catch(() => { /* TTL will clean up */ });
    }
  }

  async function loadDocsForChat(chat) {
    if (!chat || !chat.convId) return;
    try {
      const res = await fetch('/api/docs?conversation_id=' + encodeURIComponent(chat.convId));
      const data = await res.json();
      if (data.ok && Array.isArray(data.docs)) {
        // reconcile localStorage with the server store (TTL may have evicted)
        chat.docs = data.docs.filter(d => d.doc_id);
        save();
        if (chat.id === activeId) renderDocsStrip();
      }
    } catch (e) { /* offline — keep cached list */ }
  }

  async function uploadDoc(chat, file) {
    const doc = { doc_id: null, name: file.name, chunks: 0, loading: true };
    chat.docs = chat.docs || [];
    chat.docs.push(doc);
    renderDocsStrip();
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('conversation_id', chat.convId);
      const res = await fetch('/api/docs', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.ok) {
        doc.doc_id = data.doc_id;
        doc.name = data.name;
        doc.chunks = data.chunks;
        doc.chars = data.chars;
      } else {
        chat.docs = chat.docs.filter(d => d !== doc);
        setHint(data.error || 'Could not index file.');
      }
    } catch (e) {
      chat.docs = chat.docs.filter(d => d !== doc);
      setHint('Upload failed: ' + file.name);
    }
    doc.loading = false;
    renderDocsStrip();
    updateSendEnabled();
    save();
  }

  function makeThumb(file) {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        const scale = Math.min(1, 128 / Math.max(img.width, img.height));
        c.width = Math.round(img.width * scale);
        c.height = Math.round(img.height * scale);
        c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
        URL.revokeObjectURL(url);
        resolve(c.toDataURL('image/jpeg', 0.7));
      };
      img.onerror = () => { URL.revokeObjectURL(url); resolve(null); };
      img.src = url;
    });
  }

  function readAsDataURL(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = reject;
      r.readAsDataURL(file);
    });
  }

  async function addFiles(fileList) {
    const files = Array.from(fileList);
    if (!files.length) return;

    ensureChat();
    const chat = activeChat();

    for (const file of files) {
      if (pendingFiles.length >= MAX_FILES) {
        setHint('Max ' + MAX_FILES + ' files at once.');
        break;
      }

      const isImage = file.type.startsWith('image/');
      const isDoc = /\.(pdf|docx|txt|md|csv|json|log|py|js|html|css|xml|yml|yaml)$/i.test(file.name);

      if (!isImage && !isDoc) {
        setHint('Unsupported file: ' + file.name);
        continue;
      }

      if (!isImage && file.size > 10 * 1024 * 1024) {
        setHint('Document too large (max 10MB): ' + file.name);
        continue;
      }

      if (isImage) {
        const imgCount = pendingFiles.filter(f => f.kind === 'image').length;
        if (imgCount >= MAX_IMAGES) {
          setHint('Max ' + MAX_IMAGES + ' images per message.');
          continue;
        }
        if (file.size > MAX_IMG_BYTES) {
          setHint('Image too large (max 4MB): ' + file.name);
          continue;
        }
        const entry = { id: fileIdSeq++, name: file.name, kind: 'image', loading: true };
        pendingFiles.push(entry);
        renderChips();
        try {
          const [dataUrl, thumb] = await Promise.all([readAsDataURL(file), makeThumb(file)]);
          entry.data = dataUrl;
          entry.thumb = thumb;
          entry.mime = file.type || 'image/png';
          entry.loading = false;
        } catch (e) {
          pendingFiles = pendingFiles.filter(x => x.id !== entry.id);
          setHint('Could not read image: ' + file.name);
        }
        renderChips();
      } else {
        // RAG: index server-side, scoped to this conversation
        await uploadDoc(chat, file);
      }
    }
    fileInput.value = '';
  }

  function save() {
    try {
      // bump updatedAt on the active chat so date grouping stays fresh
      const ac = chats.find(c => c.id === activeId);
      if (ac && ac.messages && ac.messages.length) ac.updatedAt = new Date().toISOString();
      localStorage.setItem(STORE_KEY, JSON.stringify({ chats, activeId, nextId }));
    } catch (e) { /* storage unavailable */ }
  }

  // Conversation id: UUID matching server regex ^[A-Za-z0-9_-]{8,64}$
  function genConvId() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    return 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 12);
  }

  function load() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (!raw) return false;
      const data = JSON.parse(raw);
      if (!Array.isArray(data.chats) || !data.chats.length) return false;
      chats = data.chats;
      // backfill Phase 4 fields on chats saved before RAG existed
      chats.forEach(c => {
        if (!c.convId) c.convId = genConvId();
        if (!Array.isArray(c.docs)) c.docs = [];
        if (!c.createdAt) c.createdAt = new Date().toISOString();
        if (!c.updatedAt) c.updatedAt = c.createdAt;
      });
      activeId = data.activeId;
      nextId = data.nextId || chats.length + 1;
      if (!chats.find(c => c.id === activeId)) activeId = chats[0].id;
      return true;
    } catch (e) {
      return false;
    }
  }

  function newChatObject() {
    const now = new Date().toISOString();
    return { id: nextId++, title: 'New chat', messages: [], convId: genConvId(), docs: [], createdAt: now, updatedAt: now };
  }

  function activeChat() {
    return chats.find(c => c.id === activeId);
  }

  function getModelMode() {
    try {
      const v = localStorage.getItem(MODEL_MODE_KEY);
      if (v === 'fast' || v === 'reasoning' || v === 'auto') return v;
    } catch (e) { /* storage unavailable */ }
    return 'auto';
  }

  function loadModelMode() {
    // topbar selector removed — mode is chosen automatically; value only lives in localStorage
  }

  function getWebSearchMode() {
    return webSearchModeEl ? webSearchModeEl.value : 'auto';
  }

  function loadWebSearchMode() {
    if (!webSearchModeEl) return;
    try {
      const v = localStorage.getItem(WEB_SEARCH_KEY);
      if (v === 'always' || v === 'never' || v === 'auto') webSearchModeEl.value = v;
    } catch (e) { /* storage unavailable */ }
    webSearchModeEl.addEventListener('change', () => {
      try { localStorage.setItem(WEB_SEARCH_KEY, webSearchModeEl.value); } catch (e) { /* ignore */ }
    });
  }

  function ensureChat() {
    if (!chats.length) {
      const c = newChatObject();
      chats.push(c);
      activeId = c.id;
    }
    if (!activeChat()) activeId = chats[0].id;
  }

  // ---------- Voice settings (Phase 6) ----------
  const VOICE_LANG_KEY = 'astra_voice_lang';
  const AUTO_SPEAK_KEY = 'astra_auto_speak';
  const MAX_RECORD_SECONDS = 60;
  const voiceLangEl = document.getElementById('voiceLang');
  const autoSpeakToggle = document.getElementById('autoSpeakToggle');
  let autoSpeak = false;
  let voiceLang = 'auto';

  function loadVoiceSettings() {
    try {
      const v = localStorage.getItem(VOICE_LANG_KEY);
      if (v) voiceLang = v;
      autoSpeak = localStorage.getItem(AUTO_SPEAK_KEY) === '1';
    } catch (e) { /* storage unavailable */ }
    if (voiceLangEl) voiceLangEl.value = voiceLang;
    renderAutoSpeakBtn();
    if (voiceLangEl) {
      voiceLangEl.addEventListener('change', () => {
        voiceLang = voiceLangEl.value;
        try { localStorage.setItem(VOICE_LANG_KEY, voiceLang); } catch (e) { /* ignore */ }
        showSettingsToast('Voice language saved');
      });
    }
    if (autoSpeakToggle) {
      autoSpeakToggle.checked = autoSpeak;
      autoSpeakToggle.addEventListener('change', () => {
        autoSpeak = autoSpeakToggle.checked;
        try { localStorage.setItem(AUTO_SPEAK_KEY, autoSpeak ? '1' : '0'); } catch (e) { /* ignore */ }
        renderAutoSpeakBtn();
        if (!autoSpeak) stopSpeaking();
        showSettingsToast(autoSpeak ? 'Auto-read on' : 'Auto-read off');
      });
    }
  }

  function renderAutoSpeakBtn() {
    if (autoSpeakToggle) autoSpeakToggle.checked = autoSpeak;
  }

  // ---------- Phase 8: settings (theme, response language, modal) ----------
  const THEME_KEY = 'astra_theme';
  const REPLY_LANG_KEY = 'astra_reply_lang';
  const settingsBtn = document.getElementById('settingsBtn');
  const settingsModal = document.getElementById('settingsModal');
  const settingsClose = document.getElementById('settingsClose');
  const settingsMemBtn = document.getElementById('settingsMemBtn');
  const replyLangEl = document.getElementById('replyLang');
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  let replyLang = 'auto';
  let themePref = 'dark'; // 'dark' | 'light' | 'system'
  const settingsModelModeEl = document.getElementById('settingsModelMode');
  const settingsSearchSection = document.getElementById('settingsSearchSection');
  const settingsToast = document.getElementById('settingsToast');
  const settingsToastText = document.getElementById('settingsToastText');
  const settingsToastIcon = document.getElementById('settingsToastIcon');
  let settingsToastTimer = null;

  // toast feedback for settings operations
  function showSettingsToast(msg, ok) {
    if (!settingsToast) return;
    settingsToastText.textContent = msg;
    settingsToastIcon.textContent = ok === false ? '✕' : '✓';
    settingsToastIcon.className = ok === false ? 'err' : '';
    settingsToast.style.display = 'flex';
    clearTimeout(settingsToastTimer);
    settingsToastTimer = setTimeout(() => { settingsToast.style.display = 'none'; }, 2200);
  }

  // resolve 'system' to actual light/dark via matchMedia
  function resolveTheme(pref) {
    if (pref === 'system') {
      return (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) ? 'light' : 'dark';
    }
    return pref === 'light' ? 'light' : 'dark';
  }

  function applyTheme(pref) {
    themePref = pref || 'dark';
    const t = resolveTheme(themePref);
    document.documentElement.setAttribute('data-theme', t);
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', t === 'light' ? '#f4f6fb' : '#0b1020');
    // update settings seg-btns
    document.querySelectorAll('.seg-btn[data-theme]').forEach(b => {
      const active = b.dataset.theme === themePref;
      b.classList.toggle('active', active);
      b.setAttribute('aria-checked', active ? 'true' : 'false');
    });
    // update topbar toggle icons
    if (themeToggleBtn) {
      const iconDark = themeToggleBtn.querySelector('.theme-icon-dark');
      const iconLight = themeToggleBtn.querySelector('.theme-icon-light');
      if (iconDark) iconDark.style.display = t === 'dark' ? '' : 'none';
      if (iconLight) iconLight.style.display = t === 'light' ? '' : 'none';
    }
    try { localStorage.setItem(THEME_KEY, themePref); } catch (e) { /* ignore */ }
  }

  // topbar quick toggle: dark → light → dark (respects system when pref is system)
  function cycleTheme() {
    const t = resolveTheme(themePref);
    applyTheme(t === 'dark' ? 'light' : 'dark');
  }

  // listen for OS theme changes (only matters when pref is 'system')
  if (window.matchMedia) {
    const mq = window.matchMedia('(prefers-color-scheme: light)');
    const onSchemeChange = () => { if (themePref === 'system') applyTheme('system'); };
    if (mq.addEventListener) mq.addEventListener('change', onSchemeChange);
    else if (mq.addListener) mq.addListener(onSchemeChange);
  }

  function loadSettings() {
    try {
      const stored = localStorage.getItem(THEME_KEY) || 'dark';
      applyTheme(stored === 'light' || stored === 'system' ? stored : 'dark');
      const rl = localStorage.getItem(REPLY_LANG_KEY);
      if (rl) replyLang = rl;
    } catch (e) { applyTheme('dark'); }
    if (replyLangEl) replyLangEl.value = replyLang;
  }

  function openSettings() {
    if (!settingsModal) return;
    // sync settings model mode with stored preference
    if (settingsModelModeEl) settingsModelModeEl.value = getModelMode();
    settingsModal.style.display = '';
    settingsClose && settingsClose.focus();
  }

  function closeSettings() {
    if (settingsModal) settingsModal.style.display = 'none';
  }

  if (settingsBtn) settingsBtn.addEventListener('click', openSettings);
  if (settingsClose) settingsClose.addEventListener('click', closeSettings);
  if (settingsModal) {
    settingsModal.addEventListener('click', (e) => {
      if (e.target === settingsModal) closeSettings();
    });
  }
  if (settingsMemBtn) {
    settingsMemBtn.addEventListener('click', () => {
      closeSettings();
      openMemModal();
    });
  }
  // settings model mode → save to localStorage (topbar selector removed)
  if (settingsModelModeEl) {
    settingsModelModeEl.addEventListener('change', () => {
      try { localStorage.setItem(MODEL_MODE_KEY, settingsModelModeEl.value); } catch (e) { /* ignore */ }
      showSettingsToast('Response mode: ' + settingsModelModeEl.options[settingsModelModeEl.selectedIndex].text.split('—')[0].trim());
    });
  }
  document.querySelectorAll('.seg-btn[data-theme]').forEach(b => {
    b.addEventListener('click', () => applyTheme(b.dataset.theme));
  });
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', cycleTheme);
  }
  if (replyLangEl) {
    replyLangEl.addEventListener('change', () => {
      replyLang = replyLangEl.value;
      try { localStorage.setItem(REPLY_LANG_KEY, replyLang); } catch (e) { /* ignore */ }
    });
  }
  // Esc closes any open modal / menu
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (settingsModal && settingsModal.style.display !== 'none') closeSettings();
    else if (memModal && memModal.style.display !== 'none') closeMemModal();
    else if (attachMenu && attachMenu.style.display === 'block') closeAttachMenu();
  });

  // hide search settings section when the search provider isn't configured
  async function loadFeatureAvailability() {
    try {
      const res = await fetch('/health');
      const data = await res.json();
      const f = data.features || {};
      if (settingsSearchSection) settingsSearchSection.style.display = f.search ? '' : 'none';
    } catch (e) {
      if (settingsSearchSection) settingsSearchSection.style.display = 'none';
    }
  }


  // ---------- Voice: text-to-speech (server: Edge Neural voices) ----------
  const ttsAudio = new Audio();
  let ttsUrl = null;
  let ttsState = 'idle'; // idle | loading | playing | paused
  let ttsText = '';

  const SPK_ICON = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>';
  const PAUSE_ICON = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>';
  const DL_ICON = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>';

  let hintActive = false;

  // Speech uses the server TTS (Gemini, female voice) only — no browser voices.
  let speakToken = 0;

  function setHint(msg) {
    hintActive = true;
    inputEl.placeholder = msg;
    clearTimeout(hintTimer);
    hintTimer = setTimeout(() => {
      hintActive = false;
      if (!isRecording() && !transcribing) {
        inputEl.placeholder = PLACEHOLDER;
      }
    }, 3500);
  }

  function setSpeakBtn(btn, state) {
    if (!btn) return;
    btn.classList.remove('speaking', 'loading', 'paused');
    if (state === 'loading') {
      btn.classList.add('loading');
      btn.innerHTML = SPK_ICON + ' Loading…';
    } else if (state === 'playing') {
      btn.classList.add('speaking');
      btn.innerHTML = PAUSE_ICON + ' Pause';
    } else if (state === 'paused') {
      btn.classList.add('paused');
      btn.innerHTML = SPK_ICON + ' Resume';
    } else {
      btn.innerHTML = SPK_ICON + ' Listen';
    }
  }

  function resetSpeakBtn(btn) {
    ttsState = 'idle';
    ttsText = '';
    setSpeakBtn(btn, 'idle');
    if (speakingBtn === btn) speakingBtn = null;
  }

  function stopSpeaking() {
    speakToken++;  // abort any in-flight TTS fetch (prevents double playback)
    try {
      ttsAudio.pause();
      ttsAudio.removeAttribute('src');
      if (ttsUrl) {
        URL.revokeObjectURL(ttsUrl);
        ttsUrl = null;
      }
    } catch (e) { /* ignore */ }
    try { window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
    if (speakingBtn) resetSpeakBtn(speakingBtn);
    speakingBtn = null;
    ttsState = 'idle';
  }

  async function speakText(text, btn) {
    // same button while playing/paused → pause/resume toggle
    if (speakingBtn === btn && (ttsState === 'playing' || ttsState === 'paused')) {
      if (ttsState === 'playing') {
        ttsAudio.pause();
        ttsState = 'paused';
        setSpeakBtn(btn, 'paused');
      } else {
        await ttsAudio.play();
        ttsState = 'playing';
        setSpeakBtn(btn, 'playing');
      }
      return;
    }

    stopSpeaking();
    speakingBtn = btn || null;
    ttsText = text;
    if (btn) setSpeakBtn(btn, 'loading');
    ttsState = 'loading';
    const token = ++speakToken;

    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, lang: voiceLang })
      });

      if (!res.ok) throw new Error('tts failed');

      const blob = await res.blob();
      if (!blob.size) throw new Error('empty audio');
      if (token !== speakToken) return;  // superseded by a newer speak/stop

      if (ttsUrl) URL.revokeObjectURL(ttsUrl);
      ttsUrl = URL.createObjectURL(blob);

      setSpeakBtn(btn, 'playing');
      ttsState = 'playing';

      ttsAudio.src = ttsUrl;
      ttsAudio.onended = () => { if (token === speakToken) resetSpeakBtn(btn); };
      ttsAudio.onerror = () => { if (token === speakToken) resetSpeakBtn(btn); };
      await ttsAudio.play();
    } catch (e) {
      if (token !== speakToken) return;
      resetSpeakBtn(btn);
      setHint('Voice unavailable — try again.');
    }
  }

  // Download reply audio (server cache makes repeats fast)
  async function downloadSpeech(text) {
    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, lang: voiceLang })
      });
      if (!res.ok) throw new Error('tts failed');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'astra_speech.mp3';
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setHint('Download failed — try again.');
    }
  }

  // ---------- Voice: speech-to-text (server: Whisper) ----------
  let mediaRecorder = null;
  let mediaStream = null;
  let audioChunks = [];
  let recordTimer = null;
  let recordSeconds = 0;

  function isRecording() {
    return !!(mediaRecorder && mediaRecorder.state === 'recording');
  }

  function setMicState(state) {
    micBtn.classList.remove('listening', 'busy', 'denied');

    if (state === 'recording') {
      micBtn.classList.add('listening');
      micBtn.title = 'Recording… click to stop (max ' + MAX_RECORD_SECONDS + 's)';
      inputEl.placeholder = 'Recording… click mic to stop';
    } else if (state === 'busy') {
      micBtn.classList.add('busy');
      micBtn.title = 'Transcribing…';
      inputEl.placeholder = 'Transcribing your voice…';
    } else if (state === 'denied') {
      micBtn.classList.add('denied');
      micBtn.title = 'Microphone blocked';
      inputEl.placeholder = 'Microphone blocked — allow mic access in your browser, then retry';
    } else {
      micBtn.title = 'Speak your message';
      if (!hintActive) inputEl.placeholder = PLACEHOLDER;
    }
  }

  function startRecordTimer() {
    recordSeconds = 0;
    stopRecordTimer();
    recordTimer = setInterval(() => {
      recordSeconds += 1;
      if (isRecording()) {
        inputEl.placeholder = 'Recording… ' + recordSeconds + 's / ' + MAX_RECORD_SECONDS + 's — click mic to stop';
      }
      if (recordSeconds >= MAX_RECORD_SECONDS && isRecording()) {
        setHint('Max ' + MAX_RECORD_SECONDS + 's reached — transcribing…');
        try { mediaRecorder.stop(); } catch (e) { /* ignore */ }
      }
    }, 1000);
  }

  function stopRecordTimer() {
    if (recordTimer) { clearInterval(recordTimer); recordTimer = null; }
  }

  async function transcribeBlob(blob) {
    transcribing = true;
    setMicState('busy');

    const mime = (blob.type || 'audio/webm').split(';')[0];
    const ext = mime.includes('mp4') ? 'mp4' : mime.includes('ogg') ? 'ogg' : 'webm';
    const fd = new FormData();
    fd.append('file', blob, 'speech.' + ext);
    // explicit user language preference; omit → server/Whisper auto-detects
    if (voiceLang === 'en' || voiceLang === 'hi') fd.append('lang', voiceLang);

    try {
      const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
      const data = await res.json();

      if (data.ok && data.text) {
        // review-before-send: fill the editable input; user edits + presses Enter
        inputEl.value = data.text;
        inputEl.dispatchEvent(new Event('input'));
        inputEl.focus();
        hintActive = false;
        inputEl.placeholder = PLACEHOLDER;
        setHint('Voice transcribed — review, edit if needed, then press Enter.');
      } else if (res.status === 429) {
        setHint('Voice limit reached — try again shortly.');
      } else if (res.status === 422) {
        setHint(data.error || "Didn't catch any speech — try again.");
      } else if (res.status === 413) {
        setHint(data.error || 'Recording too long — try a shorter message.');
      } else if (res.status === 400) {
        setHint(data.error || 'Voice service not configured.');
      } else {
        setHint(data.error || "Couldn't hear clearly — try again.");
      }
    } catch (e) {
      setHint('Transcription failed — check connection.');
    } finally {
      transcribing = false;
      setMicState('idle');
    }
  }

  function stopListening() {
    stopRecordTimer();
    if (isRecording()) {
      try { mediaRecorder.stop(); } catch (e) { /* ignore */ }
    }
  }

  function initVoiceInput() {
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      micBtn.style.display = 'none';
      return;
    }

    micBtn.addEventListener('click', async () => {
      if (busy || transcribing) return;

      if (isRecording()) {
        try { mediaRecorder.stop(); } catch (e) { /* ignore */ }
        return;
      }

      stopSpeaking();

      // permission request state — clear feedback while the browser prompt is open
      micBtn.title = 'Requesting microphone permission…';
      inputEl.placeholder = 'Allow microphone access in your browser…';

      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (e) {
        setMicState('denied');
        setHint('Mic access denied — click the 🔒 icon in the address bar to allow, then retry. You can still type your message.');
        return;
      }

      audioChunks = [];

      try {
        mediaRecorder = new MediaRecorder(mediaStream);
      } catch (e) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
        setMicState('idle');
        setHint('Recording not supported in this browser — try Chrome or Edge.');
        return;
      }

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        stopRecordTimer();
        if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
        setMicState('idle');

        const type = (mediaRecorder.mimeType || 'audio/webm').split(';')[0];
        const blob = new Blob(audioChunks, { type });
        audioChunks = [];

        if (!blob.size || blob.size < 800) {
          setHint("Didn't catch that — try again (speak closer to the mic).");
          return;
        }
        transcribeBlob(blob);
      };

      mediaRecorder.start();
      startRecordTimer();
      setMicState('recording');
    });

    // never keep recording in the background
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) stopListening();
    });
    window.addEventListener('pagehide', () => stopListening());
  }

  // ---------- Context menu (right-click delete) ----------
  let ctxChatId = null;

  function hideCtxMenu() {
    ctxMenu.style.display = 'none';
    ctxChatId = null;
  }

  document.addEventListener('pointerdown', (e) => {
    if (!ctxMenu.contains(e.target)) hideCtxMenu();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') hideCtxMenu();
  });

  scrollArea.addEventListener('scroll', hideCtxMenu);
  chatList.addEventListener('scroll', hideCtxMenu);

  ctxDelete.addEventListener('click', () => {
    if (ctxChatId != null) deleteChat(ctxChatId);
    hideCtxMenu();
  });

  const ctxRename = document.getElementById('ctxRename');
  if (ctxRename) {
    ctxRename.addEventListener('click', () => {
      if (ctxChatId == null) return hideCtxMenu();
      const chat = chats.find(c => c.id === ctxChatId);
      hideCtxMenu();
      if (!chat) return;
      const title = window.prompt('Rename chat', chat.title);
      if (!title || !title.trim()) return;
      chat.title = title.trim().slice(0, MAX_TITLE * 2);
      save();
      renderSidebar();
      if (chat.convId) {
        api(`/api/conversations/${chat.convId}`, {
          method: 'PATCH',
          body: JSON.stringify({ title: chat.title })
        }).catch(() => {});
      }
    });
  }

  // ---------- Rendering ----------
  function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function isSafeUrl(u) {
    try {
      const url = new URL(u, window.location.origin);
      return url.protocol === 'http:' || url.protocol === 'https:';
    } catch (e) { return false; }
  }

  // Tiny syntax highlighter for escaped code (no external deps).
  // SINGLE-PASS tokenizer: the input is scanned exactly once, so markup
  // inserted for one token can never be re-matched by a later rule. (The old
  // sequential string/number/comment/keyword passes each re-scanned the
  // previous pass's <span class="..."> output, so `class` inside an attribute
  // got keyword-wrapped into `<span <span class="tok-kw">class</span>="tok-num">`
  // and the browser rendered those fragments as literal text.)
  const HL_KEYWORDS = /\b(def|return|if|elif|else|for|while|in|not|and|or|import|from|as|class|try|except|finally|raise|with|lambda|yield|pass|break|continue|global|None|True|False|self|function|const|let|var|new|this|typeof|instanceof|await|async|export|default|extends|super|public|private|static|void|interface|type|enum|struct|union|printf|include|using|namespace|template|fn|let|mut|pub|impl|match|func|package|var|nil|true|false|do|end|then|require|module|puts|print|echo|isset|foreach|endif|switch|case|break|default)\b/;
  const HL_STRINGS = /&quot;(?:[^&\\]|\\.)*?&quot;|&#39;(?:[^&\\]|\\.)*?&#39;|'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"/;
  const HL_HASH_LANGS = ['python', 'ruby', 'bash', 'shell', 'sh', 'yaml', 'yml', 'toml', 'perl', 'r', 'make', 'dockerfile', 'ps1', 'powershell'];

  function highlightCode(escaped, lang) {
    const l = (lang || '').toLowerCase();
    // python-like langs comment with '#'; other langs get '//' plus '#' and /* */
    const comment = HL_HASH_LANGS.indexOf(l) >= 0
      ? '#[^\\n]*'
      : '\\/\\*[\\s\\S]*?\\*\\/|\\/\\/[^\\n]*|#[^\\n]*';
    const re = new RegExp(
      HL_STRINGS.source + '|(' + comment + ')|' + HL_KEYWORDS.source + '|\\b(\\d+(?:\\.\\d+)?)\\b',
      'g'
    );
    return escaped.replace(re, (m, com, kw, num) => {
      if (com != null) return '<span class="tok-com">' + m + '</span>';
      if (kw != null) return '<span class="tok-kw">' + m + '</span>';
      if (num != null) return '<span class="tok-num">' + m + '</span>';
      return '<span class="tok-str">' + m + '</span>';
    });
  }

  // Safe markdown-lite: code fences are pulled out RAW first, then all
  // remaining text is HTML-escaped before any rule adds our own tags.
  // Supports: headings, bold/italic, inline code, fenced code (+highlight),
  // ordered/unordered/task lists, tables, blockquotes, safe [text](url) links.
  // Citations: [Dn] = document chunks (non-clickable chips), [n] = web sources (links).
  function formatAssistantHtml(text, sources, docSources) {
    const blocks = [];
    // Fences are extracted from the RAW text (before escaping) so:
    //  - data-code keeps the exact original source for the Copy button,
    //  - the placeholder tokens survive every later rule untouched.
    const makeBlock = (raw, lang) => {
      const clean = raw.replace(/\n$/, '');
      const l = (lang || '').toLowerCase() || 'text';
      blocks.push(
        '<pre class="code-block"><button class="block-copy" type="button" data-code="' +
        encodeURIComponent(clean) + '" title="Copy code">Copy</button>' +
        '<code class="lang-' + escapeHtml(l) + '">' + highlightCode(escapeHtml(clean), l) + '</code></pre>'
      );
      return '\u0000' + (blocks.length - 1) + '\u0000';
    };
    let t = String(text == null ? '' : text);
    // closed fence with a language (\r? tolerates CRLF sources)
    t = t.replace(/```([a-zA-Z0-9_+-]*)\r?\n([\s\S]*?)```/g, (m, lang, code) => makeBlock(code, lang));
    // legacy fence: no newline after the opening ```
    t = t.replace(/```([\s\S]*?)```/g, (m, code) => makeBlock(code, 'text'));
    // unterminated fence: opening ``` at line start runs to end of input
    t = t.replace(/(^|\n)```([a-zA-Z0-9_+-]*)([\s\S]*)$/, (m, br, lang, rest) => br + makeBlock(rest, lang));
    // now escape everything that is not already a code placeholder
    t = escapeHtml(t);

    // tables: lines starting with | — header row + separator + body rows
    t = t.replace(/((?:^|\n)(?:\|[^\n]*\|)(?:\n\|[^\n]*\|)+)/g, (m) => {
      const lines = m.trim().split('\n').map(l => l.trim()).filter(l => l.startsWith('|'));
      if (lines.length < 2) return m;
      const splitRow = (l) => l.slice(1, -1).split('|').map(c => c.trim());
      const isSep = (l) => /^[\s|:-]+$/.test(l) && l.includes('-');
      const header = splitRow(lines[0]);
      let bodyStart = 1;
      if (lines.length > 1 && isSep(lines[1])) bodyStart = 2;
      const rows = lines.slice(bodyStart).filter(l => !isSep(l)).map(splitRow);
      let html = '<table><thead><tr>' + header.map(h => '<th>' + inlineMd(h) + '</th>').join('') + '</tr></thead><tbody>';
      rows.forEach(r => { html += '<tr>' + r.map(c => '<td>' + inlineMd(c) + '</td>').join('') + '</tr>'; });
      html += '</tbody></table>';
      return '\n' + html + '\n';
    });

    // headings (line-level)
    t = t.replace(/(^|\n)#{1,4}\s+(.*)/g, (m, br, h) => br + '<h3>' + inlineMd(h) + '</h3>');
    // blockquotes
    t = t.replace(/(^|\n)&gt;\s?(.*)/g, (m, br, q) => br + '<blockquote>' + inlineMd(q) + '</blockquote>');
    // unordered list
    t = t.replace(/(^|\n)((?:[-*]\s+.*(?:\n|$))+)/g, (m, br, list) => {
      const items = list.trim().split('\n').map(l => l.replace(/^[-*]\s+/, ''));
      return br + '<ul>' + items.map(i => '<li>' + inlineMd(i) + '</li>').join('') + '</ul>';
    });
    // ordered list
    t = t.replace(/(^|\n)((?:\d+\.\s+.*(?:\n|$))+)/g, (m, br, list) => {
      const items = list.trim().split('\n').map(l => l.replace(/^\d+\.\s+/, ''));
      return br + '<ol>' + items.map(i => '<li>' + inlineMd(i) + '</li>').join('') + '</ol>';
    });

    // paragraphs around leftover text
    t = t.split(/\n{2,}/).map(p => {
      const trimmed = p.trim();
      if (!trimmed) return '';
      // split around code placeholders so a <pre> is never nested in a <p>
      if (trimmed.indexOf('\u0000') >= 0) {
        return trimmed.split(/(\u0000\d+\u0000)/).map(seg => {
          if (/^\u0000\d+\u0000$/.test(seg)) return seg;
          const s = seg.trim();
          if (!s) return '';
          if (/^<(?:table|ul|ol|h3|blockquote|pre|div)/.test(s)) return s;
          return '<p>' + s.replace(/\n/g, '<br>') + '</p>';
        }).join('');
      }
      // don't wrap block-level tags we just produced in <p>
      if (/^\s*<(?:table|ul|ol|h3|blockquote|pre|div)/.test(trimmed)) return trimmed;
      return '<p>' + trimmed.replace(/\n/g, '<br>') + '</p>';
    }).join('');

    // safe markdown links [text](url) — http(s) only
    t = t.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (m, label, url) => {
      if (!isSafeUrl(url)) return label;
      return '<a href="' + escapeHtml(url) + '" target="_blank" rel="noopener noreferrer">' + label + '</a>';
    });

    if (docSources && docSources.length) {
      t = t.replace(/\[D(\d{1,2})\]/g, (m, n) => {
        const src = docSources[parseInt(n, 10) - 1];
        if (!src) return m;
        const label = (src.filename || 'Document') + (src.page != null ? ' · page ' + src.page : '');
        return '<span class="cite doc" title="' + escapeHtml(label) + '">D' + n + '</span>';
      });
    }
    if (sources && sources.length) {
      t = t.replace(/\[(\d{1,2})\]/g, (m, n) => {
        const src = sources[parseInt(n, 10) - 1];
        if (!src || !isSafeUrl(src.url)) return m;
        return '<a class="cite" href="' + escapeHtml(src.url) +
          '" target="_blank" rel="noopener noreferrer">' + n + '</a>';
      });
    }

    // restore code blocks LAST — never wrapped, re-scanned, or nested in <p>
    t = t.replace(/\u0000(\d+)\u0000/g, (m, i) => blocks[+i]);
    return t;
  }

  // inline-only markdown (bold/italic/code) for use inside table cells etc.
  function inlineMd(s) {
    let x = s;
    x = x.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    x = x.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
    x = x.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    return x;
  }

  function buildDocSourcesEl(sources) {
    const details = document.createElement('details');
    details.className = 'sources doc-sources';
    const summary = document.createElement('summary');
    summary.textContent = 'Document sources (' + sources.length + ')';
    details.appendChild(summary);
    const ol = document.createElement('ol');
    sources.forEach(s => {
      const li = document.createElement('li');
      const name = document.createElement('span');
      name.className = 'src-name';
      name.textContent = s.filename || s.title || 'Document';
      li.appendChild(name);
      if (s.page != null) {
        const badge = document.createElement('span');
        badge.className = 'src-page';
        badge.textContent = 'p. ' + s.page;
        li.appendChild(badge);
      }
      if (s.section) {
        const sec = document.createElement('span');
        sec.className = 'src-date';
        sec.textContent = ' — ' + String(s.section).slice(0, 60);
        li.appendChild(sec);
      }
      if (s.snippet) {
        const snip = document.createElement('div');
        snip.className = 'src-snippet';
        snip.textContent = String(s.snippet).slice(0, 220);
        li.appendChild(snip);
      }
      ol.appendChild(li);
    });
    details.appendChild(ol);
    return details;
  }

  function buildSourcesEl(sources) {
    const details = document.createElement('details');
    details.className = 'sources';
    const summary = document.createElement('summary');
    summary.textContent = 'Sources (' + sources.length + ')';
    details.appendChild(summary);
    const ol = document.createElement('ol');
    sources.forEach(s => {
      const li = document.createElement('li');
      const label = String(s.title || s.url || 'Source').slice(0, 140);
      if (isSafeUrl(s.url)) {
        const a = document.createElement('a');
        a.href = s.url;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.textContent = label;
        li.appendChild(a);
      } else {
        li.textContent = label;
      }
      if (s.published) {
        const date = document.createElement('span');
        date.className = 'src-date';
        date.textContent = ' — ' + String(s.published).slice(0, 40);
        li.appendChild(date);
      }
      ol.appendChild(li);
    });
    details.appendChild(ol);
    return details;
  }

  function renderSidebar() {
    chatList.innerHTML = '';
    const visible = searchQuery
      ? chats.filter(c => (c.title || '').toLowerCase().includes(searchQuery))
      : chats;

    if (!visible.length) {
      const empty = document.createElement('div');
      empty.className = 'chat-empty';
      empty.innerHTML = '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>';
      empty.appendChild(document.createTextNode(searchQuery ? 'No chats match your search.' : 'No chats yet.'));
      chatList.appendChild(empty);
      return;
    }

    // group by date when not searching
    let lastGroup = '';
    visible.forEach(chat => {
      if (!searchQuery) {
        const group = chatDateGroup(chat);
        if (group !== lastGroup) {
          lastGroup = group;
          const gl = document.createElement('div');
          gl.className = 'chat-group-label';
          gl.textContent = group;
          chatList.appendChild(gl);
        }
      }

      const item = document.createElement('button');
      item.className = 'chat-item' + (chat.id === activeId ? ' active' : '');
      item.title = chat.title;

      const label = document.createElement('span');
      label.textContent = chat.title;
      item.appendChild(label);

      // more button (⋯) — opens context menu (works on touch too)
      const more = document.createElement('button');
      more.className = 'chat-more';
      more.innerHTML = '⋯';
      more.title = 'Chat options';
      more.setAttribute('aria-label', 'Chat options');
      more.addEventListener('click', (e) => {
        e.stopPropagation();
        ctxChatId = chat.id;
        ctxMenu.style.display = 'block';
        const rect = more.getBoundingClientRect();
        const x = Math.min(rect.right + 4, window.innerWidth - 170);
        const y = Math.min(rect.top, window.innerHeight - 56);
        ctxMenu.style.left = x + 'px';
        ctxMenu.style.top = y + 'px';
      });
      item.appendChild(more);

      item.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        ctxChatId = chat.id;
        ctxMenu.style.display = 'block';
        const x = Math.min(e.clientX, window.innerWidth - 170);
        const y = Math.min(e.clientY, window.innerHeight - 56);
        ctxMenu.style.left = x + 'px';
        ctxMenu.style.top = y + 'px';
      });

      item.addEventListener('click', async () => {
        if (busy) return;
        activeId = chat.id;
        save();
        renderSidebar();
        renderMessages();
        renderDocsStrip();
        closeSidebarMobile();
        // guest mode: lazy-load messages from the server on first open
        if (!chat.synced && chat.convId) {
          await loadChatMessages(chat);
          renderMessages();
        }
        loadDocsForChat(chat);
        inputEl.focus();
      });

      chatList.appendChild(item);
    });
  }

  // date grouping label for a chat
  function chatDateGroup(chat) {
    const d = chat.updatedAt ? new Date(chat.updatedAt) : (chat.createdAt ? new Date(chat.createdAt) : null);
    if (!d || isNaN(d)) return 'Earlier';
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today); yesterday.setDate(today.getDate() - 1);
    const weekAgo = new Date(today); weekAgo.setDate(today.getDate() - 7);
    if (d >= today) return 'Today';
    if (d >= yesterday) return 'Yesterday';
    if (d >= weekAgo) return 'Last 7 days';
    if (d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear()) return 'This month';
    return 'Earlier';
  }

  // loading skeleton for chat list (during sync)
  function showChatLoading() {
    chatList.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.className = 'chat-loading';
    for (let i = 0; i < 4; i++) {
      const skel = document.createElement('div');
      skel.className = 'chat-loading-skel';
      wrap.appendChild(skel);
    }
    chatList.appendChild(wrap);
  }

  function renderMessages() {
    stopSpeaking();
    const chat = activeChat();
    const msgs = chat ? chat.messages : [];
    const hasMsgs = msgs.length > 0;

    emptyState.style.display = hasMsgs ? 'none' : '';
    messagesEl.classList.toggle('show', hasMsgs);
    exportBtn.disabled = !hasMsgs;

    messagesEl.replaceChildren();

    msgs.forEach((m, i) => {
      if (m.role === 'user') {
        const wrap = document.createElement('div');
        wrap.className = 'msg user';
        const bubble = document.createElement('div');
        bubble.className = 'bubble-user';

        if (m.attachments && m.attachments.length) {
          const attWrap = document.createElement('div');
          attWrap.className = 'bubble-attachments';
          m.attachments.forEach(a => {
            if (a.kind === 'image' && a.thumb) {
              const img = document.createElement('img');
              img.src = a.thumb;
              img.alt = a.name || 'image';
              img.className = 'bubble-img';
              img.title = a.name || '';
              attWrap.appendChild(img);
            } else {
              const chip = document.createElement('span');
              chip.className = 'bubble-doc';
              chip.textContent = (a.kind === 'doc' ? '📄 ' : '📎 ') + (a.name || 'file');
              attWrap.appendChild(chip);
            }
          });
          bubble.appendChild(attWrap);
        }

        const textSpan = document.createElement('span');
        textSpan.textContent = m.content;
        bubble.appendChild(textSpan);
        wrap.appendChild(bubble);
        messagesEl.appendChild(wrap);
        return;
      }

      const wrap = document.createElement('div');
      wrap.className = 'msg assistant' + (m.error ? ' error' : '');

      const body = document.createElement('div');
      body.className = 'assistant-body';

      const name = document.createElement('div');
      name.className = 'assistant-name';
      name.textContent = m.error ? 'Astra AI · error' : 'Astra AI';

      const text = document.createElement('div');
      text.className = 'assistant-text' + (m.streaming ? ' streaming' : '');
      if (m.streaming) {
        text.textContent = m.content;
      } else {
        text.innerHTML = formatAssistantHtml(m.content, m.sources, m.docSources);
      }

      body.appendChild(name);
      body.appendChild(text);

      if (!m.streaming && m.searchNotice) {
        const notice = document.createElement('div');
        notice.className = 'search-notice';
        notice.textContent = '⚠ ' + m.searchNotice;
        body.appendChild(notice);
      }

      if (!m.streaming && m.docSources && m.docSources.length) {
        body.appendChild(buildDocSourcesEl(m.docSources));
      }

      if (!m.streaming && m.sources && m.sources.length) {
        body.appendChild(buildSourcesEl(m.sources));
      }

      const actions = document.createElement('div');
      actions.className = 'msg-actions';

      const copyBtn = document.createElement('button');
      copyBtn.className = 'mini-btn';
      copyBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(m.content).then(() => {
          copyBtn.innerHTML = '✓ Copied';
          setTimeout(() => {
            copyBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';
          }, 1400);
        }).catch(() => {});
      });
      actions.appendChild(copyBtn);

      const speakBtn = document.createElement('button');
      speakBtn.className = 'mini-btn';
      speakBtn.innerHTML = SPK_ICON + ' Listen';
      speakBtn.title = 'Read aloud (click again to pause/resume)';
      speakBtn.addEventListener('click', () => speakText(m.content, speakBtn));
      actions.appendChild(speakBtn);

      const dlBtn = document.createElement('button');
      dlBtn.className = 'mini-btn';
      dlBtn.innerHTML = DL_ICON + ' Save';
      dlBtn.title = 'Download speech as MP3';
      dlBtn.addEventListener('click', () => downloadSpeech(m.content));
      actions.appendChild(dlBtn);

      if (m.error && i === msgs.length - 1) {
        const retryBtn = document.createElement('button');
        retryBtn.className = 'mini-btn';
        retryBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6"/><path d="M20.5 15a9 9 0 1 1-2.1-9.4L23 10"/></svg> Retry';
        retryBtn.addEventListener('click', () => retryLast());
        actions.appendChild(retryBtn);
      } else if (!m.error && !m.streaming && i === msgs.length - 1 && msgs[i - 1] && msgs[i - 1].role === 'user') {
        const regenBtn = document.createElement('button');
        regenBtn.className = 'mini-btn';
        regenBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6"/><path d="M20.5 15a9 9 0 1 1-2.1-9.4L23 10"/></svg> Regenerate';
        regenBtn.title = 'Generate a fresh reply for the last message';
        regenBtn.addEventListener('click', () => regenerateLast());
        actions.appendChild(regenBtn);
      }

      body.appendChild(actions);
      wrap.appendChild(body);
      messagesEl.appendChild(wrap);
    });

    scrollBottom(false);
  }

  function showTyping() {
    const wrap = document.createElement('div');
    wrap.className = 'msg assistant';
    wrap.id = 'typingMsg';
    wrap.innerHTML =
      '<div class="assistant-body">' +
      '<div class="assistant-name">Astra AI</div>' +
      '<div class="typing"><i></i><i></i><i></i><span class="typing-label">thinking…</span></div>' +
      '</div>';
    messagesEl.appendChild(wrap);
    scrollBottom(true);
  }

  function hideTyping() {
    const el = document.getElementById('typingMsg');
    if (el) el.remove();
  }

  function setTypingStatus(label) {
    const el = document.getElementById('typingMsg');
    if (!el) return;
    const name = el.querySelector('.assistant-name');
    if (name) name.textContent = label;
    const lab = el.querySelector('.typing-label');
    if (lab) {
      // extract the state word after "· " for the small italic label
      const parts = label.split('·');
      lab.textContent = parts.length > 1 ? parts[parts.length - 1].trim() : 'thinking…';
    }
  }

  function scrollBottom(smooth) {
    requestAnimationFrame(() => {
      scrollArea.scrollTo({ top: scrollArea.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
    });
  }

  // ---------- Chat actions ----------
  function makeTitle(text) {
    const clean = text.trim().replace(/\s+/g, ' ');
    return clean.length > MAX_TITLE ? clean.slice(0, MAX_TITLE) + '…' : clean;
  }

  function startNewChat() {
    if (busy) return;
    // Only adopt a genuinely empty local chat. Server-synced chats are created
    // with messages: [] and synced:false (history lazy-loads on first click),
    // so `messages.length === 0` alone would switch into an old conversation
    // instead of starting a new one.
    const existing = chats.find(c => c.messages.length === 0 && c.synced !== false);
    if (existing) {
      activeId = existing.id;
    } else {
      const c = newChatObject();
      chats.unshift(c);
      activeId = c.id;
    }
    save();
    renderSidebar();
    renderMessages();
    renderDocsStrip();
    closeSidebarMobile();
    inputEl.focus();
  }

  function deleteChat(id) {
    if (busy) return;
    const gone = chats.find(c => c.id === id);
    if (!gone) return;
    if (!window.confirm('Delete "' + (gone.title || 'this chat') + '"? This cannot be undone.')) return;
    chats = chats.filter(c => c.id !== id);
    if (!chats.length) {
      const c = newChatObject();
      chats.push(c);
    }
    if (activeId === id) activeId = chats[0].id;
    save();
    renderSidebar();
    renderMessages();
    renderDocsStrip();
    // guest mode: also delete server-side (best effort)
    if (gone && gone.convId) {
      api(`/api/conversations/${gone.convId}`, { method: 'DELETE' }).catch(() => {});
    }
  }

  function exportChat() {
    const chat = activeChat();
    if (!chat || !chat.messages.length) return;

    const lines = chat.messages.map(m => {
      const who = m.role === 'user' ? 'You' : 'Astra AI';
      let att = '';
      if (m.attachments && m.attachments.length) {
        att = '\n[Attached: ' + m.attachments.map(a => a.name).join(', ') + ']';
      }
      return `${who}:\n${m.content}${att}\n`;
    });

    const blob = new Blob([`Astra AI — ${chat.title}\n\n${lines.join('\n')}`], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'astra_chat.txt';
    a.click();
    URL.revokeObjectURL(url);
  }

  // ---------- Sending ----------
  function setBusy(state) {
    busy = state;
    sendIcon.style.display = state ? 'none' : '';
    stopIcon.style.display = state ? '' : 'none';
    sendBtn.classList.toggle('stop', state);
    sendBtn.title = state ? 'Stop' : 'Send';
    updateSendEnabled();
  }

  function updateSendEnabled() {
    if (busy) { sendBtn.disabled = false; return; }
    const chat = activeChat();
    const hasDocs = !!(chat && chat.docs && chat.docs.some(d => d.doc_id));
    sendBtn.disabled = inputEl.value.trim().length === 0 && pendingFiles.length === 0 && !hasDocs;
  }

  async function send(text) {
    if (busy) return;
    if (text) text = text.trim();

    stopSpeaking();
    stopListening();

    if (pendingFiles.some(f => f.loading)) {
      setHint('Wait for files to finish loading.');
      return;
    }

    ensureChat();
    const chat = activeChat();

    const docs = (chat.docs || []).filter(d => d.doc_id);
    const files = pendingFiles.filter(f => !f.loading);
    const images = files.filter(f => f.kind === 'image' && f.data).slice(0, MAX_IMAGES);

    if (!text && !images.length && !docs.length) return;

    if (!text) {
      text = images.length
        ? 'What is in this image?'
        : 'Please answer using the attached documents.';
    }

    const attachMeta = [
      ...images.map(f => ({ kind: 'image', name: f.name, thumb: f.thumb })),
      ...docs.map(d => ({ kind: 'doc', name: d.name })),
    ];

    if (chat.messages.length === 0) {
      chat.title = makeTitle(text);
      chats = [chat, ...chats.filter(c => c.id !== chat.id)];
    }

    chat.messages.push({ role: 'user', content: text, attachments: attachMeta });
    inputEl.value = '';
    inputEl.style.height = 'auto';
    pendingFiles = [];
    renderChips();
    updateSendEnabled();

    save();
    renderSidebar();
    renderMessages();

    setBusy(true);
    showTyping();

    controller = new AbortController();

    let assistantMsg = null;
    let liveTextEl = null;
    let sawDone = false;
    let streamError = null;
    let pendingSources = null;
    let searchNotice = null;
    let pendingDocSources = null;

    const handleStreamEvent = (ev, data) => {
      if (ev === 'docs') {
        if (data.status === 'ok') {
          pendingDocSources = Array.isArray(data.sources) ? data.sources : null;
        }
      } else if (ev === 'meta') {
        setTypingStatus('Astra AI…');
      } else if (ev === 'search') {
        if (data.status === 'started') {
          setTypingStatus('Astra AI · searching the web…');
        } else if (data.status === 'ok') {
          setTypingStatus('Astra AI');
          pendingSources = Array.isArray(data.sources) ? data.sources : null;
        } else if (data.status === 'failed') {
          setTypingStatus('Astra AI');
          searchNotice = data.error || 'Live search unavailable.';
        }
      } else if (ev === 'delta' && data && data.text) {
        if (!assistantMsg) {
          hideTyping();
          assistantMsg = { role: 'assistant', content: '', error: false, streaming: true };
          if (pendingDocSources && pendingDocSources.length) assistantMsg.docSources = pendingDocSources;
          if (pendingSources && pendingSources.length) assistantMsg.sources = pendingSources;
          if (searchNotice) assistantMsg.searchNotice = searchNotice;
          chat.messages.push(assistantMsg);
          renderMessages();
          const nodes = messagesEl.querySelectorAll('.msg.assistant .assistant-text');
          liveTextEl = nodes[nodes.length - 1] || null;
        }
        assistantMsg.content += data.text;
        if (liveTextEl) liveTextEl.textContent = assistantMsg.content;
        scrollBottom(false);
      } else if (ev === 'done') {
        sawDone = true;
      } else if (ev === 'memory') {
        if (data && Array.isArray(data.suggestions) && data.suggestions.length) {
          showMemToast(data.suggestions.length);
        }
      } else if (ev === 'error') {
        streamError = data;
      }
    };

    try {
      const payload = {
        messages: chat.messages
          .slice(-24)
          .map(m => {
            let c = m.content.slice(0, 6000);
            if (m.role === 'user') {
              const userDocs = (m.attachments || []).filter(a => a.kind === 'doc');
              if (userDocs.length) {
                c += '\n[User previously attached: ' + userDocs.map(d => d.name).join(', ') + ']';
              }
            }
            return { role: m.role, content: c };
          }),
        conversation_id: chat.convId
      };
      if (docs.length) payload.doc_ids = docs.map(d => d.doc_id);
      const modelMode = getModelMode();
      if (modelMode && modelMode !== 'auto') payload.model = modelMode;
      payload.web_search = getWebSearchMode();
      if (replyLang && replyLang !== 'auto') payload.reply_lang = replyLang;
      // coding vs chat is auto-detected server-side from the message content
      if (images.length) {
        payload.images = images.map(im => ({ mime: im.mime, data: im.data.split(',', 2)[1] || im.data }));
      }

      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Device-Id': getDeviceId() },
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      const ct = res.headers.get('content-type') || '';
      if (!res.ok || !ct.includes('text/event-stream')) {
        // validation error (JSON 400) or server hiccup before the stream started
        let reply = 'Server error. Please try again.';
        if (ct.includes('application/json')) {
          try { const j = await res.json(); if (j.reply) reply = j.reply; } catch (e) { /* keep default */ }
        }
        hideTyping();
        chat.messages.push({ role: 'assistant', content: reply, error: true });
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf('\n\n')) !== -1) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          let evName = 'message';
          let dataStr = '';
          for (const line of frame.split('\n')) {
            if (line.startsWith('event:')) evName = line.slice(6).trim();
            else if (line.startsWith('data:')) dataStr += line.slice(5).trim();
          }
          if (!dataStr) continue;
          let data;
          try { data = JSON.parse(dataStr); } catch (e) { continue; }
          handleStreamEvent(evName, data);
        }
      }

      hideTyping();
      if (assistantMsg) {
        assistantMsg.streaming = false;
        if (streamError) {
          // preserve any text that already arrived
          assistantMsg.error = true;
          if (!assistantMsg.content) {
            assistantMsg.content = streamError.reply || 'Something went wrong. Please try again.';
          }
        } else if (!sawDone) {
          assistantMsg.error = true;
          if (!assistantMsg.content) {
            assistantMsg.content = 'No response received.';
          }
        }
      } else {
        const fallback = {
          role: 'assistant',
          content: streamError
            ? (streamError.reply || 'Something went wrong. Please try again.')
            : 'No response received.',
          error: true
        };
        if (pendingDocSources && pendingDocSources.length) fallback.docSources = pendingDocSources;
        if (pendingSources && pendingSources.length) fallback.sources = pendingSources;
        if (searchNotice) fallback.searchNotice = searchNotice;
        chat.messages.push(fallback);
      }
    } catch (err) {
      hideTyping();
      if (err.name === 'AbortError') {
        if (assistantMsg) {
          // keep whatever streamed before Stop was pressed
          assistantMsg.streaming = false;
        } else {
          chat.messages.push({
            role: 'assistant',
            content: '⏹ Generation stopped.',
            error: false
          });
        }
      } else {
        if (assistantMsg && assistantMsg.content) {
          assistantMsg.streaming = false;
          assistantMsg.error = true;
        } else {
          chat.messages.push({
            role: 'assistant',
            content: 'Network error. Please check your connection and try again.',
            error: true
          });
        }
      }
    } finally {
      controller = null;
      pendingSources = null;
      searchNotice = null;
      pendingDocSources = null;
      setBusy(false);
      save();
      renderSidebar();
      renderMessages();
      inputEl.focus();
      // server persisted this turn — refresh list (titles/order)
      syncConversations().catch(() => {});

      // auto-read new reply (only if enabled + clean final answer)
      if (autoSpeak) {
        const last = chat.messages[chat.messages.length - 1];
        if (last && last.role === 'assistant' && !last.error && !last.streaming
            && last.content && last.content.trim().length > 2
            && !last.content.startsWith('⏹')) {
          speakText(last.content, null);
        }
      }
    }
  }

  function retryLast() {
    if (busy) return;
    const chat = activeChat();
    if (!chat) return;

    while (chat.messages.length && chat.messages[chat.messages.length - 1].error) {
      chat.messages.pop();
    }

    let idx = -1;
    for (let i = chat.messages.length - 1; i >= 0; i--) {
      if (chat.messages[i].role === 'user') { idx = i; break; }
    }
    if (idx === -1) return;

    const lastUser = chat.messages[idx].content;
    chat.messages.splice(idx);
    save();
    renderMessages();
    send(lastUser);
  }

  // Regenerate: drop the last assistant reply and ask again (same user message)
  function regenerateLast() {
    if (busy) return;
    const chat = activeChat();
    if (!chat || !chat.messages.length) return;
    const last = chat.messages[chat.messages.length - 1];
    if (last.role !== 'assistant' || last.streaming) return;
    let idx = -1;
    for (let i = chat.messages.length - 2; i >= 0; i--) {
      if (chat.messages[i].role === 'user') { idx = i; break; }
    }
    if (idx === -1) return;
    const lastUser = chat.messages[idx].content;
    chat.messages.splice(idx);
    save();
    renderMessages();
    send(lastUser);
  }

  function stopGeneration() {
    if (controller) controller.abort();
  }

  // ---------- Sidebar (mobile) ----------
  function openSidebarMobile() {
    sidebar.classList.add('open');
    overlay.classList.add('show');
  }

  function closeSidebarMobile() {
    sidebar.classList.remove('open');
    overlay.classList.remove('show');
  }

  // ---------- Events ----------
  menuBtn.addEventListener('click', () => {
    if (window.innerWidth <= 860) {
      if (sidebar.classList.contains('open')) closeSidebarMobile();
      else openSidebarMobile();
    } else {
      sidebar.classList.toggle('hidden');
    }
  });

  overlay.addEventListener('click', closeSidebarMobile);
  newChatBtn.addEventListener('click', startNewChat);
  exportBtn.addEventListener('click', exportChat);

  window.addEventListener('resize', () => {
    if (window.innerWidth > 860) {
      sidebar.classList.remove('open');
      overlay.classList.remove('show');
    } else {
      sidebar.classList.remove('hidden');
    }
  });

  sendBtn.addEventListener('click', () => {
    if (busy) stopGeneration();
    else send(inputEl.value);
  });

  const ACCEPT = {
    all: 'image/png,image/jpeg,image/webp,image/gif,.pdf,.docx,.txt,.md,.csv,.json,.log,.py,.js,.html,.css,.xml,.yml,.yaml',
    image: 'image/png,image/jpeg,image/webp,image/gif',
    doc: '.pdf,.docx,.txt,.md,.csv,.json,.log,.py,.js,.html,.css,.xml,.yml,.yaml'
  };

  function openAttachMenu() {
    attachMenu.style.display = 'block';
    attachBtn.setAttribute('aria-expanded', 'true');
  }

  function closeAttachMenu() {
    attachMenu.style.display = 'none';
    attachBtn.setAttribute('aria-expanded', 'false');
  }

  attachBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (busy) return;
    if (attachMenu.style.display === 'block') closeAttachMenu();
    else openAttachMenu();
  });

  attachMenu.addEventListener('click', (e) => {
    const item = e.target.closest('.attach-menu-item');
    if (!item) return;
    closeAttachMenu();
    fileInput.accept = ACCEPT[item.dataset.mode] || ACCEPT.all;
    fileInput.click();
  });

  document.addEventListener('click', (e) => {
    if (attachMenu.style.display === 'block' && !attachMenu.contains(e.target) && e.target !== attachBtn) {
      closeAttachMenu();
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length) addFiles(fileInput.files);
  });

  inputEl.addEventListener('paste', (e) => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    const files = [];
    for (const item of items) {
      if (item.kind === 'file') {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }
    if (files.length) {
      e.preventDefault();
      addFiles(files);
    }
  });

  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (busy) return;
      send(inputEl.value);
    }
  });

  inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + 'px';
    updateSendEnabled();
  });

  document.querySelectorAll('.sugg').forEach(btn => {
    btn.addEventListener('click', () => send(btn.dataset.prompt));
  });

  // ---------- Phase 5: guest device sync (no login) ----------
  const accountBtn = document.getElementById('accountBtn');
  const accountBtnLabel = document.getElementById('accountBtnLabel');
  const chatSearch = document.getElementById('chatSearch');
  const memModal = document.getElementById('memModal');
  const memClose = document.getElementById('memClose');
  const memToggle = document.getElementById('memToggle');
  const memPendingSection = document.getElementById('memPendingSection');
  const memPendingList = document.getElementById('memPendingList');
  const memActiveList = document.getElementById('memActiveList');
  const memNewInput = document.getElementById('memNewInput');
  const memAddBtn = document.getElementById('memAddBtn');
  const memUserEmail = document.getElementById('memUserEmail');
  const memAccountError = document.getElementById('memAccountError');
  const memToast = document.getElementById('memToast');
  const memToastText = document.getElementById('memToastText');
  const memToastReview = document.getElementById('memToastReview');
  const memToastClose = document.getElementById('memToastClose');

  let searchQuery = '';

  // Guest identity: one generated id per browser (localStorage). Sent as
  // X-Device-Id on every API call — the server scopes chats/memories to it.
  const DEVICE_ID_KEY = 'astra_device_id';
  function getDeviceId() {
    let id = null;
    try { id = localStorage.getItem(DEVICE_ID_KEY); } catch (e) { /* storage blocked */ }
    if (!id || !/^[A-Za-z0-9_-]{8,64}$/.test(id)) {
      id = (window.crypto && crypto.randomUUID)
        ? crypto.randomUUID()
        : 'd' + Date.now().toString(36) + Math.random().toString(36).slice(2, 14);
      try { localStorage.setItem(DEVICE_ID_KEY, id); } catch (e) { /* storage blocked */ }
    }
    return id;
  }

  async function api(path, opts = {}) {
    const init = { credentials: 'same-origin', ...opts };
    init.headers = {
      'X-Device-Id': getDeviceId(),
      ...(opts.body ? { 'Content-Type': 'application/json' } : {}),
      ...(opts.headers || {})
    };
    const res = await fetch(path, init);
    let data = null;
    try { data = await res.json(); } catch (e) { /* non-JSON */ }
    return { res, data };
  }

  function renderAccount() {
    accountBtnLabel.textContent = 'Guest';
    if (memUserEmail) memUserEmail.textContent = 'Guest · this device';
  }

  // Upsert server conversations into the local list (this device).
  async function syncConversations() {
    showChatLoading();
    try {
      const { res, data } = await api('/api/conversations');
      if (!res.ok || !data || !Array.isArray(data.conversations)) return;
      for (const sc of data.conversations) {
        let chat = chats.find(c => c.convId === sc.id);
        if (!chat) {
          chat = { id: nextId++, title: sc.title, messages: [], convId: sc.id, docs: [], synced: false, updatedAt: sc.updated_at || sc.created_at, createdAt: sc.created_at };
          chats.push(chat);
        } else {
          chat.title = sc.title;
          chat.updatedAt = sc.updated_at || sc.created_at || chat.updatedAt;
          chat.createdAt = sc.created_at || chat.createdAt;
        }
      }
      if (chats.length === 0) {
        const c = newChatObject();
        chats.push(c);
      }
      if (!chats.find(c => c.id === activeId)) activeId = chats[0].id;
      save();
    } finally {
      // Always restore the list: showChatLoading() wiped it, and if fetch
      // rejects (offline/DNS/reset) the callers swallow the rejection with
      // .catch(() => {}) — without this the sidebar stays on skeletons forever.
      renderSidebar();
    }
  }

  async function loadChatMessages(chat) {
    const { res, data } = await api(`/api/conversations/${chat.convId}/messages`);
    // Only mark synced on success — flagging a failed load would prevent any
    // retry this session and leave the chat permanently empty.
    if (res.ok && data && Array.isArray(data.messages)) {
      chat.messages = data.messages.map(m => ({ role: m.role, content: m.content }));
      chat.synced = true;
      save();
    }
  }

  function showMemToast(count) {
    if (!memToast) return;
    memToastText.textContent = count === 1
      ? 'Astra found 1 thing it could remember.'
      : `Astra found ${count} things it could remember.`;
    memToast.style.display = 'flex';
  }

  function hideMemToast() { if (memToast) memToast.style.display = 'none'; }

  // ----- memory modal -----
  async function refreshMemories() {
    const { res, data } = await api('/api/memories');
    if (!res.ok || !data) return;
    const all = data.memories || [];
    const pending = all.filter(m => m.status === 'pending');
    const active = all.filter(m => m.status === 'active');

    memPendingSection.style.display = pending.length ? '' : 'none';
    memPendingList.replaceChildren();
    pending.forEach(m => memPendingList.appendChild(memItem(m, true)));
    memActiveList.replaceChildren();
    active.forEach(m => memActiveList.appendChild(memItem(m, false)));
  }

  function memItem(m, isPending) {
    const li = document.createElement('li');
    li.className = 'mem-item' + (isPending ? ' pending' : '');

    const tag = document.createElement('span');
    tag.className = 'mem-tag';
    tag.textContent = m.kind || (isPending ? 'suggest' : 'fact');
    li.appendChild(tag);

    const text = document.createElement('span');
    text.className = 'mem-text';
    text.textContent = m.content;
    li.appendChild(text);

    if (isPending) {
      const ok = document.createElement('button');
      ok.className = 'btn-small';
      ok.textContent = 'Approve';
      ok.addEventListener('click', async () => {
        await api(`/api/memories/${m.id}/approve`, { method: 'POST', body: '{}' });
        refreshMemories();
      });
      const no = document.createElement('button');
      no.className = 'btn-small danger';
      no.textContent = 'Discard';
      no.addEventListener('click', async () => {
        await api(`/api/memories/${m.id}/reject`, { method: 'POST', body: '{}' });
        refreshMemories();
      });
      li.appendChild(ok);
      li.appendChild(no);
    } else {
      const edit = document.createElement('button');
      edit.className = 'btn-small';
      edit.textContent = 'Edit';
      edit.addEventListener('click', async () => {
        const content = window.prompt('Edit memory', m.content);
        if (!content || !content.trim() || content.trim() === m.content) return;
        await api(`/api/memories/${m.id}`, {
          method: 'PATCH',
          body: JSON.stringify({ content: content.trim() })
        });
        refreshMemories();
      });
      const del = document.createElement('button');
      del.className = 'btn-small danger';
      del.textContent = 'Delete';
      del.addEventListener('click', async () => {
        await api(`/api/memories/${m.id}`, { method: 'DELETE' });
        refreshMemories();
      });
      li.appendChild(edit);
      li.appendChild(del);
    }
    return li;
  }

  async function refreshMemoryPref() {
    const { res, data } = await api('/api/memories/preferences');
    if (res.ok && data) memToggle.checked = !!data.memory_enabled;
  }

  function openMemModal() {
    memAccountError.style.display = 'none';
    memModal.style.display = 'flex';
    refreshMemoryPref();
    refreshMemories();
  }

  function closeMemModal() { memModal.style.display = 'none'; }

  // ----- events: memory & data -----
  accountBtn.addEventListener('click', openMemModal);

  memClose.addEventListener('click', closeMemModal);
  memModal.addEventListener('click', (e) => { if (e.target === memModal) closeMemModal(); });

  memToggle.addEventListener('change', async () => {
    const { res } = await api('/api/memories/preferences', {
      method: 'PATCH',
      body: JSON.stringify({ memory_enabled: memToggle.checked })
    });
    await refreshMemoryPref();
    showSettingsToast(res.ok ? (memToggle.checked ? 'Memory enabled' : 'Memory disabled') : 'Could not update memory', res.ok);
  });

  memAddBtn.addEventListener('click', async () => {
    const content = memNewInput.value.trim();
    if (content.length < 2) return;
    const { res } = await api('/api/memories', { method: 'POST', body: JSON.stringify({ content }) });
    memNewInput.value = '';
    refreshMemories();
    showSettingsToast(res.ok ? 'Memory added' : 'Could not add memory', res.ok);
  });

  memNewInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); memAddBtn.click(); }
  });

  document.getElementById('clearChatsBtn').addEventListener('click', async () => {
    if (!window.confirm('Delete ALL chats on this device? This cannot be undone.')) return;
    const { res } = await api('/api/conversations?confirm=true', { method: 'DELETE' });
    if (res.ok) {
      chats = chats.filter(c => !c.synced && !c.convId);
      if (!chats.length) chats.push(newChatObject());
      activeId = chats[0].id;
      save();
      renderSidebar();
      renderMessages();
      renderDocsStrip();
      closeMemModal();
      showSettingsToast('All chats deleted');
    } else {
      showSettingsToast('Could not delete chats', false);
    }
  });

  document.getElementById('clearMemoriesBtn').addEventListener('click', async () => {
    if (!window.confirm('Delete ALL saved memories?')) return;
    const { res } = await api('/api/memories?confirm=true', { method: 'DELETE' });
    if (res.ok) {
      refreshMemories();
      showSettingsToast('All memories deleted');
    } else {
      showSettingsToast('Could not delete memories', false);
    }
  });

  memToastReview.addEventListener('click', () => { hideMemToast(); openMemModal(); });
  memToastClose.addEventListener('click', hideMemToast);

  // ----- search -----
  if (chatSearch) {
    chatSearch.style.display = '';
    chatSearch.addEventListener('input', () => {
      searchQuery = chatSearch.value.trim().toLowerCase();
      renderSidebar();
    });
  }

  // per-code-block copy button (delegated)
  if (messagesEl) {
    messagesEl.addEventListener('click', (e) => {
      const btn = e.target.closest('.block-copy');
      if (!btn) return;
      const code = decodeURIComponent(btn.getAttribute('data-code') || '');
      navigator.clipboard.writeText(code).then(() => {
        btn.textContent = '✓';
        setTimeout(() => { btn.textContent = 'Copy'; }, 1200);
      }).catch(() => {});
    });
  }

  // ---------- Init ----------
  initVoiceInput();
  loadVoiceSettings();
  loadSettings();
  loadModelMode();
  loadWebSearchMode();
  loadFeatureAvailability();

  if (!load()) {
    const c = newChatObject();
    chats.push(c);
    activeId = c.id;
  }

  ensureChat();
  renderSidebar();
  renderMessages();
  renderChips();
  renderDocsStrip();
  loadDocsForChat(activeChat());
  updateSendEnabled();
  inputEl.focus();
  renderAccount();
  syncConversations().catch(() => { /* offline — local chats still work */ });
})();
