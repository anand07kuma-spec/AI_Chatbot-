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
  const langBtn = document.getElementById('langBtn');
  const ctxMenu = document.getElementById('ctxMenu');
  const ctxDelete = document.getElementById('ctxDelete');

  const STORE_KEY = 'astra_chats_v1';
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

  // ---------- Persistence ----------
  function save() {
    try {
      sessionStorage.setItem(STORE_KEY, JSON.stringify({ chats, activeId, nextId }));
    } catch (e) { /* storage unavailable */ }
  }

  function load() {
    try {
      const raw = sessionStorage.getItem(STORE_KEY);
      if (!raw) return false;
      const data = JSON.parse(raw);
      if (!Array.isArray(data.chats) || !data.chats.length) return false;
      chats = data.chats;
      activeId = data.activeId;
      nextId = data.nextId || chats.length + 1;
      if (!chats.find(c => c.id === activeId)) activeId = chats[0].id;
      return true;
    } catch (e) {
      return false;
    }
  }

  function newChatObject() {
    return { id: nextId++, title: 'New chat', messages: [] };
  }

  function activeChat() {
    return chats.find(c => c.id === activeId);
  }

  function ensureChat() {
    if (!chats.length) {
      const c = newChatObject();
      chats.push(c);
      activeId = c.id;
    }
    if (!activeChat()) activeId = chats[0].id;
  }

  // ---------- Voice: text-to-speech (server: Edge Neural voices) ----------
  const ttsAudio = new Audio();

  const SPK_ICON = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>';
  const STOP_SMALL_ICON = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';

  let hintActive = false;

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

  function resetSpeakBtn(btn) {
    if (!btn) return;
    btn.innerHTML = SPK_ICON + ' Listen';
    btn.classList.remove('speaking', 'loading');
    if (speakingBtn === btn) speakingBtn = null;
  }

  function stopSpeaking() {
    try {
      ttsAudio.pause();
      ttsAudio.removeAttribute('src');
    } catch (e) { /* ignore */ }
    if (speakingBtn) resetSpeakBtn(speakingBtn);
    speakingBtn = null;
  }

  async function speakText(text, btn) {
    if (speakingBtn === btn) {
      stopSpeaking();
      return;
    }

    stopSpeaking();
    speakingBtn = btn;
    btn.classList.add('loading');
    btn.innerHTML = SPK_ICON + ' Loading…';

    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (!res.ok) throw new Error('tts failed');

      const blob = await res.blob();
      if (!blob.size) throw new Error('empty audio');

      const url = URL.createObjectURL(blob);

      btn.classList.remove('loading');
      btn.classList.add('speaking');
      btn.innerHTML = STOP_SMALL_ICON + ' Stop';

      ttsAudio.src = url;
      ttsAudio.onended = () => resetSpeakBtn(btn);
      ttsAudio.onerror = () => resetSpeakBtn(btn);
      await ttsAudio.play();
    } catch (e) {
      resetSpeakBtn(btn);
      setHint('Speech failed — try again.');
    }
  }

  // ---------- Voice: speech-to-text (server: Whisper) ----------
  let mediaRecorder = null;
  let mediaStream = null;
  let audioChunks = [];

  function isRecording() {
    return !!(mediaRecorder && mediaRecorder.state === 'recording');
  }

  function setMicState(state) {
    micBtn.classList.remove('listening', 'busy');

    if (state === 'recording') {
      micBtn.classList.add('listening');
      micBtn.title = 'Recording… click to stop';
      inputEl.placeholder = 'Recording (' + langEntry().name + ')… click mic to stop';
    } else if (state === 'busy') {
      micBtn.classList.add('busy');
      micBtn.title = 'Transcribing…';
      inputEl.placeholder = 'Transcribing your voice…';
    } else {
      micBtn.title = 'Speak your message';
      if (!hintActive) inputEl.placeholder = PLACEHOLDER;
    }
  }

  async function transcribeBlob(blob) {
    transcribing = true;
    setMicState('busy');

    const fd = new FormData();
    fd.append('file', blob, 'speech.webm');
    fd.append('lang', sttLang);

    try {
      const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
      const data = await res.json();

      if (data.ok && data.text) {
        inputEl.value = data.text;
        inputEl.dispatchEvent(new Event('input'));
        inputEl.focus();
        hintActive = false;
        inputEl.placeholder = PLACEHOLDER;
      } else if (res.status === 429) {
        setHint('Voice limit reached — try again shortly.');
      } else if (res.status === 400) {
        setHint('Voice service not configured.');
      } else {
        setHint("Couldn't hear clearly — try again.");
      }
    } catch (e) {
      setHint('Transcription failed — check connection.');
    } finally {
      transcribing = false;
      setMicState('idle');
    }
  }

  function stopListening() {
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

      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (e) {
        setHint('Microphone blocked — allow mic access in browser.');
        return;
      }

      audioChunks = [];

      try {
        mediaRecorder = new MediaRecorder(mediaStream);
      } catch (e) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
        setHint('Recording not supported in this browser.');
        return;
      }

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
        setMicState('idle');

        const type = (mediaRecorder.mimeType || 'audio/webm').split(';')[0];
        const blob = new Blob(audioChunks, { type });
        audioChunks = [];

        if (!blob.size || blob.size < 800) {
          setHint("Didn't catch that — try again.");
          return;
        }
        transcribeBlob(blob);
      };

      mediaRecorder.start();
      setMicState('recording');
    });
  }

  // ---------- Context menu (right-click delete) ----------
  let ctxChatId = null;

  function hideCtxMenu() {
    ctxMenu.style.display = 'none';
    ctxChatId = null;
  }

  document.addEventListener('click', (e) => {
    if (!ctxMenu.contains(e.target)) hideCtxMenu();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') hideCtxMenu();
  });

  scrollArea.addEventListener('scroll', hideCtxMenu);

  ctxDelete.addEventListener('click', () => {
    if (ctxChatId != null) deleteChat(ctxChatId);
    hideCtxMenu();
  });

  // ---------- Mic language (STT) ----------
  const LANGS = [
    { code: 'en', label: 'EN', name: 'English / Hinglish' },
    { code: 'hi', label: 'हिं', name: 'Hindi' },
    { code: 'ur', label: 'اردو', name: 'Urdu' }
  ];
  let sttLang = 'en';
  try {
    const saved = localStorage.getItem('astra_stt_lang');
    if (LANGS.some(l => l.code === saved)) sttLang = saved;
  } catch (e) { /* ignore */ }

  function langEntry() {
    return LANGS.find(l => l.code === sttLang) || LANGS[0];
  }

  function renderLang() {
    const entry = langEntry();
    langBtn.textContent = entry.label;
    langBtn.title = 'Mic language: ' + entry.name + ' (click to change)';
  }

  langBtn.addEventListener('click', () => {
    const i = LANGS.findIndex(l => l.code === sttLang);
    sttLang = LANGS[(i + 1) % LANGS.length].code;
    try { localStorage.setItem('astra_stt_lang', sttLang); } catch (e) { /* ignore */ }
    renderLang();
  });

  renderLang();

  // ---------- Rendering ----------
  function renderSidebar() {
    chatList.innerHTML = '';
    chats.forEach(chat => {
      const item = document.createElement('button');
      item.className = 'chat-item' + (chat.id === activeId ? ' active' : '');
      item.title = chat.title;

      const label = document.createElement('span');
      label.textContent = chat.title;
      item.appendChild(label);

      item.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        ctxChatId = chat.id;
        ctxMenu.style.display = 'block';
        const x = Math.min(e.clientX, window.innerWidth - 170);
        const y = Math.min(e.clientY, window.innerHeight - 56);
        ctxMenu.style.left = x + 'px';
        ctxMenu.style.top = y + 'px';
      });

      item.addEventListener('click', () => {
        if (busy) return;
        activeId = chat.id;
        save();
        renderSidebar();
        renderMessages();
        closeSidebarMobile();
      });

      chatList.appendChild(item);
    });
  }

  function clearNode(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function renderMessages() {
    stopSpeaking();
    const chat = activeChat();
    const msgs = chat ? chat.messages : [];
    const hasMsgs = msgs.length > 0;

    emptyState.style.display = hasMsgs ? 'none' : '';
    messagesEl.classList.toggle('show', hasMsgs);
    exportBtn.disabled = !hasMsgs;

    clearNode(messagesEl);

    msgs.forEach((m, i) => {
      if (m.role === 'user') {
        const wrap = document.createElement('div');
        wrap.className = 'msg user';
        const bubble = document.createElement('div');
        bubble.className = 'bubble-user';
        bubble.textContent = m.content;
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
      text.className = 'assistant-text';
      text.textContent = m.content;

      body.appendChild(name);
      body.appendChild(text);

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
      speakBtn.title = 'Read aloud';
      speakBtn.addEventListener('click', () => speakText(m.content, speakBtn));
      actions.appendChild(speakBtn);

      if (m.error && i === msgs.length - 1) {
        const retryBtn = document.createElement('button');
        retryBtn.className = 'mini-btn';
        retryBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6"/><path d="M20.5 15a9 9 0 1 1-2.1-9.4L23 10"/></svg> Retry';
        retryBtn.addEventListener('click', () => retryLast());
        actions.appendChild(retryBtn);
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
      '<div class="typing"><i></i><i></i><i></i></div>' +
      '</div>';
    messagesEl.appendChild(wrap);
    scrollBottom(true);
  }

  function hideTyping() {
    const el = document.getElementById('typingMsg');
    if (el) el.remove();
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
    const existing = chats.find(c => c.messages.length === 0);
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
    closeSidebarMobile();
    inputEl.focus();
  }

  function deleteChat(id) {
    if (busy) return;
    chats = chats.filter(c => c.id !== id);
    if (!chats.length) {
      const c = newChatObject();
      chats.push(c);
    }
    if (activeId === id) activeId = chats[0].id;
    save();
    renderSidebar();
    renderMessages();
  }

  function exportChat() {
    const chat = activeChat();
    if (!chat || !chat.messages.length) return;

    const lines = chat.messages.map(m => {
      const who = m.role === 'user' ? 'You' : 'Astra AI';
      return `${who}:\n${m.content}\n`;
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
    sendBtn.disabled = busy ? false : inputEl.value.trim().length === 0;
  }

  async function send(text) {
    if (busy) return;
    text = text.trim();
    if (!text) return;

    stopSpeaking();
    stopListening();

    ensureChat();
    const chat = activeChat();

    if (chat.messages.length === 0) {
      chat.title = makeTitle(text);
      chats = [chat, ...chats.filter(c => c.id !== chat.id)];
    }

    chat.messages.push({ role: 'user', content: text });
    inputEl.value = '';
    inputEl.style.height = 'auto';

    save();
    renderSidebar();
    renderMessages();
    emptyState.style.display = 'none';
    messagesEl.classList.add('show');

    setBusy(true);
    showTyping();

    controller = new AbortController();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: chat.messages.map(m => ({ role: m.role, content: m.content }))
        }),
        signal: controller.signal
      });

      const data = await res.json();
      hideTyping();

      chat.messages.push({
        role: 'assistant',
        content: data.reply || 'No response received.',
        error: !data.ok
      });
    } catch (err) {
      hideTyping();
      if (err.name === 'AbortError') {
        chat.messages.push({
          role: 'assistant',
          content: '⏹ Generation stopped.',
          error: false
        });
      } else {
        chat.messages.push({
          role: 'assistant',
          content: 'Network error. Please check your connection and try again.',
          error: true
        });
      }
    } finally {
      controller = null;
      setBusy(false);
      save();
      renderSidebar();
      renderMessages();
      inputEl.focus();
    }
  }

  function retryLast() {
    if (busy) return;
    const chat = activeChat();
    if (!chat) return;

    while (chat.messages.length && chat.messages[chat.messages.length - 1].error) {
      chat.messages.pop();
    }

    let lastUser = null;
    for (let i = chat.messages.length - 1; i >= 0; i--) {
      if (chat.messages[i].role === 'user') { lastUser = chat.messages[i].content; break; }
    }
    if (!lastUser) return;

    chat.messages.pop();
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

  sendBtn.addEventListener('click', () => {
    if (busy) stopGeneration();
    else send(inputEl.value);
  });

  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
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

  // ---------- Init ----------
  initVoiceInput();

  if (!load()) {
    const c = newChatObject();
    chats.push(c);
    activeId = c.id;
  }

  ensureChat();
  renderSidebar();
  renderMessages();
  updateSendEnabled();
  inputEl.focus();
})();
