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

  // ---------- Persistence ----------
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
  let pendingFiles = [];  // {id, name, kind: 'image'|'doc', mime?, data?, text?, thumb?, loading?}

  // ---------- Attachments ----------
  const MAX_FILES = 5;
  const MAX_IMG_BYTES = 8 * 1024 * 1024;
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

      if (isImage && file.size > MAX_IMG_BYTES) {
        setHint('Image too large (max 8MB): ' + file.name);
        continue;
      }

      const entry = { id: fileIdSeq++, name: file.name, kind: isImage ? 'image' : 'doc', loading: true };
      pendingFiles.push(entry);
      renderChips();

      if (isImage) {
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
      } else {
        try {
          const fd = new FormData();
          fd.append('file', file);
          const res = await fetch('/api/extract', { method: 'POST', body: fd });
          const data = await res.json();
          if (data.ok) {
            entry.text = data.text;
            entry.loading = false;
          } else {
            pendingFiles = pendingFiles.filter(x => x.id !== entry.id);
            setHint(data.error || 'Could not read file.');
          }
        } catch (e) {
          pendingFiles = pendingFiles.filter(x => x.id !== entry.id);
          setHint('Upload failed: ' + file.name);
        }
      }
      renderChips();
    }
    fileInput.value = '';
  }

  function save() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({ chats, activeId, nextId }));
    } catch (e) { /* storage unavailable */ }
  }

  function load() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
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
  let ttsUrl = null;

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
      if (ttsUrl) {
        URL.revokeObjectURL(ttsUrl);
        ttsUrl = null;
      }
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

      if (ttsUrl) URL.revokeObjectURL(ttsUrl);
      ttsUrl = URL.createObjectURL(blob);

      btn.classList.remove('loading');
      btn.classList.add('speaking');
      btn.innerHTML = STOP_SMALL_ICON + ' Stop';

      ttsAudio.src = ttsUrl;
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
      inputEl.placeholder = 'Recording… click mic to stop';
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

    const mime = (blob.type || 'audio/webm').split(';')[0];
    const ext = mime.includes('mp4') ? 'mp4' : mime.includes('ogg') ? 'ogg' : 'webm';
    const fd = new FormData();
    fd.append('file', blob, 'speech.' + ext);

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
    sendBtn.disabled = busy ? false : (inputEl.value.trim().length === 0 && pendingFiles.length === 0);
  }

  async function send(text) {
    if (busy) return;
    if (text) text = text.trim();
    if (!text && !pendingFiles.length) return;

    stopSpeaking();
    stopListening();

    const files = pendingFiles.filter(f => !f.loading);
    if (pendingFiles.some(f => f.loading)) {
      setHint('Wait for files to finish loading.');
      return;
    }

    const images = files.filter(f => f.kind === 'image' && f.data);
    const docs = files.filter(f => f.kind === 'doc' && f.text);

    if (!text) {
      text = images.length && !docs.length
        ? 'What is in this image?'
        : 'Please analyze the attached files.';
    }

    let content = text;
    if (docs.length) {
      const docParts = docs.map(d => '\n\n[Document: ' + d.name + ']\n' + d.text.slice(0, 8000));
      content += docParts.join('');
    }

    const attachMeta = files.map(f => ({
      kind: f.kind,
      name: f.name,
      thumb: f.kind === 'image' ? f.thumb : null,
    }));

    ensureChat();
    const chat = activeChat();

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

    try {
      const payload = {
        messages: chat.messages
          .slice(-24)
          .map(m => {
            let c = m.content.slice(0, 6000);
            if (m.role === 'user') {
              const userDocs = (m.attachments || []).filter(a => a.kind === 'doc');
              if (userDocs.length && !c.includes('[Document:')) {
                c += '\n[User previously attached: ' + userDocs.map(d => d.name).join(', ') + ']';
              }
            }
            return { role: m.role, content: c };
          })
      };
      if (images.length) {
        payload.images = images.map(im => ({ mime: im.mime, data: im.data.split(',', 1)[1] || im.data }));
      }

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      const ct = res.headers.get('content-type') || '';
      const data = ct.includes('application/json') ? await res.json() : { ok: false, reply: 'Server error. Please try again.' };
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

  attachBtn.addEventListener('click', () => {
    if (!busy) fileInput.click();
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
  renderChips();
  updateSendEnabled();
  inputEl.focus();
})();
