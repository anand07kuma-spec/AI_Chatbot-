(() => {
  'use strict';

  // ---------- Elements ----------
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('overlay');
  const menuBtn = document.getElementById('menuBtn');
  const newChatBtn = document.getElementById('newChatBtn');
  const clearBtn = document.getElementById('clearBtn');
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
  let listening = false;
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

  // ---------- Voice: text-to-speech (speak replies) ----------
  const hasTTS = 'speechSynthesis' in window;

  const SPK_ICON = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>';
  const STOP_SMALL_ICON = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';

  function resetSpeakBtn(btn) {
    if (!btn) return;
    btn.innerHTML = SPK_ICON + ' Listen';
    btn.classList.remove('speaking');
    if (speakingBtn === btn) speakingBtn = null;
  }

  function stopSpeaking() {
    if (hasTTS && window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
    }
    if (speakingBtn) resetSpeakBtn(speakingBtn);
    speakingBtn = null;
  }

  function speakText(text, btn) {
    if (!hasTTS) return;

    if (speakingBtn === btn) {
      stopSpeaking();
      return;
    }

    stopSpeaking();

    const u = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    const preferred =
      voices.find(v => v.lang === 'en-IN') ||
      voices.find(v => v.lang && v.lang.replace('_', '-').startsWith('en')) ||
      voices[0];
    if (preferred) u.voice = preferred;
    u.rate = 1.03;
    u.pitch = 1;

    speakingBtn = btn;
    btn.innerHTML = STOP_SMALL_ICON + ' Stop';
    btn.classList.add('speaking');

    u.onend = () => resetSpeakBtn(btn);
    u.onerror = () => resetSpeakBtn(btn);
    window.speechSynthesis.speak(u);
  }

  // ---------- Voice: speech-to-text (voice input) ----------
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;

  function setHint(msg) {
    inputEl.placeholder = msg;
    clearTimeout(hintTimer);
    hintTimer = setTimeout(() => {
      inputEl.placeholder = PLACEHOLDER;
    }, 3200);
  }

  function stopListening() {
    if (listening && recognition) {
      try { recognition.stop(); } catch (e) { /* ignore */ }
    }
  }

  function initVoiceInput() {
    if (!SR) {
      micBtn.style.display = 'none';
      return;
    }
    recognition = new SR();
    recognition.lang = 'en-IN';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      listening = true;
      micBtn.classList.add('listening');
      micBtn.title = 'Listening… click to stop';
      inputEl.placeholder = 'Listening… speak now';
      stopSpeaking();
    };

    recognition.onresult = (e) => {
      const text = (e.results[0][0].transcript || '').trim();
      if (text) {
        inputEl.value = text;
        inputEl.dispatchEvent(new Event('input'));
        inputEl.focus();
      }
    };

    recognition.onerror = (e) => {
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
        setHint('Microphone blocked — allow mic access in browser.');
      } else if (e.error === 'no-speech') {
        setHint('Didn\'t catch that — please try again.');
      } else if (e.error !== 'aborted') {
        setHint('Voice input failed — try again.');
      }
    };

    recognition.onend = () => {
      listening = false;
      micBtn.classList.remove('listening');
      micBtn.title = 'Speak your message';
      if (inputEl.placeholder === 'Listening… speak now') {
        inputEl.placeholder = PLACEHOLDER;
      }
    };

    micBtn.addEventListener('click', () => {
      if (busy) return;
      if (listening) {
        try { recognition.stop(); } catch (e) { /* ignore */ }
        return;
      }
      stopSpeaking();
      try { recognition.start(); } catch (e) { /* already started */ }
    });
  }

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

      const del = document.createElement('span');
      del.className = 'chat-del';
      del.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>';
      del.title = 'Delete chat';
      del.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteChat(chat.id);
      });
      item.appendChild(del);

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

      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.textContent = '✦';

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

      if (hasTTS) {
        const speakBtn = document.createElement('button');
        speakBtn.className = 'mini-btn';
        speakBtn.innerHTML = SPK_ICON + ' Listen';
        speakBtn.title = 'Read aloud';
        speakBtn.addEventListener('click', () => speakText(m.content, speakBtn));
        actions.appendChild(speakBtn);
      }

      if (m.error && i === msgs.length - 1) {
        const retryBtn = document.createElement('button');
        retryBtn.className = 'mini-btn';
        retryBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6"/><path d="M20.5 15a9 9 0 1 1-2.1-9.4L23 10"/></svg> Retry';
        retryBtn.addEventListener('click', () => retryLast());
        actions.appendChild(retryBtn);
      }

      body.appendChild(actions);
      wrap.appendChild(avatar);
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
      '<div class="avatar">✦</div>' +
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

  function clearActiveChat() {
    if (busy) return;
    const chat = activeChat();
    if (!chat) return;
    chat.messages = [];
    chat.title = 'New chat';
    save();
    renderSidebar();
    renderMessages();
    inputEl.focus();
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
  clearBtn.addEventListener('click', clearActiveChat);
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
  if (hasTTS) {
    window.speechSynthesis.getVoices();
    window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
  }
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
