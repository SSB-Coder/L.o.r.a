/**
 * L.O.R.A Mobile Web App Client
 * Connects to Gateway via REST and WebSocket /v1/events with token auth + job status polling.
 */

(function () {
  let orb = null;
  let ws = null;
  let activeJobId = null;
  let pollTimer = null;
  let currentDepth = 'standard';
  let ttsEnabled = true;
  let submitting = false;
  let citationTrigger = null;
  const completedJobs = new Set();
  let speechEpoch = 0;
  let cancelPending = false;

  function finishJob(jobId) {
    if (jobId && completedJobs.has(jobId)) return false;
    if (jobId) completedJobs.add(jobId);
    if (completedJobs.size > 100) completedJobs.delete(completedJobs.values().next().value);
    clearPollTimer();
    activeJobId = null;
    queueBanner.classList.add('hidden');
    updateStatus('idle', 'Connected');
    return true;
  }

  // DOM Elements
  const statusText = document.getElementById('status-text');
  const statusDot = document.querySelector('.status-dot');
  const queueBanner = document.getElementById('queue-banner');
  const queuePos = document.getElementById('queue-pos');
  const chatFeed = document.getElementById('chat-feed');
  const askForm = document.getElementById('ask-form');
  const queryInput = document.getElementById('query-input');
  const depthChips = document.querySelectorAll('.depth-chip[data-depth]');
  const nbToggle = document.getElementById('nb-toggle');
  const ttsToggle = document.getElementById('tts-toggle');
  const floatingStop = document.getElementById('floating-stop-container');
  const stopBtn = document.getElementById('floating-stop-btn');
  const citationSheet = document.getElementById('citation-sheet');
  const sheetScrim = document.getElementById('sheet-scrim');
  const closeSheetBtn = document.getElementById('close-sheet-btn');

  // Token retrieval (URL ?token= -> pairing cookie -> localStorage)
  function getToken() {
    const urlParams = new URLSearchParams(window.location.hash.slice(1));
    const urlToken = urlParams.get('token');
    if (urlToken) {
      localStorage.setItem('lora_mobile_token', urlToken);
      history.replaceState(null, '', window.location.pathname);
      return urlToken;
    }
    return localStorage.getItem('lora_mobile_token') || '';
  }

  // Initialize the mobile pearl renderer.
  function initVisualizer() {
    orb = new HydroOrbVisualizer('hydro-orb-canvas');
    if (orb) {
      orb.start();
    }
  }

  // Update Status
  function updateStatus(state, message) {
    document.body.dataset.state = state;
    statusText.textContent = message || state.charAt(0).toUpperCase() + state.slice(1);
    if (orb) {
      orb.setState(state);
    }

    if (state === 'idle') {
      floatingStop.classList.add('hidden');
    } else if (state === 'thinking') {
      floatingStop.classList.remove('hidden');
    } else if (state === 'verifying') {
      floatingStop.classList.remove('hidden');
    } else if (state === 'speaking') {
      floatingStop.classList.remove('hidden');
    }
    document.getElementById('submit-btn').disabled = submitting || !!activeJobId;
    chatFeed.setAttribute('aria-busy', String(submitting || !!activeJobId));
  }

  // WebSocket Connection Lifecycle
  function connectWebSocket() {
    const token = getToken();
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/v1/events`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      // Mandatory 3-second authentication handshake
      ws.send(JSON.stringify({ type: 'auth', token: token }));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleWsEvent(msg);
      } catch (err) {
        console.error("Failed to parse event:", err);
      }
    };

    ws.onclose = async () => {
      try {
        const r = await fetch('/v1/notebooks', { headers: { 'Authorization': `Bearer ${token}` } });
        if (r.ok) {
          updateStatus('idle', 'Connected');
        } else if (r.status === 401 || r.status === 403) {
          updateStatus('disconnected', 'Rescan your desktop QR');
        } else {
          updateStatus('disconnected', 'Disconnected');
        }
      } catch (_e) {
        updateStatus('disconnected', 'Disconnected');
      }
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => {
      console.warn("WebSocket error:", err);
    };
  }

  // Handle incoming events from gateway
  function handleWsEvent(msg) {
    if (msg.job_id && (completedJobs.has(msg.job_id) || (activeJobId && msg.job_id !== activeJobId))) return;
    if (msg.type === 'auth_ok') {
      updateStatus('idle', 'Connected');
    } else if (msg.type === 'state') {
      updateStatus(msg.state);
    } else if (msg.type === 'queue') {
      if (msg.position > 0) {
        queueBanner.classList.remove('hidden');
        queuePos.textContent = msg.position;
      } else {
        queueBanner.classList.add('hidden');
      }
    } else if (msg.type === 'partial_text') {
      appendPartialToken(msg.text);
    } else if (msg.type === 'answer') {
      if (activeJobId && msg.job_id && msg.job_id !== activeJobId) return;
      if (finishJob(msg.job_id)) renderCompletedAnswer(msg);
    } else if (msg.type === 'error') {
      clearPollTimer();
      activeJobId = null;
      appendSystemMessage(`Error: ${msg.message}`);
      updateStatus('idle', 'Connected');
    }
  }

  function clearPollTimer() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function startJobPolling(jobId) {
    clearPollTimer();
    const token = getToken();
    pollTimer = setInterval(async () => {
      if (!activeJobId || activeJobId !== jobId) {
        clearPollTimer();
        return;
      }
      try {
        const res = await fetch(`/v1/jobs/${jobId}`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (!res.ok) return;
        const job = await res.json();
        if (activeJobId !== jobId) return; // A socket answer or cancellation won the race.
        if (job.status === 'completed' && job.result) {
          if (!finishJob(jobId)) return;
          const r = job.result;
          const citations = (r.sources || []).map((s, idx) => ({
            id: s.id || `S${idx + 1}`,
            title: s.title || '',
            source_url: s.url || '',
            quote: s.quote || '',
            tier: s.tier ?? 2
          }));
          renderCompletedAnswer({
            text: r.answer || '',
            citations: citations,
            kind: r.kind || 'chat',
            source_check: r.verification?.source_check
          });
        } else if (job.status === 'failed') {
          clearPollTimer();
          activeJobId = null;
          appendSystemMessage(`Error: ${job.error || 'Job failed'}`);
          updateStatus('idle', 'Connected');
        } else if (job.status === 'cancelled') {
          finishJob(jobId);
          discardPartialAnswer();
          appendSystemMessage('Response stopped.');
        }
      } catch (e) {}
    }, 1200);
  }

  // Append user or assistant messages
  function appendMessage(role, text) {
    document.getElementById('welcome')?.remove();
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = text;
    msgDiv.appendChild(bubble);
    chatFeed.appendChild(msgDiv);
    chatFeed.scrollTop = chatFeed.scrollHeight;
    return bubble;
  }

  function appendSystemMessage(text) {
    document.getElementById('welcome')?.remove();
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system-message';
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = text;
    msgDiv.appendChild(bubble);
    chatFeed.appendChild(msgDiv);
    chatFeed.scrollTop = chatFeed.scrollHeight;
  }

  let currentAssistantBubble = null;

  function discardPartialAnswer() {
    currentAssistantBubble?.parentElement.remove();
    currentAssistantBubble = null;
  }

  function appendPartialToken(text) {
    if (!currentAssistantBubble) {
      currentAssistantBubble = appendMessage('assistant', '');
    }
    currentAssistantBubble.textContent += text;
    chatFeed.scrollTop = chatFeed.scrollHeight;
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Chat stays conversational. Research has sources, without a truth score.
  function renderCompletedAnswer(data) {
    if (currentAssistantBubble) currentAssistantBubble.parentElement.remove();
    currentAssistantBubble = null;
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant-message';
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    let formattedText = escapeHtml(data.text || '');
    if (data.citations && data.citations.length > 0) {
      data.citations.forEach((c) => {
        const badgeTag = `[${escapeHtml(c.id)}]`;
        const htmlBadge = `<span class="citation-badge" data-id="${escapeHtml(c.id)}" data-title="${encodeURIComponent(c.title || '')}" data-url="${encodeURIComponent(c.source_url || '')}" data-quote="${encodeURIComponent(c.quote || '')}" data-tier="${c.tier || 1}">${escapeHtml(c.id)}</span>`;
        formattedText = formattedText.replaceAll(badgeTag, htmlBadge);
      });
    }
    bubble.innerHTML = formattedText;

    if (data.source_check) {
      const details = document.createElement('details');
      details.className = 'source-check';
      const label = document.createElement('summary');
      label.textContent = data.source_check.status === 'references_checked' ? 'Sources cited' : 'Source references need attention';
      const note = document.createElement('p');
      note.textContent = data.source_check.note || '';
      details.append(label, note);
      bubble.appendChild(details);
    }

    msgDiv.appendChild(bubble);
    chatFeed.appendChild(msgDiv);
    chatFeed.scrollTop = chatFeed.scrollHeight;

    bubble.querySelectorAll('.citation-badge').forEach((badge) => {
      badge.setAttribute('role', 'button');
      badge.tabIndex = 0;
      badge.setAttribute('aria-label', `Open source ${badge.dataset.id}`);
      badge.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); badge.click(); }
      });
      badge.addEventListener('click', () => {
        showCitationSheet(
          badge.dataset.id,
          decodeURIComponent(badge.dataset.title),
          decodeURIComponent(badge.dataset.url),
          decodeURIComponent(badge.dataset.quote),
          badge.dataset.tier
        );
      });
    });

    if (ttsEnabled && 'speechSynthesis' in window && data.text) {
      speakAnswer(data.text);
    }
  }

  function showCitationSheet(id, title, url, quote, tier) {
    citationTrigger = document.activeElement;
    document.getElementById('citation-badge-label').textContent = id;
    document.getElementById('citation-title').textContent = title || 'Source Reference';
    document.getElementById('citation-url').textContent = url || '';
    document.getElementById('citation-tier').textContent = `Tier ${tier}`;
    document.getElementById('citation-quote').textContent = quote || 'Supporting excerpt recorded in notebook.';
    citationSheet.classList.remove('hidden');
    closeSheetBtn.focus();
  }

  function hideCitationSheet() {
    citationSheet.classList.add('hidden');
    citationTrigger?.focus();
  }

  function speakAnswer(text) {
    const epoch = ++speechEpoch;
    window.speechSynthesis.cancel();
    const cleanSpeech = text.replace(/\[S\d+[^\]]*\]/g, '').trim();
    if (!cleanSpeech) return;

    const utterance = new SpeechSynthesisUtterance(cleanSpeech);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
      if (epoch !== speechEpoch) return;
      updateStatus('speaking');
      if (orb) {
        orb.setAudioLevel(0.6);
      }
    };

    utterance.onend = utterance.onerror = () => {
      if (epoch !== speechEpoch) return;
      updateStatus('idle', 'Connected');
      if (orb) {
        orb.setAudioLevel(0.0);
      }
    };

    window.speechSynthesis.speak(utterance);
  }

  async function loadNotebooks() {
    const token = getToken();
    try {
      const res = await fetch('/v1/notebooks', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (!res.ok) return;
      const data = await res.json();
      const nbs = data.notebooks || [];
      if (!nbs.length) {
        appendSystemMessage('No saved notebooks yet. Save a research session on desktop or ask "Save this to my notebook".');
        return;
      }
      const listStr = nbs.map(n => `• ${n.title} (${n.id} · ${n.depth_mode})`).join('\n');
      appendSystemMessage(`Saved Notebooks (${nbs.length}):\n${listStr}`);
    } catch (err) {
      appendSystemMessage(`Could not load notebooks: ${err.message}`);
    }
  }

  // Submit Query
  async function submitQuery(text) {
    if (submitting || activeJobId) return;
    submitting = true;
    cancelPending = false;
    ++speechEpoch;
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    orb?.setAudioLevel(0);
    const token = getToken();
    appendMessage('user', text);
    updateStatus('thinking', 'Queued...');

    try {
      const res = await fetch('/v1/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          text: text,
          depth: currentDepth,
          run_mode: 'quick'
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        appendSystemMessage(`Request failed: ${errData.detail || res.statusText}`);
        updateStatus('idle', 'Connected');
        return;
      }

      const data = await res.json();
      if (!completedJobs.has(data.job_id)) {
        activeJobId = data.job_id;
        if (cancelPending) {
          await stopJob();
        } else {
          startJobPolling(activeJobId);
          updateStatus('thinking', 'Thinking');
        }
      }
      if (data.queue_position > 0) {
        queueBanner.classList.remove('hidden');
        queuePos.textContent = data.queue_position;
      }
    } catch (err) {
      appendSystemMessage(`Network error: ${err.message}`);
      updateStatus('idle', 'Connected');
    } finally {
      submitting = false;
      document.getElementById('submit-btn').disabled = !!activeJobId;
      chatFeed.setAttribute('aria-busy', String(!!activeJobId));
    }
  }

  // Cancel / Stop
  async function stopJob() {
    ++speechEpoch;
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    orb?.setAudioLevel(0);
    const jobId = activeJobId;
    if (!jobId) {
      cancelPending = submitting;
      updateStatus(submitting ? 'thinking' : 'idle', submitting ? 'Stopping' : 'Connected');
      return;
    }
    const token = getToken();
    clearPollTimer();
    try {
      const response = await fetch('/v1/stop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ job_id: jobId })
      });
      if (!response.ok) throw new Error('Stop request failed');
      if (finishJob(jobId)) {
        discardPartialAnswer();
        appendSystemMessage('Response stopped.');
      }
    } catch (err) {
      console.warn("Failed to stop job:", err);
      if (activeJobId === jobId) startJobPolling(jobId);
    }
  }

  function setupEvents() {
    askForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const val = queryInput.value.trim();
      if (!val) return;
      queryInput.value = '';
      queryInput.style.height = 'auto';
      submitQuery(val);
    });

    depthChips.forEach((chip) => {
      chip.addEventListener('click', () => {
        depthChips.forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        currentDepth = chip.dataset.depth;
        depthChips.forEach(c => c.setAttribute('aria-pressed', String(c === chip)));
        document.getElementById('depth-label').textContent = chip.firstChild.textContent.trim();
        chip.closest('details').open = false;
      });
    });

    if (nbToggle) {
      nbToggle.addEventListener('click', loadNotebooks);
    }

    ttsToggle.addEventListener('click', () => {
      ttsEnabled = !ttsEnabled;
      ttsToggle.classList.toggle('active', ttsEnabled);
      ttsToggle.setAttribute('aria-pressed', String(ttsEnabled));
      if (!ttsEnabled) {
        ++speechEpoch;
        window.speechSynthesis.cancel();
        orb?.setAudioLevel(0);
        updateStatus(activeJobId ? 'thinking' : 'idle', activeJobId ? 'Thinking' : 'Connected');
      }
    });

    stopBtn.addEventListener('click', stopJob);
    sheetScrim.addEventListener('click', hideCitationSheet);
    closeSheetBtn.addEventListener('click', hideCitationSheet);
    queryInput.addEventListener('input', () => {
      queryInput.style.height = 'auto';
      queryInput.style.height = `${Math.min(queryInput.scrollHeight, 120)}px`;
    });
    queryInput.addEventListener('keydown', e => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); askForm.requestSubmit(); }
    });
    document.addEventListener('keydown', e => {
      if (!citationSheet.classList.contains('hidden') && e.key === 'Escape') hideCitationSheet();
      if (!citationSheet.classList.contains('hidden') && e.key === 'Tab') { e.preventDefault(); closeSheetBtn.focus(); }
    });
    document.addEventListener('click', e => {
      document.querySelectorAll('.style-menu[open]').forEach(menu => { if (!menu.contains(e.target)) menu.open = false; });
    });
    const viewport = window.visualViewport;
    const fitViewport = () => { document.getElementById('app-container').style.height = `${viewport ? viewport.height : window.innerHeight}px`; };
    viewport?.addEventListener('resize', fitViewport);
    window.addEventListener('resize', fitViewport);
    window.addEventListener('pagehide', () => orb?.stop());
    window.addEventListener('pageshow', () => orb?.start());
    fitViewport();
  }

  window.addEventListener('DOMContentLoaded', () => {
    initVisualizer();
    setupEvents();
    connectWebSocket();
  });
})();
