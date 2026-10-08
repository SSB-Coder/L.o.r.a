/**
 * L.O.R.A Mobile Web App Client
 * Connects to Gateway via REST and WebSocket /v1/events with token auth.
 */

(function () {
  let orb = null;
  let ws = null;
  let activeJobId = null;
  let currentDepth = 'standard';
  let ttsEnabled = true;
  let activeSpeech = null;

  // DOM Elements
  const statusText = document.getElementById('status-text');
  const statusDot = document.querySelector('.status-dot');
  const queueBanner = document.getElementById('queue-banner');
  const queuePos = document.getElementById('queue-pos');
  const chatFeed = document.getElementById('chat-feed');
  const askForm = document.getElementById('ask-form');
  const queryInput = document.getElementById('query-input');
  const depthChips = document.querySelectorAll('.depth-chip');
  const ttsToggle = document.getElementById('tts-toggle');
  const floatingStop = document.getElementById('floating-stop-container');
  const stopBtn = document.getElementById('floating-stop-btn');
  const citationSheet = document.getElementById('citation-sheet');
  const sheetScrim = document.getElementById('sheet-scrim');
  const closeSheetBtn = document.getElementById('close-sheet-btn');

  // Token retrieval
  function getToken() {
    const urlParams = new URLSearchParams(window.location.search);
    const urlToken = urlParams.get('token');
    if (urlToken) {
      localStorage.setItem('lora_mobile_token', urlToken);
      return urlToken;
    }
    return localStorage.getItem('lora_mobile_token') || '';
  }

  // Initialize WebGL Hydro-Orb
  function initVisualizer() {
    orb = new HydroOrbVisualizer('hydro-orb-canvas');
    if (orb) {
      orb.start();
    }
  }

  // Update Status
  function updateStatus(state, message) {
    statusText.textContent = message || state.charAt(0).toUpperCase() + state.slice(1);
    if (orb) {
      orb.setState(state);
    }

    if (state === 'idle') {
      statusDot.style.backgroundColor = '#00e5ff';
      statusDot.style.boxShadow = '0 0 8px #00e5ff';
      floatingStop.classList.add('hidden');
    } else if (state === 'thinking') {
      statusDot.style.backgroundColor = '#7c4dff';
      statusDot.style.boxShadow = '0 0 8px #7c4dff';
      floatingStop.classList.remove('hidden');
    } else if (state === 'verifying') {
      statusDot.style.backgroundColor = '#0077b6';
      statusDot.style.boxShadow = '0 0 8px #0077b6';
      floatingStop.classList.remove('hidden');
    } else if (state === 'speaking') {
      statusDot.style.backgroundColor = '#00e5ff';
      statusDot.style.boxShadow = '0 0 8px #00e5ff';
    }
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

    ws.onclose = () => {
      updateStatus('idle', 'Disconnected');
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => {
      console.warn("WebSocket error:", err);
    };
  }

  // Handle incoming events from gateway
  function handleWsEvent(msg) {
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
      renderCompletedAnswer(msg);
      updateStatus('idle', 'Connected');
    } else if (msg.type === 'error') {
      appendSystemMessage(`Error: ${msg.message}`);
      updateStatus('idle', 'Connected');
    }
  }

  // Append user or assistant messages
  function appendMessage(role, text) {
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

  // Render finalized verified answer with clickable citations and confidence
  function renderCompletedAnswer(data) {
    currentAssistantBubble = null;
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant-message';
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    // Safely HTML-escape text first to prevent XSS
    let formattedText = escapeHtml(data.text || '');
    if (data.citations && data.citations.length > 0) {
      data.citations.forEach((c) => {
        const badgeTag = `[${escapeHtml(c.id)}]`;
        const htmlBadge = `<span class="citation-badge" data-id="${escapeHtml(c.id)}" data-title="${encodeURIComponent(c.title || '')}" data-url="${encodeURIComponent(c.source_url || '')}" data-quote="${encodeURIComponent(c.quote || '')}" data-tier="${c.tier || 1}">${escapeHtml(c.id)}</span>`;
        formattedText = formattedText.replaceAll(badgeTag, htmlBadge);
      });
    }
    bubble.innerHTML = formattedText;

    // Confidence and metadata footer
    const metaDiv = document.createElement('div');
    metaDiv.className = 'verif-meta';

    const confLevel = (data.confidence || 'HIGH').toLowerCase();
    const confPill = document.createElement('span');
    confPill.className = `confidence-pill conf-${confLevel}`;
    confPill.textContent = `Confidence: ${confLevel}`;
    metaDiv.appendChild(confPill);

    bubble.appendChild(metaDiv);

    // Unverified claims section built safely with DOM nodes
    if (data.unverified && data.unverified.length > 0) {
      const unverDiv = document.createElement('div');
      unverDiv.className = 'unverified-box';
      const strong = document.createElement('strong');
      strong.textContent = 'Could not verify:';
      unverDiv.appendChild(strong);
      const ul = document.createElement('ul');
      data.unverified.forEach((u) => {
        const li = document.createElement('li');
        li.textContent = u;
        ul.appendChild(li);
      });
      unverDiv.appendChild(ul);
      bubble.appendChild(unverDiv);
    }

    msgDiv.appendChild(bubble);
    chatFeed.appendChild(msgDiv);
    chatFeed.scrollTop = chatFeed.scrollHeight;

    // Attach citation click listeners
    bubble.querySelectorAll('.citation-badge').forEach((badge) => {
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

    // Client-side Web Speech TTS readout (0 MB host GPU load)
    if (ttsEnabled && 'speechSynthesis' in window && data.text) {
      speakAnswer(data.text);
    }
  }

  // Show citation detail bottom sheet
  function showCitationSheet(id, title, url, quote, tier) {
    document.getElementById('citation-badge-label').textContent = id;
    document.getElementById('citation-title').textContent = title || 'Source Reference';
    document.getElementById('citation-url').textContent = url || '';
    document.getElementById('citation-tier').textContent = `Tier ${tier}`;
    document.getElementById('citation-quote').textContent = quote || 'Supporting excerpt recorded in notebook.';
    citationSheet.classList.remove('hidden');
  }

  function hideCitationSheet() {
    citationSheet.classList.add('hidden');
  }

  // Client-side Web Speech TTS
  function speakAnswer(text) {
    window.speechSynthesis.cancel();
    // Strip markdown citations for clean speech
    const cleanSpeech = text.replace(/\[S\d+[^\]]*\]/g, '').trim();
    if (!cleanSpeech) return;

    const utterance = new SpeechSynthesisUtterance(cleanSpeech);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
      updateStatus('speaking');
      // Simulate audio excitation on the Hydro-Orb
      if (orb) {
        orb.setAudioLevel(0.6);
      }
    };

    utterance.onend = () => {
      updateStatus('idle', 'Connected');
      if (orb) {
        orb.setAudioLevel(0.0);
      }
    };

    window.speechSynthesis.speak(utterance);
  }

  // Submit Query
  async function submitQuery(text) {
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
          run_mode: 'standard'
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        appendSystemMessage(`Request failed: ${errData.detail || res.statusText}`);
        updateStatus('idle', 'Connected');
        return;
      }

      const data = await res.json();
      activeJobId = data.job_id;
      if (data.queue_position > 0) {
        queueBanner.classList.remove('hidden');
        queuePos.textContent = data.queue_position;
      }
    } catch (err) {
      appendSystemMessage(`Network error: ${err.message}`);
      updateStatus('idle', 'Connected');
    }
  }

  // Cancel / Stop
  async function stopJob() {
    if (!activeJobId) return;
    const token = getToken();
    try {
      await fetch('/v1/stop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ job_id: activeJobId })
      });
      appendSystemMessage("Research cancelled by user.");
      updateStatus('idle', 'Connected');
      activeJobId = null;
      window.speechSynthesis.cancel();
    } catch (err) {
      console.warn("Failed to stop job:", err);
    }
  }

  // Setup Event Listeners
  function setupEvents() {
    askForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const val = queryInput.value.trim();
      if (!val) return;
      queryInput.value = '';
      submitQuery(val);
    });

    depthChips.forEach((chip) => {
      chip.addEventListener('click', () => {
        depthChips.forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        currentDepth = chip.dataset.depth;
      });
    });

    ttsToggle.addEventListener('click', () => {
      ttsEnabled = !ttsEnabled;
      ttsToggle.classList.toggle('active', ttsEnabled);
      if (!ttsEnabled) {
        window.speechSynthesis.cancel();
      }
    });

    stopBtn.addEventListener('click', stopJob);
    sheetScrim.addEventListener('click', hideCitationSheet);
    closeSheetBtn.addEventListener('click', hideCitationSheet);
  }

  // Boot Application
  window.addEventListener('DOMContentLoaded', () => {
    initVisualizer();
    setupEvents();
    connectWebSocket();
  });
})();
