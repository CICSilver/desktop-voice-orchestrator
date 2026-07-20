const token = new URLSearchParams(location.search).get('token') || '';
const SAMPLE_RATE = 16000;
const state = {
  telemetry: [],
  vad: [],
  wakes: [],
  candidates: [],
  config: null,
  maxPoints: 400,
  source: 'processed',
  activeMode: 'live',
  recordingActive: null,
  latestSample: {live: null, replay: null},
  activation: {
    live: {active: false, status: '未激活'},
    replay: {active: false, status: '未激活'}
  },
  announcement: {live: null, replay: null},
  view: {
    frozen: false,
    snapshot: null,
    endSample: 0,
    durationSeconds: 20,
    freezeReason: '',
    drag: null,
    cursorRatio: null
  }
};
const $ = (id) => document.getElementById(id);

function modeOf(message) {
  return (message.source || message.payload?.source) === 'replay' ? 'replay' : 'live';
}

function candidateEndSample(candidate) {
  const spans = candidate.source_spans || [];
  const finalSpan = spans.length ? spans[spans.length - 1] : null;
  return Number(finalSpan?.end ?? candidate.wake_span?.end ?? candidate.trigger_sample ?? 0);
}

function resetModeStatus(mode) {
  state.latestSample[mode] = null;
  state.activation[mode] = {active: false, status: '未激活'};
  state.announcement[mode] = null;
  if (mode === state.activeMode) renderCommandStatus();
}

function hydrateActivation(mode, payload, status = '') {
  const active = payload.active !== false && !!payload.activation_id;
  const labels = {
    dormant: '未激活',
    keyword_turn: '关键词话段',
    armed_idle: '等待后续',
    followup_turn: '后续话段'
  };
  state.activation[mode] = {
    active,
    status: status || labels[payload.state] || (active ? '已激活' : (payload.status || '未激活')),
    activationId: payload.activation_id || '',
    turnIndex: payload.turn_index,
    idleDeadlineSample: Number(payload.idle_deadline_sample || 0),
    hardDeadlineSample: Number(payload.hard_deadline_sample || 0),
    reason: payload.reason || ''
  };
  if (mode === state.activeMode) renderCommandStatus();
}

function hydrateAnnouncement(mode, payload, fallbackStatus = '') {
  if (!payload || typeof payload !== 'object' || !payload.text) return;
  state.announcement[mode] = {
    text: payload.text,
    kind: payload.kind || '',
    status: payload.status || fallbackStatus
  };
  if (mode === state.activeMode) renderCommandStatus();
}

function updateActivation(message) {
  const mode = modeOf(message);
  const payload = message.payload || {};
  const atSample = payload.at_sample ?? message.timestamp_sample;
  if (atSample != null) {
    state.latestSample[mode] = Math.max(state.latestSample[mode] ?? 0, Number(atSample));
  }
  if (message.type === 'activation_started' || message.type === 'activation_refreshed') {
    hydrateActivation(mode, payload, message.type === 'activation_refreshed' ? '已续期' : '已激活');
  } else {
    hydrateActivation(mode, {...payload, active: false},
      message.type === 'activation_expired' ? '已过期' : '已取消');
  }
}

function renderCommandStatus() {
  const activation = state.activation[state.activeMode];
  const turn = activation.turnIndex == null ? '' : ` · #${activation.turnIndex}`;
  const reason = !activation.active && activation.reason ? ` · ${activation.reason}` : '';
  $('activation-state').textContent = `${activation.status}${turn}${reason}`;

  if (!activation.active) {
    $('activation-countdown').textContent = '—';
  } else {
    const now = state.latestSample[state.activeMode];
    if (now == null) {
      $('activation-countdown').textContent = '等待样本时间';
    } else {
      const hardRemaining = Math.max(0, activation.hardDeadlineSample - now) / SAMPLE_RATE;
      const idle = activation.idleDeadlineSample > 0
        ? `空闲 ${(Math.max(0, activation.idleDeadlineSample - now) / SAMPLE_RATE).toFixed(1)}s`
        : '话段进行中';
      $('activation-countdown').textContent = `${idle} / 上限 ${hardRemaining.toFixed(1)}s`;
    }
  }

  const announcement = state.announcement[state.activeMode];
  if (!announcement) {
    $('announcement-text').textContent = '尚无播报';
    return;
  }
  const status = announcement.status && announcement.status !== 'delivered'
    ? ` [${announcement.status}]` : '';
  $('announcement-text').textContent = `${announcement.text || '无文字内容'}${status}`;
}

async function command(action, payload = {}) {
  try {
    const response = await fetch(`/api/command?token=${encodeURIComponent(token)}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action, ...payload})
    });
    const result = await response.json();
    if (!result.ok) addEvent({type: 'command_error', payload: {message: result.error || 'unknown error'}, seq: '—'});
    return result;
  } catch (error) {
    addEvent({type: 'command_error', level: 'error', payload: {message: String(error)}, seq: '—'});
    return {ok: false, error: String(error)};
  }
}

function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${location.host}/ws?token=${encodeURIComponent(token)}`);
  ws.onopen = () => {
    $('connection').textContent = '在线';
    $('connection').className = 'pill online';
    command('state.get');
  };
  ws.onclose = () => {
    $('connection').textContent = '离线';
    $('connection').className = 'pill offline';
    setTimeout(connect, 1200);
  };
  ws.onmessage = ({data}) => handle(JSON.parse(data));
}

function handle(message) {
  const {type, payload} = message;
  if (type === 'telemetry') {
    const messageMode = modeOf(message);
    if (messageMode !== state.activeMode) return;
    state.latestSample[messageMode] = Number(payload.sample || 0);
    state.telemetry.push(payload);
    state.vad.push(!!payload.vad);
    while (state.telemetry.length > state.maxPoints) state.telemetry.shift();
    while (state.vad.length > state.maxPoints) state.vad.shift();

    const cutoff = Number(payload.sample || 0) - 20 * SAMPLE_RATE;
    state.wakes = state.wakes.filter(hit => Number(hit.wake_span?.end ?? hit.at ?? 0) >= cutoff);
    state.candidates = state.candidates.filter(item => candidateEndSample(item) >= cutoff);
    $('rms').textContent = Number(payload[state.source]?.rms ?? payload.processed?.rms ?? 0).toFixed(3);
    $('queue-depth').textContent = payload.queue_depth ?? 0;
    $('drops').textContent = payload.telemetry_dropped ?? 0;
    if (payload.aec) updateAec(payload.aec);
    if (message.source === 'replay') $('replay-seek').value = Number(payload.sample || 0) / SAMPLE_RATE;
    renderCommandStatus();
    syncViewControls();
  } else if (type === 'kws_hit') {
    if ((message.source === 'replay' ? 'replay' : 'live') === state.activeMode) {
      state.wakes.push({at: message.timestamp_sample, ...payload});
    }
    addEvent(message);
  } else if (type === 'candidate') {
    if ((message.source === 'replay' ? 'replay' : 'live') === state.activeMode) {
      state.candidates.push(payload);
    }
    addEvent(message);
  } else if (type === 'aec_status' || type === 'aec_stats') {
    updateAec(payload || {});
    addEvent(message);
  } else if (type === 'asr_status') {
    $('asr-state').textContent = payload.state || payload.status || '未知';
    addEvent(message);
  } else if (type === 'asr_partial' || type === 'asr_final') {
    const final = type === 'asr_final' || payload.is_final;
    $(final ? 'asr-final' : 'asr-partial').textContent = payload.text || '—';
    $('asr-state').textContent = final ? '最终结果' : '识别中';
    $('asr-utterance').textContent = payload.utterance_id || '—';
    $('asr-latency').textContent = Number.isFinite(Number(payload.latency_ms)) ? `${Number(payload.latency_ms).toFixed(0)} ms` : '—';
    $('asr-rtf').textContent = Number.isFinite(Number(payload.rtf)) ? Number(payload.rtf).toFixed(3) : '—';
    $('asr-revision').textContent = payload.revision ?? '—';
    addEvent(message);
  } else if (type === 'command_plan' || type === 'command_rejected') {
    $('command-plan').textContent = type === 'command_rejected'
      ? `拒绝：${payload.reason || '规则不匹配'}`
      : (payload.summary || payload.normalized_text || JSON.stringify(payload.actions || []));
    $('action-state').textContent = type === 'command_plan' ? '已规划' : '已拒绝';
    addEvent(message);
  } else if (type === 'activation_started' || type === 'activation_refreshed' ||
             type === 'activation_expired' || type === 'activation_cancelled') {
    updateActivation(message);
    addEvent(message);
  } else if (type === 'activation_state') {
    hydrateActivation(modeOf(message), payload);
    addEvent(message);
  } else if (type === 'announcement' || type.startsWith('announcement_')) {
    const mode = modeOf(message);
    hydrateAnnouncement(mode, payload, type.slice('announcement_'.length));
    addEvent(message);
  } else if (type.startsWith('action_')) {
    $('action-state').textContent = type.slice('action_'.length);
    $('action-result').textContent = payload.message || payload.error_code || JSON.stringify(payload);
    addEvent(message);
  } else if (type === 'config_state' || type === 'config_applied') {
    const pending = payload.restart_required || [];
    loadConfig(pending.length && payload.saved_config ? payload.saved_config : (payload.config || payload));
    $('restart-required').hidden = !pending.length;
    $('restart-required').textContent = pending.length ? `已保存，重启后生效：${pending.join('、')}` : '';
    if (payload.activation_state) hydrateActivation(modeOf(message), payload.activation_state);
    if (payload.latest_announcement) {
      hydrateAnnouncement(modeOf(message), payload.latest_announcement);
    }
    addEvent(message);
  } else if (type === 'runtime_state') {
    if (payload.activation) hydrateActivation(modeOf(message), payload.activation);
    if (payload.latest_announcement) {
      hydrateAnnouncement(modeOf(message), payload.latest_announcement);
    }
    addEvent(message);
  } else if (type === 'recording_state') {
    const active = !!payload.active;
    const previous = state.recordingActive;
    state.recordingActive = active;
    $('recording').classList.toggle('active', active);
    $('recording').lastChild.textContent = active ? ` 录制中 · ${payload.session || ''}` : ' 未录制';

    if (active) resumeLiveView();
    else if (previous === true && !state.view.frozen) freezeView('录制停止');
    addEvent(message);
  } else if (type === 'sessions') {
    const select = $('sessions');
    const current = select.value;
    select.innerHTML = '<option value="">选择回放会话</option>';
    for (const session of payload.sessions || []) select.add(new Option(session.name, session.path));
    select.value = current;
  } else if (type === 'runtime_mode') {
    const mode = payload.mode === 'replay' ? 'replay' : 'live';
    setActiveMode(mode, mode !== state.activeMode);
    addEvent(message);
  } else if (type === 'replay_state') {
    if (state.activeMode === 'replay') {
      $('source').textContent = payload.error ? 'REPLAY !' : (payload.playing ? 'REPLAY ▶' : 'REPLAY');
    }
    $('replay-seek').max = Math.max(1, Number(payload.duration_seconds || 0));
    $('replay-seek').value = Number(payload.position_seconds || 0);
    if (payload.error) {
      $('action-result').textContent = `回放失败：${payload.error}`;
    }
    addEvent(message);
  } else {
    addEvent(message);
  }
}

function updateAec(aec) {
  $('aec-state').textContent = aec.state || aec.status || (aec.enabled === false ? '禁用' : '运行中');
  $('aec-delay').textContent = aec.external_delay_ms == null ? '—' : `${Number(aec.external_delay_ms).toFixed(1)} ms`;
  $('aec-drift').textContent = aec.drift_ppm == null ? '—' : `${Number(aec.drift_ppm).toFixed(1)} ppm`;
  $('aec-erle').textContent = aec.erle_db == null ? '—' : `${Number(aec.erle_db).toFixed(1)} dB`;
  $('aec-fifo').textContent = aec.render_fifo_ms == null ? '—' : `${Number(aec.render_fifo_ms).toFixed(0)} ms`;
  const processing = aec.processing_p95_ms ?? aec.processing_max_ms;
  $('aec-p95').textContent = processing == null ? '—' : `${Number(processing).toFixed(2)} ms`;
  $('aec-resets').textContent = aec.reset_count ?? 0;
}

function addEvent(message) {
  const row = document.createElement('tr');
  const at = message.timestamp_sample ? (message.timestamp_sample / SAMPLE_RATE).toFixed(2) + 's' : new Date().toLocaleTimeString();
  const detail = JSON.stringify(message.payload ?? {}).slice(0, 500);
  const level = message.level || (String(message.type).includes('error') || message.type === 'config_rejected' ? 'error' : 'info');
  row.innerHTML = `<td>${message.seq ?? '—'}</td><td>${at}</td><td class="level-${level}">${level}</td><td>${message.type}</td><td></td>`;
  row.lastChild.textContent = detail;
  $('events').prepend(row);
  while ($('events').children.length > 250) $('events').lastChild.remove();
}

function loadConfig(config) {
  state.config = structuredClone(config);
  state.maxPoints = Math.max(20, Number(config.web?.telemetry_hz || 20) * 20);
  $('config-revision').textContent = `rev ${config.revision ?? '—'}`;
  for (const input of $('config-form').elements) {
    if (!input.name) continue;
    const [group, key] = input.name.split('.');
    if (config[group]?.[key] !== undefined) {
      if (input.type === 'checkbox') input.checked = !!config[group][key];
      else if (input.dataset.list !== undefined && Array.isArray(config[group][key])) {
        input.value = config[group][key].join(', ');
      } else input.value = config[group][key];
    }
  }
}

function configPatch() {
  const patch = {};
  for (const input of $('config-form').elements) {
    if (!input.name) continue;
    const [group, key] = input.name.split('.');
    patch[group] ??= {};
    patch[group][key] = input.type === 'checkbox' ? input.checked
      : (input.type === 'number' ? Number(input.value)
        : (input.dataset.list !== undefined
          ? input.value.split(/[,，\n]+/).map((value) => value.trim()).filter(Boolean)
          : input.value));
  }
  return patch;
}

function activeViewData() {
  return state.view.snapshot || state;
}

function clearTimelineData() {
  state.telemetry = [];
  state.vad = [];
  state.wakes = [];
  state.candidates = [];
  state.view.frozen = false;
  state.view.snapshot = null;
  state.view.endSample = 0;
  state.view.freezeReason = '';
  state.view.drag = null;
  state.view.cursorRatio = null;
}

function setActiveMode(mode, clear = false) {
  const changed = state.activeMode !== mode;
  if (clear || changed) clearTimelineData();
  state.activeMode = mode;
  if (clear || changed) resetModeStatus(mode);
  $('source').textContent = mode === 'replay' ? 'REPLAY' : 'LIVE';
  for (const id of ['replay-play', 'replay-pause', 'replay-speed', 'replay-seek']) {
    $(id).disabled = mode !== 'replay';
  }
  syncViewControls();
  renderCommandStatus();
}

function snapshotCurrentView() {
  return {
    telemetry: state.telemetry.slice(),
    vad: state.vad.slice(),
    wakes: structuredClone(state.wakes),
    candidates: structuredClone(state.candidates)
  };
}

function sampleOf(frame) {
  return Number(frame?.sample || 0);
}

function viewLimits(data) {
  if (!data.telemetry.length) return null;
  const first = sampleOf(data.telemetry[0]);
  const last = sampleOf(data.telemetry.at(-1));
  const windowSamples = state.view.durationSeconds * SAMPLE_RATE;
  return {first, last, windowSamples, minEnd: Math.min(last, first + windowSamples)};
}

function clampViewEnd(data, requested) {
  const limits = viewLimits(data);
  if (!limits) return 0;
  return Math.max(limits.minEnd, Math.min(limits.last, Number(requested || limits.last)));
}

function viewBounds(data = activeViewData()) {
  const limits = viewLimits(data);
  if (!limits) return null;
  const requested = state.view.frozen ? state.view.endSample : limits.last;
  const end = clampViewEnd(data, requested);
  return {...limits, start: end - limits.windowSamples, end};
}

function freezeView(reason = '手动查看') {
  if (state.view.frozen || !state.telemetry.length) return;
  state.view.snapshot = snapshotCurrentView();
  state.view.frozen = true;
  state.view.freezeReason = reason;
  state.view.endSample = sampleOf(state.view.snapshot.telemetry.at(-1));
  state.view.cursorRatio = null;
  syncViewControls();
}

function resumeLiveView() {
  state.view.frozen = false;
  state.view.snapshot = null;
  state.view.endSample = 0;
  state.view.freezeReason = '';
  state.view.drag = null;
  state.view.cursorRatio = null;
  syncViewControls();
}

function syncViewControls() {
  const mode = $('view-mode');
  mode.textContent = state.view.frozen ? '已冻结' : (state.activeMode === 'replay' ? '回放跟随' : '实时跟随');
  mode.className = `pill ${state.view.frozen ? 'view-frozen' : 'view-live'}`;
  mode.dataset.mode = state.view.frozen ? 'frozen' : 'live';
  mode.title = state.view.frozen ? state.view.freezeReason :
      (state.activeMode === 'replay' ? '波形跟随当前回放样本' : '波形跟随最新实时样本');
  $('view-live').disabled = !state.view.frozen && state.activeMode === 'live';
  $('window-label').textContent = `窗口：${state.view.durationSeconds} 秒`;

  const data = activeViewData();
  const bounds = viewBounds(data);
  const scrubber = $('view-scrubber');
  scrubber.disabled = !bounds;
  if (!bounds) {
    $('view-range').textContent = '等待音频…';
    return;
  }

  scrubber.min = (bounds.minEnd / SAMPLE_RATE).toFixed(3);
  scrubber.max = (bounds.last / SAMPLE_RATE).toFixed(3);
  scrubber.value = (bounds.end / SAMPLE_RATE).toFixed(3);
  $('view-range').textContent = `${Math.max(0, bounds.start / SAMPLE_RATE).toFixed(2)}–${(bounds.end / SAMPLE_RATE).toFixed(2)} s`;
}

function canvas(id) {
  const element = $(id);
  const dpr = devicePixelRatio || 1;
  const rect = element.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (element.width !== width || element.height !== height) {
    element.width = width;
    element.height = height;
  }
  const context = element.getContext('2d');
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {c: context, w: rect.width, h: rect.height};
}

function drawCursor(track, x, label = '') {
  track.c.save();
  track.c.strokeStyle = 'rgba(242,189,102,.9)';
  track.c.lineWidth = 1;
  track.c.beginPath();
  track.c.moveTo(x, 0);
  track.c.lineTo(x, track.h);
  track.c.stroke();
  if (label) {
    track.c.font = '11px Consolas, monospace';
    const width = track.c.measureText(label).width + 10;
    const left = Math.min(Math.max(0, x + 5), track.w - width);
    track.c.fillStyle = 'rgba(20,16,9,.88)';
    track.c.fillRect(left, track.h - 23, width, 18);
    track.c.fillStyle = '#f2bd66';
    track.c.fillText(label, left + 5, track.h - 10);
  }
  track.c.restore();
}

function draw() {
  const wave = canvas('waveform');
  const aec = canvas('aec-band');
  const vad = canvas('vad');
  const kws = canvas('kws');
  wave.c.clearRect(0, 0, wave.w, wave.h);
  aec.c.clearRect(0, 0, aec.w, aec.h);
  vad.c.clearRect(0, 0, vad.w, vad.h);
  kws.c.clearRect(0, 0, kws.w, kws.h);

  const data = activeViewData();
  const bounds = viewBounds(data);
  if (!bounds) {
    requestAnimationFrame(draw);
    return;
  }

  const xAt = (sample, width) => (Number(sample) - bounds.start) / bounds.windowSamples * width;
  const pointWidth = Math.max(1, wave.w / Math.max(1, state.view.durationSeconds * Number(state.config?.web?.telemetry_hz || 20)));

  const strokeWave = (source, color, width = 1) => {
    wave.c.strokeStyle = color;
    wave.c.lineWidth = width;
    wave.c.beginPath();
    data.telemetry.forEach((frame) => {
      const sample = sampleOf(frame);
      if (sample < bounds.start || sample > bounds.end) return;
      const value = frame[source] || frame.processed || {min: 0, max: 0};
      const x = xAt(sample, wave.w);
      let y1 = wave.h / 2 - (value.max || 0) * wave.h * .44;
      let y2 = wave.h / 2 - (value.min || 0) * wave.h * .44;
      if (Math.abs(y2 - y1) < 1) {
        const center = (y1 + y2) / 2;
        y1 = center - .5;
        y2 = center + .5;
      }
      wave.c.moveTo(x, y1);
      wave.c.lineTo(x, y2);
    });
    wave.c.stroke();
  };
  if (state.source === 'compare') {
    strokeWave('microphone', '#f2bd66', 1);
    strokeWave('processed', '#59e1d9', 1.25);
  } else {
    strokeWave(state.source, state.source === 'microphone' ? '#f2bd66' : '#59e1d9');
  }

  data.telemetry.forEach((frame) => {
    const sample = sampleOf(frame);
    if (sample < bounds.start || sample > bounds.end) return;
    const status = frame.aec || {};
    aec.c.fillStyle = status.degraded ? 'rgba(255,107,114,.72)'
      : status.active ? 'rgba(89,225,217,.72)' : 'rgba(83,97,115,.55)';
    aec.c.fillRect(xAt(sample, aec.w), 6, Math.ceil(pointWidth) + 1, aec.h - 12);
  });

  vad.c.fillStyle = 'rgba(116,226,154,.72)';
  data.telemetry.forEach((frame, index) => {
    const sample = sampleOf(frame);
    if (data.vad[index] && sample >= bounds.start && sample <= bounds.end) {
      vad.c.fillRect(xAt(sample, vad.w), 8, Math.ceil(pointWidth) + 1, vad.h - 16);
    }
  });

  kws.c.strokeStyle = '#ff6b72';
  kws.c.fillStyle = 'rgba(255,107,114,.18)';
  for (const hit of data.wakes) {
    const x = xAt(hit.at, kws.w);
    if (x >= 0 && x <= kws.w) {
      kws.c.fillRect(x - 2, 0, 4, kws.h);
      kws.c.fillStyle = '#ff9ca1';
      kws.c.fillText(hit.keyword || 'KWS', x + 4, 17);
    }
    for (const [index, sample] of (hit.token_samples || []).entries()) {
      const tokenX = xAt(sample, kws.w);
      if (tokenX >= 0 && tokenX <= kws.w) {
        kws.c.fillRect(tokenX, 24, 1, kws.h - 24);
        kws.c.fillText(hit.tokens?.[index] || '', tokenX + 2, kws.h - 5);
      }
    }
    const wake = hit.wake_span;
    if (wake) {
      const wakeX = xAt(wake.start, wave.w);
      const wakeWidth = (wake.end - wake.start) / bounds.windowSamples * wave.w;
      wave.c.fillStyle = 'rgba(255,107,114,.22)';
      wave.c.fillRect(wakeX, 0, wakeWidth, wave.h);
    }
  }

  wave.c.fillStyle = 'rgba(93,168,255,.14)';
  for (const item of data.candidates) {
    for (const span of item.source_spans || []) {
      const x = xAt(span.start, wave.w);
      const width = (span.end - span.start) / bounds.windowSamples * wave.w;
      wave.c.fillRect(x, 0, width, wave.h);
    }
    const first = item.source_spans?.[0];
    if (first) {
      const x = xAt(first.start, wave.w);
      wave.c.fillStyle = '#8bc1ff';
      wave.c.fillText(item.position || item.origin || 'candidate', x + 4, 16);
      wave.c.fillStyle = 'rgba(93,168,255,.14)';
    }
  }

  if (state.view.cursorRatio !== null) {
    const ratio = Math.max(0, Math.min(1, state.view.cursorRatio));
    const cursorSample = bounds.start + ratio * bounds.windowSamples;
    drawCursor(wave, ratio * wave.w, `${(cursorSample / SAMPLE_RATE).toFixed(3)} s`);
    drawCursor(aec, ratio * aec.w);
    drawCursor(vad, ratio * vad.w);
    drawCursor(kws, ratio * kws.w);
  }

  requestAnimationFrame(draw);
}

function installTimelineDrag(element) {
  const wrap = element.closest('.canvas-wrap');
  element.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    freezeView('手动拖动');
    const bounds = viewBounds();
    if (!bounds) return;
    state.view.drag = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startEnd: bounds.end,
      width: Math.max(1, element.getBoundingClientRect().width)
    };
    element.setPointerCapture(event.pointerId);
    wrap.classList.add('dragging');
    event.preventDefault();
  });

  element.addEventListener('pointermove', (event) => {
    const rect = element.getBoundingClientRect();
    if (state.view.drag?.pointerId === event.pointerId) {
      const delta = event.clientX - state.view.drag.startX;
      const requested = state.view.drag.startEnd - delta / state.view.drag.width * state.view.durationSeconds * SAMPLE_RATE;
      state.view.endSample = clampViewEnd(activeViewData(), requested);
      syncViewControls();
      return;
    }
    state.view.cursorRatio = (event.clientX - rect.left) / Math.max(1, rect.width);
  });

  const endDrag = (event) => {
    if (state.view.drag?.pointerId !== event.pointerId) return;
    state.view.drag = null;
    wrap.classList.remove('dragging');
    if (element.hasPointerCapture(event.pointerId)) element.releasePointerCapture(event.pointerId);
  };
  element.addEventListener('pointerup', endDrag);
  element.addEventListener('pointercancel', endDrag);
  element.addEventListener('pointerleave', () => {
    if (!state.view.drag) state.view.cursorRatio = null;
  });
}

$('wave-source').onchange = (event) => state.source = event.target.value;
$('clear-events').onclick = () => $('events').innerHTML = '';
$('record-start').onclick = () => command('recording.start');
$('record-stop').onclick = async () => {
  const wasFollowing = !state.view.frozen;
  if (state.recordingActive === true) freezeView('录制停止');
  const result = await command('recording.stop');
  if (!result.ok && wasFollowing) resumeLiveView();
};
$('replay-open').onclick = async () => {
  const session = $('sessions').value;
  if (!session) {
    addEvent({type: 'command_error', level: 'error', payload: {message: '请先选择回放会话'}, seq: '—'});
    return;
  }
  const previousMode = state.activeMode;
  setActiveMode('replay', true);
  const result = await command('replay.open', {session});
  if (!result.ok) setActiveMode(previousMode, true);
};
$('replay-play').onclick = () => command('replay.play');
$('replay-pause').onclick = () => command('replay.pause');
$('replay-speed').onchange = () => command('replay.speed', {speed: Number($('replay-speed').value)});
$('replay-seek').onchange = () => {
  const seconds = Number($('replay-seek').value);
  clearTimelineData();
  resetModeStatus('replay');
  syncViewControls();
  command('replay.seek', {seconds});
};
$('config-form').onsubmit = (event) => {
  event.preventDefault();
  command('config.apply', {patch: configPatch(), save: false});
};
$('config-save').onclick = () => command('config.apply', {patch: configPatch(), save: true});
$('config-reset').onclick = () => state.config && loadConfig(state.config);
$('view-live').onclick = async () => {
  if (state.activeMode === 'replay') {
    const result = await command('live.resume');
    if (result.ok && state.activeMode !== 'live') setActiveMode('live', true);
    return;
  }
  resumeLiveView();
};
$('view-duration').onchange = () => {
  freezeView('调整显示窗口');
  state.view.durationSeconds = Number($('view-duration').value);
  state.view.endSample = clampViewEnd(activeViewData(), state.view.endSample);
  syncViewControls();
};
$('view-scrubber').addEventListener('pointerdown', () => freezeView('拖动时间位置'));
$('view-scrubber').addEventListener('input', () => {
  freezeView('拖动时间位置');
  state.view.endSample = Number($('view-scrubber').value) * SAMPLE_RATE;
  syncViewControls();
});
for (const id of ['waveform', 'aec-band', 'vad', 'kws']) installTimelineDrag($(id));

syncViewControls();
setActiveMode('live');
connect();
draw();
