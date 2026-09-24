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
  recordingPending: false,
  recordingStartedAtMs: 0,
  recordingSession: '',
  sessions: new Map(),
  loadedSession: null,
  timelineOrigin: {live: null, replay: null},
  latestSample: {live: null, replay: null},
  replayPlaying: false,
  audioFinished: false,
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
const replayAudio = $('replay-audio');

function showControlMessage(message = '', level = '') {
  const output = $('control-message');
  output.textContent = message;
  output.className = `control-message ${level}`.trim();
}

function sessionName(path = '') {
  return String(path).split(/[\\/]/).filter(Boolean).at(-1) || '';
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainder = total % 60;
  return hours
    ? `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
    : `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
}

function parseActionLabel(action) {
  if (action.type === 'media.play') return '播放';
  if (action.type === 'media.pause') return '暂停';
  if (action.type === 'audio.volume.adjust') {
    const delta = Number(action.volume_delta_percent || 0);
    return `音量${delta >= 0 ? '+' : ''}${delta}%`;
  }
  return action.type || '未知动作';
}

function resetParseDebug(message = '等待回放或手动输入') {
  $('parse-state').textContent = state.activeMode === 'replay' ? '等待解析' : '仅回放可用';
  $('parse-state').className = 'pill';
  $('parse-input').textContent = '—';
  $('parse-normalized').textContent = '—';
  $('parse-actions').textContent = '—';
  $('parse-error').textContent = message;
  $('parse-error').className = '';
  $('parse-meta').textContent = '只调用当前规则解析器，强制 dry-run，不提交系统动作。';
}

function renderParseDebug(payload) {
  const matched = Boolean(payload.matched);
  const plan = payload.plan || {};
  const actions = plan.actions || [];
  const error = payload.error || null;
  const kind = payload.kind === 'automatic' ? '自动' : '手动';
  $('parse-state').textContent = matched ? `${kind}解析成功` : `${kind}解析拒绝`;
  $('parse-state').className = `pill ${matched ? 'online' : 'offline'}`;
  $('parse-input').textContent = payload.input_text || '（空文本）';
  $('parse-normalized').textContent = payload.normalized_text || '—';
  $('parse-actions').textContent = actions.length
    ? actions.map(parseActionLabel).join(' → ')
    : '—';
  $('parse-error').textContent = error
    ? `${error.code || 'parse_error'} @ byte ${Number(error.byte_offset || 0)}：${error.message || '解析失败'}`
    : '规则完整匹配';
  $('parse-error').className = matched ? 'success' : 'error';
  $('parse-meta').textContent =
    `${kind} · ${payload.execution_mode || 'dry_run'} · ${payload.parser_version || 'parser'} · rev ${payload.config_revision ?? '—'}`;
}

function updateRecordingControls() {
  const active = state.recordingActive === true;
  const pending = state.recordingPending;
  const indicator = $('recording');
  indicator.classList.toggle('active', active);
  indicator.classList.toggle('pending', pending);
  $('recording-label').textContent = pending
    ? (active ? '正在停止…' : '正在启动…')
    : (active ? '录音中' : '未录制');
  const elapsed = $('recording-elapsed');
  elapsed.hidden = !active;
  elapsed.textContent = formatDuration((Date.now() - state.recordingStartedAtMs) / 1000);
  $('recording-session').textContent = state.recordingSession
    ? sessionName(state.recordingSession)
    : '尚未创建录音';
  $('recording-session').title = state.recordingSession || '';
  $('record-start').disabled = pending || active || state.activeMode !== 'live';
  $('record-stop').disabled = pending || !active;
}

function timelineSample(value, mode) {
  const numeric = Number(value || 0);
  const origin = state.timelineOrigin[mode];
  return origin == null ? numeric : Math.max(0, numeric - origin);
}

function normalizeSpan(span, mode) {
  if (!span || typeof span !== 'object') return span;
  return {
    ...span,
    start: timelineSample(span.start, mode),
    end: timelineSample(span.end, mode)
  };
}

function normalizeCandidate(candidate, mode) {
  return {
    ...candidate,
    trigger_sample: timelineSample(candidate.trigger_sample, mode),
    wake_span: normalizeSpan(candidate.wake_span, mode),
    source_spans: (candidate.source_spans || []).map(span => normalizeSpan(span, mode))
  };
}

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
  state.timelineOrigin[mode] = null;
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
    if (!result.ok) {
      const message = result.error || 'unknown error';
      showControlMessage(message, 'error');
      addEvent({type: 'command_error', level: 'error', payload: {message}, seq: '—'});
    }
    return result;
  } catch (error) {
    showControlMessage(String(error), 'error');
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
    if (payload.view_sample != null && state.timelineOrigin[messageMode] == null) {
      state.timelineOrigin[messageMode] =
        Number(payload.sample || 0) - Number(payload.view_sample || 0);
    }
    state.latestSample[messageMode] = Number(payload.sample || 0);
    state.telemetry.push(payload);
    state.vad.push(!!payload.vad);
    while (state.telemetry.length > state.maxPoints) state.telemetry.shift();
    while (state.vad.length > state.maxPoints) state.vad.shift();

    const cutoff = sampleOf(payload) - 20 * SAMPLE_RATE;
    state.wakes = state.wakes.filter(hit => Number(hit.wake_span?.end ?? hit.at ?? 0) >= cutoff);
    state.candidates = state.candidates.filter(item => candidateEndSample(item) >= cutoff);
    $('rms').textContent = Number(payload[state.source]?.rms ?? payload.processed?.rms ?? 0).toFixed(3);
    $('queue-depth').textContent = payload.queue_depth ?? 0;
    $('drops').textContent = payload.telemetry_dropped ?? 0;
    if (payload.aec) updateAec(payload.aec);
    renderCommandStatus();
    syncViewControls();
  } else if (type === 'kws_hit') {
    const messageMode = modeOf(message);
    if (messageMode === state.activeMode) {
      state.wakes.push({
        at: timelineSample(message.timestamp_sample, messageMode),
        ...payload,
        detected_at_sample: timelineSample(payload.detected_at_sample, messageMode),
        wake_span: normalizeSpan(payload.wake_span, messageMode)
      });
      if (messageMode === 'replay' && payload.keyword && !$('replay-parse-wake').value) {
        $('replay-parse-wake').value = String(payload.keyword).replace(/^@/, '');
      }
    }
    addEvent(message);
  } else if (type === 'candidate') {
    const messageMode = modeOf(message);
    if (messageMode === state.activeMode) {
      state.candidates.push(normalizeCandidate(payload, messageMode));
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
    if (final && modeOf(message) === 'replay' && payload.text) {
      $('replay-parse-text').value = payload.text;
    }
    addEvent(message);
  } else if (type === 'parse_debug') {
    if (modeOf(message) === 'replay' && state.activeMode === 'replay') {
      renderParseDebug(payload);
    }
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
    state.recordingPending = false;
    state.recordingSession = payload.session || state.recordingSession;
    state.recordingStartedAtMs = Number(payload.started_at_ms || 0) ||
      state.recordingStartedAtMs || Date.now();
    $('recording').classList.toggle('incomplete', !active && !!payload.incomplete);
    updateRecordingControls();

    if (active) resumeLiveView();
    else if (previous === true && !state.view.frozen) freezeView('录制停止');
    if (previous === true && !active) {
      showControlMessage(
        payload.incomplete ? '录音已停止，但会话被标记为不完整' : '录音已保存，可从回放列表载入',
        payload.incomplete ? 'warn' : 'success');
    }
    addEvent(message);
  } else if (type === 'sessions') {
    const select = $('sessions');
    const current = select.value;
    state.sessions.clear();
    select.innerHTML = '<option value="">选择有效录音</option>';
    for (const session of payload.sessions || []) {
      state.sessions.set(session.path, session);
      const duration = Number(session.duration_seconds || 0).toFixed(1);
      const suffix = session.complete ? '' : ' · 不完整';
      select.add(new Option(`${session.name} · ${duration}s${suffix}`, session.path));
    }
    select.value = current;
    if (!select.value) state.loadedSession = null;
    const excluded = Number(payload.excluded_sessions || 0);
    if (excluded > 0) {
      showControlMessage(`已隐藏 ${excluded} 个空白或未正常完成的录音会话`, 'warn');
    }
  } else if (type === 'runtime_mode') {
    const mode = payload.mode === 'replay' ? 'replay' : 'live';
    setActiveMode(mode, mode !== state.activeMode);
    addEvent(message);
  } else if (type === 'live_dry_run_state') {
    $('dry-run-state').hidden = !payload.enabled;
    addEvent(message);
  } else if (type === 'replay_state') {
    const replayTransition =
      Boolean(payload.playing) !== state.replayPlaying ||
      Boolean(payload.finished) ||
      Boolean(payload.error);
    state.replayPlaying = Boolean(payload.playing);
    if (state.activeMode === 'replay') {
      $('source').textContent = payload.error ? 'REPLAY !' : (payload.playing ? 'REPLAY ▶' : 'REPLAY');
    }
    $('replay-seek').max = Math.max(1, Number(payload.duration_seconds || 0));
    const audioControlsPosition =
      Boolean(state.loadedSession && replayAudio.currentSrc) &&
      Number($('replay-speed').value) !== 0;
    if (!audioControlsPosition) {
      $('replay-seek').value = Number(payload.position_seconds || 0);
    }
    if (payload.error) {
      $('replay-audio-state').textContent = '回放失败';
      $('replay-audio-state').className = 'pill error';
      showControlMessage(payload.error, 'error');
    } else if (payload.finished) {
      $('replay-audio-state').textContent = '播放完成';
      $('replay-audio-state').className = 'pill';
      showControlMessage('录音回放完成', 'success');
    } else if (payload.playing) {
      $('replay-audio-state').textContent = Number(payload.speed) === 0
        ? '仅分析'
        : '播放中';
      $('replay-audio-state').className = 'pill playing';
      if (state.audioFinished && Number(payload.speed) !== 0) {
        $('replay-audio-state').textContent = '试听完成 · 分析中';
      }
    } else if (state.loadedSession) {
      $('replay-audio-state').textContent = '已暂停';
      $('replay-audio-state').className = 'pill';
    }
    if (replayTransition) addEvent(message);
  } else {
    addEvent(message);
  }
}

function updateAec(aec) {
  $('aec-state').textContent = aec.state || aec.status || (aec.enabled === false ? '禁用' : '运行中');
  $('aec-delay').textContent = aec.external_delay_ms == null ? '—' : `${Number(aec.external_delay_ms).toFixed(1)} ms`;
  $('aec-auto-delay').textContent = aec.auto_delay_ms == null ? '—' : `${Number(aec.auto_delay_ms).toFixed(1)} ms`;
  $('aec-delay-confidence').textContent = aec.auto_delay_confidence == null ? '—' : Number(aec.auto_delay_confidence).toFixed(3);
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
  for (const id of ['replay-parse-text', 'replay-parse-wake', 'replay-parse-run']) {
    $(id).disabled = mode !== 'replay';
  }
  if (clear || changed) resetParseDebug();
  updateRecordingControls();
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
  return Number(frame?.view_sample ?? frame?.sample ?? 0);
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
    scrubber.min = 0;
    scrubber.max = 0;
    scrubber.value = 0;
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

function configureReplayTracks(session) {
  const selector = $('replay-track');
  for (const option of selector.options) {
    option.disabled = !session?.streams?.[option.value]?.present;
  }
  const preferred = session?.default_stream || 'processed';
  if (session?.streams?.[preferred]?.present) selector.value = preferred;
  if (selector.selectedOptions[0]?.disabled) {
    const available = Array.from(selector.options).find(option => !option.disabled);
    if (available) selector.value = available.value;
  }
}

function loadReplayAudio(session, preserveTime = false) {
  if (!session) return false;
  state.audioFinished = false;
  configureReplayTracks(session);
  const stream = $('replay-track').value;
  if (!session.streams?.[stream]?.present) {
    showControlMessage('该录音不包含所选音轨', 'error');
    return false;
  }
  const previousTime = preserveTime ? replayAudio.currentTime : 0;
  replayAudio.pause();
  replayAudio.src = `/api/session-audio?token=${encodeURIComponent(token)}` +
    `&session=${encodeURIComponent(session.name)}` +
    `&stream=${encodeURIComponent(stream)}`;
  replayAudio.load();
  if (previousTime > 0) {
    replayAudio.addEventListener('loadedmetadata', () => {
      replayAudio.currentTime = Math.min(previousTime, replayAudio.duration || previousTime);
    }, {once: true});
  }
  $('replay-audio-state').textContent = '正在载入';
  $('replay-audio-state').className = 'pill';
  return true;
}

replayAudio.addEventListener('loadedmetadata', () => {
  const duration = Number(replayAudio.duration || 0);
  if (Number.isFinite(duration) && duration > 0) {
    $('replay-seek').max = duration;
    $('replay-audio-state').textContent = `可播放 · ${duration.toFixed(1)}s`;
    $('replay-audio-state').className = 'pill';
    showControlMessage('录音已载入，点击“播放并分析”即可试听并查看波形', 'success');
  }
});
replayAudio.addEventListener('timeupdate', () => {
  if (!replayAudio.paused && Number($('replay-speed').value) !== 0) {
    $('replay-seek').value = replayAudio.currentTime;
  }
});
replayAudio.addEventListener('ended', () => {
  state.audioFinished = true;
  $('source').textContent = 'REPLAY';
  $('replay-audio-state').textContent = '播放完成';
  $('replay-audio-state').className = 'pill';
  if (state.replayPlaying) {
    $('replay-audio-state').textContent = '试听完成 · 分析中';
    $('replay-audio-state').className = 'pill playing';
  }
});
replayAudio.addEventListener('error', () => {
  $('replay-audio-state').textContent = '音频不可播放';
  $('replay-audio-state').className = 'pill error';
  showControlMessage('浏览器无法读取该录音音轨，请尝试原始麦克风音轨', 'error');
});

$('wave-source').onchange = (event) => state.source = event.target.value;
$('clear-events').onclick = () => $('events').innerHTML = '';
$('record-start').onclick = async () => {
  if (state.recordingPending || state.recordingActive) return;
  state.recordingPending = true;
  updateRecordingControls();
  showControlMessage('正在启动录音…');
  const result = await command('recording.start');
  if (!result.ok) {
    state.recordingPending = false;
    updateRecordingControls();
    return;
  }
  showControlMessage('录音已开始，红色状态灯和计时器会持续显示', 'success');
};
$('record-stop').onclick = async () => {
  if (state.recordingPending || state.recordingActive !== true) return;
  const wasFollowing = !state.view.frozen;
  if (state.recordingActive === true) freezeView('录制停止');
  state.recordingPending = true;
  updateRecordingControls();
  showControlMessage('正在停止并保存录音…');
  const result = await command('recording.stop');
  if (!result.ok) {
    state.recordingPending = false;
    updateRecordingControls();
    if (wasFollowing) resumeLiveView();
  }
};
$('replay-open').onclick = async () => {
  const sessionPath = $('sessions').value;
  const session = state.sessions.get(sessionPath);
  if (!sessionPath || !session) {
    showControlMessage('请先选择有效录音会话', 'error');
    addEvent({type: 'command_error', level: 'error', payload: {message: '请先选择回放会话'}, seq: '—'});
    return;
  }
  const previousMode = state.activeMode;
  const previousSession = state.loadedSession;
  state.loadedSession = session;
  setActiveMode('replay', true);
  const result = await command('replay.open', {session: sessionPath});
  if (!result.ok) {
    state.loadedSession = previousSession;
    setActiveMode(previousMode, true);
    return;
  }
  clearTimelineData();
  resetModeStatus('replay');
  resetParseDebug();
  syncViewControls();
  loadReplayAudio(session);
};
$('replay-play').onclick = async () => {
  if (!state.loadedSession) {
    showControlMessage('请先载入录音会话', 'error');
    return;
  }
  state.audioFinished = false;
  const speed = Number($('replay-speed').value);
  if (speed !== 0) {
    if (replayAudio.ended) replayAudio.currentTime = 0;
    replayAudio.playbackRate = speed;
    try {
      await replayAudio.play();
    } catch (error) {
      showControlMessage(`音频播放失败：${String(error)}`, 'error');
      return;
    }
  } else {
    replayAudio.pause();
    showControlMessage('不限速模式只运行分析，不输出声音', 'warn');
  }
  const result = await command('replay.play');
  if (!result.ok) replayAudio.pause();
};
$('replay-pause').onclick = async () => {
  replayAudio.pause();
  await command('replay.pause');
};
$('replay-speed').onchange = async () => {
  const speed = Number($('replay-speed').value);
  if (speed !== 0) replayAudio.playbackRate = speed;
  else replayAudio.pause();
  await command('replay.speed', {speed});
};
$('replay-seek').onchange = () => {
  const seconds = Number($('replay-seek').value);
  if (replayAudio.src && Number.isFinite(replayAudio.duration)) {
    replayAudio.currentTime = Math.min(seconds, replayAudio.duration);
  }
  clearTimelineData();
  resetModeStatus('replay');
  syncViewControls();
  command('replay.seek', {seconds});
};
$('replay-track').onchange = () => {
  if (state.loadedSession) loadReplayAudio(state.loadedSession, true);
};
$('replay-parse-form').onsubmit = async (event) => {
  event.preventDefault();
  if (state.activeMode !== 'replay') {
    showControlMessage('请先载入录音并进入回放模式', 'error');
    return;
  }
  $('parse-state').textContent = '正在解析';
  $('parse-state').className = 'pill';
  const result = await command('replay.parse_debug', {
    text: $('replay-parse-text').value,
    wake_word: $('replay-parse-wake').value
  });
  if (result.ok && result.parse) renderParseDebug(result.parse);
};
$('config-form').onsubmit = (event) => {
  event.preventDefault();
  command('config.apply', {patch: configPatch(), save: false});
};
$('config-save').onclick = () => command('config.apply', {patch: configPatch(), save: true});
$('config-reset').onclick = () => state.config && loadConfig(state.config);
$('view-live').onclick = async () => {
  if (state.activeMode === 'replay') {
    replayAudio.pause();
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
$('collect-link').href = `/collect.html?token=${encodeURIComponent(token)}`;
connect();
draw();
setInterval(updateRecordingControls, 500);
