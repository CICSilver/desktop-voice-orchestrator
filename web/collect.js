'use strict';

const token = new URLSearchParams(location.search).get('token') || '';
const $ = (id) => document.getElementById(id);

// Each group is recorded as its own session. The command script is the same
// for every group (rotated so no command is always first or last), which
// keeps groups comparable when only the environment changes.
const GROUPS = [
  {id: 'quiet-near', title: '安静 · 近距离（约 0.5 米）', condition: 'quiet', distance_m: 0.5, silence_s: 15,
    intro: '关掉音乐和视频，坐在离麦克风约 0.5 米的位置，用平时说话的音量。'},
  {id: 'quiet-far', title: '安静 · 2 米', condition: 'quiet', distance_m: 2.0, silence_s: 15,
    intro: '保持安静，站或坐在离麦克风约 2 米的位置，用你平时对电脑说话的音量，不要刻意提高嗓门。'},
  {id: 'music-far', title: '播放音乐 · 2 米', condition: 'music', distance_m: 2.0, silence_s: 30,
    intro: '用任意播放器通过音箱播放音乐，音量调到平时习惯的大小，然后在约 2 米处说话。第一条是纯音乐静默段，用来统计误唤醒。'},
  {id: 'music-near', title: '播放音乐 · 近距离（约 0.5 米）', condition: 'music', distance_m: 0.5, silence_s: 20, optional: true,
    intro: '继续播放音乐，回到离麦克风约 0.5 米的位置说话。'}
];

const PAUSE = {type: 'media.pause'};
const PLAY = {type: 'media.play'};
const volume = (delta) => ({type: 'audio.volume.adjust', volume_delta_percent: delta});

const COMMANDS = [
  {key: 'pre-pause', position: 'prefix', say: '{W}，暂停音乐', actions: [PAUSE]},
  {key: 'suf-play', position: 'suffix', say: '播放音乐，{W}', actions: [PLAY]},
  {key: 'pre-volup', position: 'prefix', say: '{W}，增加音量', actions: [volume(5)]},
  {key: 'suf-voldown10', position: 'suffix', say: '降低音量百分之十，{W}', actions: [volume(-10)]},
  {key: 'pre-play', position: 'prefix', say: '{W}，播放音乐', actions: [PLAY]},
  {key: 'suf-pause', position: 'suffix', say: '暂停音乐，{W}', actions: [PAUSE]},
  {key: 'pre-voldown10', position: 'prefix', say: '{W}，降低音量百分之十', actions: [volume(-10)]},
  {key: 'suf-volup', position: 'suffix', say: '增加音量，{W}', actions: [volume(5)]},
  {key: 'pre-multi', position: 'prefix', say: '{W}，暂停音乐，然后降低音量百分之十', actions: [PAUSE, volume(-10)]},
  {key: 'suf-multi', position: 'suffix', say: '播放音乐，再增加音量百分之五，{W}', actions: [PLAY, volume(5)]},
  {key: 'pre-multi2', position: 'prefix', say: '{W}，增加音量，再播放音乐', actions: [volume(5), PLAY]},
  {key: 'suf-multi2', position: 'suffix', say: '降低音量，然后暂停音乐，{W}', actions: [volume(-5), PAUSE]},
  {key: 'pre-open', position: 'prefix', say: '{W}，打开音乐', actions: [PLAY]}
];

// Silence and negatives come before any command, so no activation window can
// be open yet: speech within 6 s of a command is a legitimate follow-up turn
// and would otherwise have to be waited out before every negative.
const NEGATIVES = [
  {key: 'neg-command', kind: 'negative', say: '暂停音乐', note: '这条不要说唤醒词'},
  {key: 'neg-chat', kind: 'negative', say: '今天晚上吃点什么好呢', note: '这条不要说唤醒词'}
];

// Pause before each prompt appears. Together with reading time it keeps
// consecutive takes in separate VAD segments (endpoint silence is 0.9 s).
const GAP_S = 1.5;
const MIN_TAKE_MS = 600;

const state = {
  wake: '',
  previousDryRun: null,
  collectionId: '',
  speaker: '',
  plan: [],
  groupIndex: 0,
  group: null,
  takes: [],
  takeIndex: 0,
  attempt: 1,
  labels: [],
  session: '',
  startedAt: 0,
  phase: 'setup',
  timer: null,
  takeStart: 0,
  saved: [],
  recording: false,
  mic: -120,
  loop: -120
};

function showMessage(text = '', level = '') {
  $('message').textContent = text;
  $('message').className = `collect-message ${level}`.trim();
}

async function command(action, payload = {}) {
  try {
    const response = await fetch(`/api/command?token=${encodeURIComponent(token)}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action, ...payload})
    });
    return await response.json();
  } catch (error) {
    return {ok: false, error: String(error)};
  }
}

function show(section) {
  for (const id of ['setup', 'group-intro', 'take', 'group-done', 'finished']) {
    $(id).hidden = id !== section;
  }
}

function sessionName(path = '') {
  return String(path).split(/[\\/]/).filter(Boolean).at(-1) || '';
}

function toDb(rms) {
  return 20 * Math.log10(Math.max(Number(rms) || 0, 1e-6));
}

function setMeter(barId, labelId, db) {
  const ratio = Math.min(1, Math.max(0, (db + 60) / 60));
  $(barId).style.width = `${(ratio * 100).toFixed(1)}%`;
  $(barId).classList.toggle('hot', db > -6);
  $(labelId).textContent = db <= -119 ? '— dBFS' : `${db.toFixed(0)} dBFS`;
}

function setRecording(active) {
  state.recording = active;
  $('recording').className = `record-indicator${active ? ' active' : ''}`;
  $('recording-label').textContent = active ? `录制中 · ${sessionName(state.session)}` : '未录制';
}

function setDryRun(enabled) {
  $('dry-run').textContent = enabled ? '系统动作：已停用' : '系统动作：会执行';
  $('dry-run').className = `pill ${enabled ? 'dry' : 'live-actions'}`;
}

function updateConditionCheck() {
  if (!state.group || state.phase !== 'intro') return;
  const playing = state.loop > -45;
  const check = $('intro-check');
  if (state.group.condition === 'music') {
    check.textContent = playing ? '✓ 检测到音箱正在输出声音' : '⚠ 没有检测到音箱输出，请先开始播放音乐';
    check.className = `condition-check ${playing ? 'ok' : 'warn'}`;
  } else {
    check.textContent = playing ? '⚠ 音箱仍有声音输出，请先停止播放' : '✓ 音箱没有输出';
    check.className = `condition-check ${playing ? 'warn' : 'ok'}`;
  }
}

function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${location.host}/ws?token=${encodeURIComponent(token)}`);
  ws.onopen = () => {
    $('connection').textContent = '在线';
    $('connection').className = 'pill online';
  };
  ws.onclose = () => {
    $('connection').textContent = '离线';
    $('connection').className = 'pill offline';
    setTimeout(connect, 1200);
  };
  ws.onmessage = ({data}) => {
    const {type, payload = {}} = JSON.parse(data);
    if (type === 'telemetry') {
      state.mic = toDb(payload.microphone?.rms);
      state.loop = toDb(payload.loopback?.rms);
      setMeter('intro-mic-bar', 'intro-mic-db', state.mic);
      setMeter('intro-loop-bar', 'intro-loop-db', state.loop);
      setMeter('take-mic-bar', 'take-mic-db', state.mic);
      updateConditionCheck();
    } else if (type === 'live_dry_run_state') {
      setDryRun(Boolean(payload.enabled));
    } else if (type === 'recording_state' && !payload.active && state.recording &&
               ['countdown', 'speak'].includes(state.phase)) {
      // Another page stopped the recording underneath us.
      clearTimer();
      setRecording(false);
      state.phase = 'intro';
      showIntro();
      showMessage('录音被其他页面停止了，本组没有保存标注，请重新录这一组。', 'error');
    }
  };
}

function renderGroups() {
  $('groups').replaceChildren(...GROUPS.map((group) => {
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.value = group.id;
    input.checked = !group.optional;
    const title = document.createElement('strong');
    title.textContent = group.title;
    const hint = document.createElement('span');
    const minutes = Math.ceil(((COMMANDS.length + NEGATIVES.length) * (GAP_S + 4) + group.silence_s + 10) / 60);
    hint.textContent = `约 ${minutes} 分钟${group.optional ? ' · 可选' : ''}`;
    label.append(input, title, hint);
    return label;
  }));
}

function buildTakes(group, groupIndex) {
  const rotation = (groupIndex * 5) % COMMANDS.length;
  const commands = COMMANDS.slice(rotation).concat(COMMANDS.slice(0, rotation));
  const script = [
    {key: 'silence', kind: 'silence', position: 'none', say: '', actions: [], auto_s: group.silence_s,
      note: group.condition === 'music' ? '让音乐继续播放，不要说话' : '不要说话'},
    ...NEGATIVES.map((item) => ({...item, position: 'none', actions: []})),
    ...commands.map((item) => ({...item, kind: 'command'}))
  ];
  return script.map((item, index) => {
    const text = item.say.replaceAll('{W}', state.wake);
    return {
      id: `${group.id}-${String(index + 1).padStart(2, '0')}-${item.key}`,
      kind: item.kind,
      wake_position: item.position,
      prompt: item.kind === 'silence' ? `保持安静 ${item.auto_s} 秒` : text,
      text,
      expected_actions: item.actions,
      auto_s: item.auto_s ?? 0,
      note: item.note || ''
    };
  });
}

function clearTimer() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
}

function showIntro() {
  state.phase = 'intro';
  state.group = state.plan[state.groupIndex];
  $('intro-progress').textContent = `GROUP ${state.groupIndex + 1} / ${state.plan.length}`;
  $('intro-title').textContent = state.group.title;
  $('intro-text').textContent = state.group.intro;
  $('start-group').disabled = false;
  show('group-intro');
  updateConditionCheck();
}

async function startGroup() {
  $('start-group').disabled = true;
  showMessage('正在开始录音…');
  let result = await command('recording.start');
  if (!result.ok && String(result.error || '').includes('live mode')) {
    await command('live.resume');
    result = await command('recording.start');
  }
  if (!result.ok) {
    $('start-group').disabled = false;
    showMessage(`无法开始录音：${result.error || '未知错误'}`, 'error');
    return;
  }
  showMessage('');
  state.session = result.session;
  state.startedAt = Number(result.started_at_ms) || Date.now();
  // Rotate by the group's fixed position so each group's order is the same
  // whichever subset of groups is selected.
  state.takes = buildTakes(state.group, GROUPS.indexOf(state.group));
  state.takeIndex = 0;
  state.attempt = 1;
  state.labels = [];
  setRecording(true);
  show('take');
  beginTake();
}

function beginTake() {
  clearTimer();
  state.phase = 'countdown';
  const take = state.takes[state.takeIndex];
  $('take-progress').textContent = `${state.takeIndex + 1} / ${state.takes.length}`;
  $('take-group').textContent = state.group.title;
  $('take-kind').textContent = take.kind === 'command'
    ? (take.wake_position === 'prefix' ? '唤醒词在前' : '唤醒词在后')
    : take.kind === 'negative' ? '不带唤醒词' : '静默';
  $('take-progress-bar').style.width = `${(state.takeIndex / state.takes.length * 100).toFixed(1)}%`;
  $('take-prompt').textContent = '准备…';
  $('take-prompt').className = 'prompt waiting';
  $('take-note').textContent = state.attempt > 1 ? `第 ${state.attempt} 次录这一条` : '';
  for (const id of ['next-take', 'redo-take', 'skip-take']) $(id).disabled = true;
  const until = Date.now() + GAP_S * 1000;
  const tick = () => {
    const remaining = Math.max(0, until - Date.now());
    $('take-countdown').textContent = remaining > 0 ? `${(remaining / 1000).toFixed(1)} 秒后开始` : '';
    if (remaining <= 0) startSpeaking();
  };
  tick();
  state.timer = setInterval(tick, 100);
}

function startSpeaking() {
  clearTimer();
  const take = state.takes[state.takeIndex];
  state.phase = 'speak';
  state.takeStart = Date.now();
  $('take-countdown').textContent = take.kind === 'silence' ? '' : '请说：';
  $('take-prompt').textContent = take.prompt;
  $('take-prompt').className = `prompt${take.kind === 'silence' ? ' silence' : ''}`;
  $('take-note').textContent = take.note || (state.attempt > 1 ? `第 ${state.attempt} 次录这一条` : '');
  for (const id of ['next-take', 'redo-take', 'skip-take']) $(id).disabled = false;
  if (take.kind === 'silence') {
    $('next-take').disabled = true;
    const until = state.takeStart + take.auto_s * 1000;
    state.timer = setInterval(() => {
      const remaining = Math.max(0, until - Date.now());
      $('take-countdown').textContent = `剩余 ${Math.ceil(remaining / 1000)} 秒`;
      if (remaining <= 0) finishTake('ok');
    }, 200);
  }
}

function recordTake(status) {
  const take = state.takes[state.takeIndex];
  state.labels.push({
    id: take.id,
    kind: take.kind,
    status,
    attempt: state.attempt,
    prompt: take.prompt,
    text: take.text,
    wake_position: take.wake_position,
    expected_actions: take.expected_actions,
    start_ms: state.takeStart - state.startedAt,
    end_ms: Date.now() - state.startedAt
  });
}

function finishTake(status) {
  if (state.phase !== 'speak') return;
  if (status === 'ok' && Date.now() - state.takeStart < MIN_TAKE_MS) return;
  clearTimer();
  recordTake(status);
  if (status === 'discarded') {
    state.attempt += 1;
    beginTake();
    return;
  }
  state.takeIndex += 1;
  state.attempt = 1;
  if (state.takeIndex >= state.takes.length) {
    endGroup();
    return;
  }
  beginTake();
}

function labelsDocument(complete) {
  return {
    schema: 'dvo-eval-labels',
    schema_version: 1,
    collection_id: state.collectionId,
    group: {
      id: state.group.id,
      title: state.group.title,
      condition: state.group.condition,
      distance_m: state.group.distance_m
    },
    speaker: state.speaker,
    wake_word: state.wake,
    created_at: new Date().toISOString(),
    recording_started_at_ms: state.startedAt,
    clock: 'take times are milliseconds since the recording started',
    complete,
    takes: state.labels
  };
}

async function endGroup() {
  clearTimer();
  state.phase = 'saving';
  $('take-prompt').textContent = '正在保存…';
  $('take-prompt').className = 'prompt waiting';
  $('take-countdown').textContent = '';
  $('take-progress-bar').style.width = '100%';
  for (const id of ['next-take', 'redo-take', 'skip-take', 'abort-group']) $(id).disabled = true;
  const stopped = await command('recording.stop');
  setRecording(false);
  let saved = {ok: false, error: stopped.error};
  if (stopped.ok) {
    for (let attempt = 0; attempt < 3 && !saved.ok; attempt += 1) {
      saved = await command('recording.save_labels', {session: state.session, labels: labelsDocument(true)});
      if (!saved.ok) await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  $('abort-group').disabled = false;
  const name = sessionName(state.session);
  const scored = state.labels.filter((take) => take.status === 'ok').length;
  if (saved.ok) {
    state.saved.push({title: state.group.title, session: name, takes: scored});
    $('done-title').textContent = `${state.group.title} 已保存`;
    $('done-text').textContent = `会话 ${name}，有效 ${scored} 条（重录/跳过的不计）。`;
    showMessage('');
  } else {
    $('done-title').textContent = `${state.group.title} 保存失败`;
    $('done-text').textContent = `录音在 ${name}，但标注没有写进去：${saved.error || '未知错误'}。这组评估时会被跳过。`;
    showMessage('', '');
  }
  state.phase = 'done';
  const last = state.groupIndex + 1 >= state.plan.length;
  $('next-group').textContent = last ? '完成' : '下一组';
  show('group-done');
}

async function abortGroup() {
  if (!['countdown', 'speak'].includes(state.phase)) return;
  const resumePhase = state.phase;
  state.phase = 'confirming';
  if (!confirm('放弃本组？已经录下的音频会保留在会话目录里，但不会写标注，评估时会跳过。')) {
    state.phase = resumePhase;
    return;
  }
  state.phase = 'aborting';
  clearTimer();
  await command('recording.stop');
  setRecording(false);
  showMessage(`已放弃 ${state.group.title}，可以重新录这一组。`);
  showIntro();
}

async function restoreDryRun() {
  if (state.previousDryRun === null) return;
  const previous = state.previousDryRun;
  state.previousDryRun = null;
  await command('commands.live_dry_run', {enabled: previous});
}

async function finish() {
  clearTimer();
  if (state.recording) {
    await command('recording.stop');
    setRecording(false);
  }
  await restoreDryRun();
  state.phase = 'finished';
  $('saved-list').replaceChildren(...(state.saved.length ? state.saved : [{title: '没有保存任何组'}]).map((item) => {
    const li = document.createElement('li');
    li.textContent = item.session ? `${item.title}：` : item.title;
    if (item.session) {
      const code = document.createElement('code');
      code.textContent = item.session;
      li.append(code, `（${item.takes} 条）`);
    }
    return li;
  }));
  $('evaluate-command').textContent =
    '.\\build\\windows-x64\\Release\\voice_frontend.exe evaluate data\\sessions --out=data\\evaluation-report.json';
  show('finished');
}

async function startCollection() {
  const selected = new Set([...document.querySelectorAll('#groups input:checked')].map((input) => input.value));
  state.plan = GROUPS.filter((group) => selected.has(group.id));
  if (!state.plan.length) {
    $('setup-message').textContent = '至少选择一组。';
    return;
  }
  state.speaker = $('speaker').value.trim() || '未命名';
  const now = new Date();
  const pad = (value) => String(value).padStart(2, '0');
  state.collectionId = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-` +
    `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  state.saved = [];
  state.groupIndex = 0;
  if ($('use-dry-run').checked) {
    const info = await command('evaluation.info');
    const result = await command('commands.live_dry_run', {enabled: true});
    if (!result.ok) {
      $('setup-message').textContent = `无法停用系统动作：${result.error || '未知错误'}`;
      return;
    }
    state.previousDryRun = Boolean(info.live_dry_run);
  }
  showIntro();
}

function handleKey(event) {
  if (event.target instanceof HTMLInputElement) return;
  const key = event.key;
  if (state.phase === 'intro' && key === 'Enter') {
    event.preventDefault();
    if (!$('start-group').disabled) startGroup();
  } else if (state.phase === 'speak') {
    if (key === ' ' || key === 'Enter') { event.preventDefault(); finishTake('ok'); }
    else if (key === 'r' || key === 'R') { event.preventDefault(); finishTake('discarded'); }
    else if (key === 's' || key === 'S') { event.preventDefault(); finishTake('skipped'); }
    else if (key === 'Escape') { event.preventDefault(); abortGroup(); }
  } else if (state.phase === 'countdown') {
    if (key === ' ') event.preventDefault();
    if (key === 'Escape') { event.preventDefault(); abortGroup(); }
  } else if (state.phase === 'done' && key === 'Enter') {
    event.preventDefault();
    $('next-group').click();
  }
}

async function init() {
  renderGroups();
  connect();
  const info = await command('evaluation.info');
  if (!info.ok) {
    $('wake-word').textContent = '无法读取';
    $('setup-message').textContent = `无法连接运行时：${info.error || '未知错误'}。请从调试台的链接打开本页。`;
    return;
  }
  state.wake = (info.wake_words && info.wake_words[0]) || '小助手';
  $('wake-word').textContent = (info.wake_words && info.wake_words.length)
    ? info.wake_words.join(' / ')
    : `${state.wake}（未在关键词文件里找到，使用默认）`;
  setDryRun(Boolean(info.live_dry_run));
  $('start-collection').disabled = false;
}

$('start-collection').addEventListener('click', startCollection);
$('start-group').addEventListener('click', startGroup);
$('finish-early').addEventListener('click', finish);
$('next-take').addEventListener('click', () => finishTake('ok'));
$('redo-take').addEventListener('click', () => finishTake('discarded'));
$('skip-take').addEventListener('click', () => finishTake('skipped'));
$('abort-group').addEventListener('click', abortGroup);
$('next-group').addEventListener('click', () => {
  state.groupIndex += 1;
  if (state.groupIndex >= state.plan.length) finish();
  else showIntro();
});
$('finish-now').addEventListener('click', finish);
$('restart').addEventListener('click', () => {
  state.phase = 'setup';
  show('setup');
});
document.addEventListener('keydown', handleKey);
window.addEventListener('beforeunload', (event) => {
  if (state.recording) {
    event.preventDefault();
    event.returnValue = '';
  }
});
window.addEventListener('pagehide', () => {
  if (state.previousDryRun === null) return;
  fetch(`/api/command?token=${encodeURIComponent(token)}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: 'commands.live_dry_run', enabled: state.previousDryRun}),
    keepalive: true
  });
});

init();
