'use strict';

const token = new URLSearchParams(location.search).get('token') || '';
const $ = (id) => document.getElementById(id);

// Each group is recorded as its own session. The command script is the same
// for every command group, rotated by the group's fixed `rotation` so no
// command is always first or last and each group's order never changes
// between batches.
const GROUPS = [
  {id: 'quiet-near', title: '安静 · 近距离（约 0.5 米）', condition: 'quiet', distance_m: 0.5, silence_s: 15,
    rotation: 0, intro: '关掉音乐和视频，坐在离麦克风约 0.5 米的位置，用平时说话的音量。'},
  {id: 'music-near', title: '播放音乐 · 近距离（约 0.5 米）', condition: 'music', distance_m: 0.5, silence_s: 30,
    rotation: 3, intro: '用任意播放器通过音箱播放音乐，音量调到平时习惯的大小，坐在离麦克风约 0.5 米的位置说话。第一条是纯音乐静默段，用来统计误唤醒。'}
];

const BACKGROUND_GROUP = {id: 'background', title: '日常背景', condition: 'background', distance_m: 0};

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

// Personal training data: randomized prompts covering every command phrase,
// spoken amounts, both wake positions, two-command chains and the two-step
// flow (the wake word alone, then a command without it). These sessions are
// labelled purpose "train": evaluate skips them, so the evaluation batches
// stay unseen by training.
const TRAINING_GROUPS = {
  quiet: {id: 'train-quiet', title: '训练数据 · 安静', condition: 'quiet', distance_m: 0.5,
    script: 'train', purpose: 'train',
    intro: '关掉音乐和视频，在平时用电脑的位置说话。句子是随机生成的，照着念；语气和快慢自然变化就好，有提示时按提示说。'},
  music: {id: 'train-music', title: '训练数据 · 播放音乐', condition: 'music', distance_m: 0.5,
    script: 'train', purpose: 'train',
    intro: '用音箱播放你平时听的音乐（带人声的歌也要有），音量调到平时习惯的大小，照着念。'}
};
// Reading ordinary sentences: the speaker's voice with exact transcripts but
// no wake word and no command, so fine-tuning does not learn that everything
// this speaker says is a wake-up or a command.
const READING_GROUPS = {
  quiet: {id: 'read-quiet', title: '朗读 · 安静', condition: 'quiet', distance_m: 0.5,
    script: 'read', purpose: 'train',
    intro: '关掉音乐和视频，照着屏幕上的句子念，用平时说话的语气就好。念错或卡住了按 R 重录；生僻的人名地名念不顺就按 S 跳过。'},
  music: {id: 'read-music', title: '朗读 · 播放音乐', condition: 'music', distance_m: 0.5,
    script: 'read', purpose: 'train',
    intro: '用音箱播放你平时听的音乐（带人声的歌也要有），音量调到平时习惯的大小，照着句子念。念错按 R 重录，念不顺按 S 跳过。'}
};
// Share of colloquial sentences (some use command words without being commands).
const COLLOQUIAL_SHARE = 0.15;
const READ_USED_KEY = 'dvo.collect.readSentences';
let readingPool = null;

const TRAIN_PHRASES = [
  {say: '播放音乐', action: 'play'}, {say: '打开音乐', action: 'play'},
  {say: '暂停音乐', action: 'pause'},
  {say: '增加音量', action: 'up', sign: 1}, {say: '降低音量', action: 'down', sign: -1}
];
const TRAIN_CONNECTORS = ['，然后', '，再', '，然后再', '，接着'];
// Common amounts weighted up; every value the parser accepts can appear.
const TRAIN_AMOUNTS = [5, 5, 10, 10, 10, 20, 15, 3, 8, 2, 1, 12, 6, 18];
const TRAIN_STYLES = ['说快一点', '声音轻一点', '稍微大声一点', '像随口一说', '慢一点说'];
const DIGITS = '零一二三四五六七八九';

function chineseNumber(value) {
  if (value < 10) return DIGITS[value];
  if (value === 10) return '十';
  return value < 20 ? `十${DIGITS[value - 10]}` : '二十';
}

// Deterministic per collection, so a batch's prompts can be regenerated.
function seededRandom(seed) {
  let h = 1779033703 ^ seed.length;
  for (const ch of seed) {
    h = Math.imul(h ^ ch.charCodeAt(0), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return () => {
    h = (h + 0x6D2B79F5) | 0;
    let t = Math.imul(h ^ (h >>> 15), 1 | h);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function trainingPhrase(random, previousAction) {
  const pick = (list) => list[Math.floor(random() * list.length)];
  let phrase;
  do phrase = pick(TRAIN_PHRASES);
  while (phrase.action === previousAction);
  if (!phrase.sign) {
    return {say: phrase.say, action: phrase.action, actions: [phrase.action === 'play' ? PLAY : PAUSE]};
  }
  if (random() < 0.4) return {say: phrase.say, action: phrase.action, actions: [volume(5 * phrase.sign)]};
  const amount = pick(TRAIN_AMOUNTS);
  return {say: `${phrase.say}百分之${chineseNumber(amount)}`, action: phrase.action,
    actions: [volume(amount * phrase.sign)]};
}

function buildTrainingScript(group) {
  const random = seededRandom(`${state.collectionId}-${group.id}`);
  const pick = (list) => list[Math.floor(random() * list.length)];
  // Exact proportions, shuffled: 10% wake word alone, 15% without it.
  const wakeCount = Math.round(group.count * 0.1);
  const bareCount = Math.round(group.count * 0.15);
  const kinds = Array.from({length: group.count}, (_, index) =>
    index < wakeCount ? 'wake' : index < wakeCount + bareCount ? 'bare' : 'command');
  for (let index = kinds.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    [kinds[index], kinds[other]] = [kinds[other], kinds[index]];
  }
  const script = [];
  for (const kind of kinds) {
    let item;
    if (kind === 'wake') {
      item = {key: 'wake', kind: 'wake', position: 'none', say: state.wake, actions: []};
    } else if (kind === 'bare') {
      const phrase = trainingPhrase(random);
      item = {key: 'bare', kind: 'bare_command', position: 'none', say: phrase.say,
        actions: phrase.actions, note: '这条不说唤醒词'};
    } else {
      const parts = [trainingPhrase(random)];
      if (random() < 0.3) parts.push(trainingPhrase(random, parts[0].action));
      const body = parts.map((part, i) => (i ? pick(TRAIN_CONNECTORS) : '') + part.say).join('');
      const prefix = random() < 0.5;
      item = {key: prefix ? 'prefix' : 'suffix', kind: 'command', position: prefix ? 'prefix' : 'suffix',
        say: prefix ? `${state.wake}，${body}` : `${body}，${state.wake}`,
        actions: parts.flatMap((part) => part.actions)};
    }
    if (!item.note && random() < 0.3) item.note = pick(TRAIN_STYLES);
    script.push(item);
  }
  return script;
}

function usedSentences() {
  try {
    return new Set(JSON.parse(localStorage.getItem(READ_USED_KEY) || '[]'));
  } catch {
    return new Set();
  }
}

function rememberSentences(texts) {
  try {
    const used = usedSentences();
    for (const text of texts) used.add(text);
    localStorage.setItem(READ_USED_KEY, JSON.stringify([...used]));
  } catch {
    // Only a convenience: without storage a sentence may simply come up again.
  }
}

function buildReadingScript(group) {
  const random = seededRandom(`${state.collectionId}-${group.id}`);
  const used = usedSentences();
  const fresh = (list) => {
    const unused = list.filter((text) => !used.has(text));
    return unused.length ? unused : list;
  };
  const colloquial = fresh(readingPool.colloquial);
  const general = fresh(readingPool.aishell);
  const picked = new Set();
  const script = [];
  while (script.length < group.count && picked.size < colloquial.length + general.length) {
    const list = random() < COLLOQUIAL_SHARE ? colloquial : general;
    const text = list[Math.floor(random() * list.length)];
    if (picked.has(text)) continue;
    picked.add(text);
    script.push({key: 'read', kind: 'read', position: 'none', say: text, actions: []});
  }
  return script;
}

const KIND_LABELS = {negative: '不带唤醒词', bare_command: '不带唤醒词', wake: '只说唤醒词', silence: '静默',
  read: '朗读'};

// Pause before each prompt appears. Together with reading time it keeps
// consecutive takes in separate VAD segments (endpoint silence is 0.9 s).
const GAP_S = 1.5;
const MIN_TAKE_MS = 600;
// Keep recording briefly after the last take: its command is only closed
// about a second after speech ends (VAD silence + endpoint), and replay has
// no audio beyond the end of the recording.
const TAIL_MS = 2500;

// Background recording is split into sessions of this length, so a closed
// tab or a crash loses at most one segment and evaluate never has to hold
// hours of audio in memory at once.
const BACKGROUND_SEGMENT_MS = 15 * 60 * 1000;
// "Exclude" covers what was just said and the follow-up window it may have
// opened (6 s idle after the utterance is processed).
const EXCLUDE_BEFORE_MS = 30000;
const EXCLUDE_AFTER_MS = 10000;

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
  background: null,
  trainingOnly: false,
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
  for (const id of ['setup', 'group-intro', 'take', 'group-done', 'background', 'finished']) {
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
      setMeter('bg-mic-bar', 'bg-mic-db', state.mic);
      setMeter('bg-loop-bar', 'bg-loop-db', state.loop);
      updateConditionCheck();
    } else if (type === 'live_dry_run_state') {
      setDryRun(Boolean(payload.enabled));
    } else if (type === 'recording_state' && !payload.active && state.recording &&
               state.phase === 'background' && state.background && !state.background.busy &&
               sessionName(payload.session) === sessionName(state.session)) {
      stopBackground(true);
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

function buildTakes(group) {
  const rotation = ((group.rotation || 0) * 5) % COMMANDS.length;
  const commands = COMMANDS.slice(rotation).concat(COMMANDS.slice(0, rotation));
  const script = group.script === 'train' ? buildTrainingScript(group)
    : group.script === 'read' ? buildReadingScript(group) : [
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

async function beginRecording() {
  let result = await command('recording.start');
  if (!result.ok && String(result.error || '').includes('live mode')) {
    await command('live.resume');
    result = await command('recording.start');
  }
  if (result.ok) {
    state.session = result.session;
    state.startedAt = Number(result.started_at_ms) || Date.now();
    setRecording(true);
  }
  return result;
}

async function saveLabels() {
  let saved = {ok: false};
  for (let attempt = 0; attempt < 3 && !saved.ok; attempt += 1) {
    saved = await command('recording.save_labels', {session: state.session, labels: labelsDocument(true)});
    if (!saved.ok) await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return saved;
}

async function startGroup() {
  $('start-group').disabled = true;
  showMessage('正在开始录音…');
  const result = await beginRecording();
  if (!result.ok) {
    $('start-group').disabled = false;
    showMessage(`无法开始录音：${result.error || '未知错误'}`, 'error');
    return;
  }
  showMessage('');
  state.takes = buildTakes(state.group);
  state.takeIndex = 0;
  state.attempt = 1;
  state.labels = [];
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
    : KIND_LABELS[take.kind] || '';
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
    purpose: state.group.purpose || 'evaluate',
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
  await new Promise((resolve) => setTimeout(resolve, TAIL_MS));
  const stopped = await command('recording.stop');
  setRecording(false);
  const saved = stopped.ok ? await saveLabels() : {ok: false, error: stopped.error};
  $('abort-group').disabled = false;
  const name = sessionName(state.session);
  const scored = state.labels.filter((take) => take.status === 'ok').length;
  if (saved.ok) {
    if (state.group.script === 'read') {
      rememberSentences(state.labels.filter((take) => take.status === 'ok').map((take) => take.text));
    }
    state.saved.push({title: state.group.title, session: name, detail: `${scored} 条`});
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
      li.append(code, `（${item.detail}）`);
    }
    return li;
  }));
  // --since limits the report to sessions recorded in this run of the page.
  $('evaluate-command').textContent =
    '.\\build\\windows-x64\\Release\\voice_frontend.exe evaluate data\\sessions ' +
    `--since=${state.collectionId} --out=data\\evaluation-${state.collectionId}.json`;
  $('finished-lead').hidden = state.trainingOnly;
  $('evaluate-command').hidden = state.trainingOnly;
  $('finished-training').hidden = !state.trainingOnly;
  show('finished');
}

async function prepareCollection() {
  $('setup-message').textContent = '';
  state.speaker = $('speaker').value.trim() || '未命名';
  const now = new Date();
  const pad = (value) => String(value).padStart(2, '0');
  state.collectionId = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-` +
    `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  state.saved = [];
  if ($('use-dry-run').checked) {
    const info = await command('evaluation.info');
    const result = await command('commands.live_dry_run', {enabled: true});
    if (!result.ok) {
      $('setup-message').textContent = `无法停用系统动作：${result.error || '未知错误'}`;
      return false;
    }
    state.previousDryRun = Boolean(info.live_dry_run);
  }
  return true;
}

async function startCollection() {
  const selected = new Set([...document.querySelectorAll('#groups input:checked')].map((input) => input.value));
  state.plan = GROUPS.filter((group) => selected.has(group.id));
  if (!state.plan.length) {
    $('setup-message').textContent = '至少选择一组。';
    return;
  }
  if (!await prepareCollection()) return;
  state.trainingOnly = false;
  state.groupIndex = 0;
  showIntro();
}

async function startTraining() {
  const count = Math.round(Number($('training-count').value));
  if (!(count >= 10 && count <= 200)) {
    $('setup-message').textContent = '训练数据每轮 10 到 200 条。';
    return;
  }
  const reading = $('training-content').value === 'read';
  const group = (reading ? READING_GROUPS : TRAINING_GROUPS)[$('training-condition').value];
  if (reading && !readingPool) {
    try {
      const response = await fetch('/reading-sentences.json');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      readingPool = await response.json();
    } catch (error) {
      $('setup-message').textContent = `无法读取朗读句子：${error}`;
      return;
    }
  }
  if (!await prepareCollection()) return;
  state.trainingOnly = true;
  state.plan = [{...group, count}];
  state.groupIndex = 0;
  showIntro();
}

// ------------------------------------------------------------ background --

function formatDuration(ms) {
  const total = Math.floor(Math.max(0, ms) / 1000);
  const pad = (value) => String(value).padStart(2, '0');
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor(total / 60) % 60;
  return hours ? `${hours}:${pad(minutes)}:${pad(total % 60)}` : `${minutes}:${pad(total % 60)}`;
}

// The current segment is one background take, except where the speaker
// excluded a moment: that becomes a discarded take, so evaluate ignores any
// wake or command inside it.
function backgroundTakes(endAt) {
  const start = state.startedAt;
  const clip = (value) => Math.min(Math.max(value, start), endAt) - start;
  const spans = state.background.exclusions
    .map(({from, to}) => [clip(from), clip(to)])
    .filter(([from, to]) => to > from)
    .sort((a, b) => a[0] - b[0]);
  const takes = [];
  const push = (status, from, to) => {
    if (to <= from) return;
    takes.push({id: `background-${String(takes.length + 1).padStart(2, '0')}`, kind: 'background',
      status, attempt: 1, prompt: '日常背景', text: '', wake_position: 'none', expected_actions: [],
      start_ms: from, end_ms: to});
  };
  let cursor = 0;
  for (const [from, to] of spans) {
    push('ok', cursor, from);
    push('discarded', Math.max(cursor, from), to);
    cursor = Math.max(cursor, to);
  }
  push('ok', cursor, endAt - start);
  return takes;
}

function renderBackground() {
  const background = state.background;
  if (!background) return 0;
  const elapsed = background.recordedMs + (state.recording ? Date.now() - state.startedAt : 0);
  $('bg-elapsed').textContent = formatDuration(elapsed);
  $('bg-saved').textContent = `${background.saved} 段`;
  $('bg-excluded').textContent = `${background.exclusions.length} 处`;
  $('bg-progress-bar').style.width = `${Math.min(100, elapsed / background.targetMs * 100).toFixed(1)}%`;
  return elapsed;
}

async function closeBackgroundSegment(alreadyStopped) {
  const background = state.background;
  const endAt = Date.now();
  const stopped = alreadyStopped ? {ok: true} : await command('recording.stop');
  setRecording(false);
  background.recordedMs += endAt - state.startedAt;
  background.segment += 1;
  state.labels = backgroundTakes(endAt);
  const saved = stopped.ok ? await saveLabels() : {ok: false, error: stopped.error};
  const name = sessionName(state.session);
  if (saved.ok) {
    background.saved += 1;
    state.saved.push({title: `日常背景第 ${background.segment} 段`, session: name,
      detail: `${((endAt - state.startedAt) / 60000).toFixed(1)} 分钟`});
  } else {
    showMessage(`第 ${background.segment} 段的标注没有保存（${saved.error || '未知错误'}），录音在 ${name}。`, 'error');
  }
  renderBackground();
}

async function startBackground() {
  const minutes = Number($('background-minutes').value);
  if (!(minutes >= 5)) {
    $('setup-message').textContent = '背景录音至少 5 分钟。';
    return;
  }
  $('start-background').disabled = true;
  if (!await prepareCollection()) {
    $('start-background').disabled = false;
    return;
  }
  state.group = BACKGROUND_GROUP;
  state.trainingOnly = false;
  state.background = {targetMs: minutes * 60000, recordedMs: 0, segment: 0, saved: 0,
    exclusions: [], busy: true};
  const result = await beginRecording();
  $('start-background').disabled = false;
  if (!result.ok) {
    state.background = null;
    await restoreDryRun();
    $('setup-message').textContent = `无法开始录音：${result.error || '未知错误'}`;
    return;
  }
  state.background.busy = false;
  state.phase = 'background';
  $('bg-target').textContent = `目标 ${minutes} 分钟`;
  $('bg-stop').disabled = false;
  showMessage('');
  renderBackground();
  show('background');
  clearTimer();
  state.timer = setInterval(backgroundTick, 1000);
}

async function backgroundTick() {
  const background = state.background;
  if (!background || background.busy) return;
  if (renderBackground() >= background.targetMs) {
    await stopBackground(false);
    return;
  }
  if (Date.now() - state.startedAt < BACKGROUND_SEGMENT_MS) return;
  background.busy = true;
  await closeBackgroundSegment(false);
  const result = await beginRecording();
  background.busy = false;
  if (!result.ok) {
    showMessage(`无法开始下一段录音：${result.error || '未知错误'}。已录的段都已保存。`, 'error');
    background.busy = true;
    await endBackground();
  }
}

function excludeRecent() {
  if (state.phase !== 'background' || !state.background) return;
  const now = Date.now();
  state.background.exclusions.push({from: now - EXCLUDE_BEFORE_MS, to: now + EXCLUDE_AFTER_MS});
  renderBackground();
  showMessage('已排除：前 30 秒到之后 10 秒不计入统计。', 'success');
}

async function stopBackground(alreadyStopped) {
  const background = state.background;
  if (!background || background.busy) return;
  background.busy = true;
  clearTimer();
  $('bg-stop').disabled = true;
  if (state.recording) await closeBackgroundSegment(alreadyStopped);
  if (alreadyStopped) showMessage('录音被其他页面停止了，已保存到停止时为止的部分。', 'error');
  await endBackground();
}

async function endBackground() {
  clearTimer();
  state.background = null;
  await finish();
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
  } else if (state.phase === 'background' && (key === 'x' || key === 'X')) {
    event.preventDefault();
    excludeRecent();
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
  state.wake = (info.wake_words && info.wake_words[0]) || '小克';
  $('wake-word').textContent = (info.wake_words && info.wake_words.length)
    ? info.wake_words.join(' / ')
    : `${state.wake}（未在关键词文件里找到，使用默认）`;
  setDryRun(Boolean(info.live_dry_run));
  $('start-collection').disabled = false;
  $('start-background').disabled = false;
  $('start-training').disabled = false;
}

$('start-collection').addEventListener('click', startCollection);
$('start-background').addEventListener('click', startBackground);
$('start-training').addEventListener('click', startTraining);
$('bg-exclude').addEventListener('click', excludeRecent);
$('bg-stop').addEventListener('click', () => stopBackground(false));
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
// Browsers freeze hidden tabs (Edge sleeping tabs, Chrome memory saver), which
// stops the segment and target timers while the recording itself continues.
// Catch up as soon as the page runs again.
const catchUpBackground = () => {
  if (state.phase === 'background' && document.visibilityState === 'visible') backgroundTick();
};
document.addEventListener('visibilitychange', catchUpBackground);
document.addEventListener('resume', catchUpBackground);

window.addEventListener('beforeunload', (event) => {
  if (state.recording) {
    event.preventDefault();
    event.returnValue = '';
  }
});
window.addEventListener('pagehide', () => {
  const send = (payload) => fetch(`/api/command?token=${encodeURIComponent(token)}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
    keepalive: true
  });
  // A recording outlives its page otherwise and runs until the program stops.
  // Its labels cannot be saved from here, so the session stays unlabeled.
  if (state.recording) send({action: 'recording.stop'});
  if (state.previousDryRun !== null) {
    send({action: 'commands.live_dry_run', enabled: state.previousDryRun});
  }
});

init();
