'use strict';

const token = new URLSearchParams(location.search).get('token') || '';
const $ = (id) => document.getElementById(id);
const MAX_LINES = 300;

const state = {
  active: false,
  previousDryRun: null
};

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

function showMessage(text = '') {
  $('message').textContent = text;
}

function setDryRun(enabled) {
  $('dry-run').textContent = enabled ? '系统动作：已停用' : '系统动作：会执行';
  $('dry-run').className = `pill ${enabled ? 'dry' : 'live-actions'}`;
}

function setActive(active) {
  state.active = active;
  $('state').className = `record-indicator${active ? ' active' : ''}`;
  $('state-label').textContent = active ? '转写中' : '未转写';
  $('toggle').textContent = active ? '停止转写' : '开始转写';
}

function clock() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, '0');
  return `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
}

function addLine(kind, text, detail = '') {
  const item = document.createElement('li');
  item.className = kind;
  const time = document.createElement('span');
  time.className = 'time';
  time.textContent = clock();
  const body = document.createElement('span');
  body.className = 'text';
  body.textContent = text;
  item.append(time, body);
  if (detail) {
    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = detail;
    item.append(meta);
  }
  $('lines').prepend(item);
  while ($('lines').children.length > MAX_LINES) $('lines').lastElementChild.remove();
  $('empty').hidden = true;
}

function onEvent(type, payload) {
  if (type === 'transcript') {
    const seconds = (Number(payload.duration_ms) / 1000).toFixed(1);
    const text = payload.error ? `（识别失败：${payload.error}）` : (payload.text || '（没有识别出文字）');
    addLine(payload.wake_word_heard ? 'heard wake' : 'heard', text,
      `${seconds} 秒语音 · 识别 ${Math.round(Number(payload.decode_ms) || 0)} ms` +
      (payload.wake_word_heard ? ' · 听到唤醒词' : ''));
  } else if (type === 'command_plan') {
    const dry = payload.execution_mode === 'dry_run';
    addLine('plan', `${dry ? '会执行' : '已执行'}：${payload.normalized_text || ''}`,
      `识别为「${payload.raw_text || ''}」${dry ? ' · 系统动作已停用' : ''}`);
  } else if (type === 'command_rejected') {
    addLine('reject', `未执行：「${payload.text || ''}」`, payload.reason || '');
  } else if (type === 'live_dry_run_state') {
    setDryRun(Boolean(payload.enabled));
  } else if (type === 'transcribe_state') {
    setActive(Boolean(payload.enabled));
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
    onEvent(type, payload);
  };
}

async function start() {
  $('toggle').disabled = true;
  showMessage('');
  if ($('use-dry-run').checked) {
    const info = await command('evaluation.info');
    const result = await command('commands.live_dry_run', {enabled: true});
    if (!result.ok) {
      showMessage(`无法停用系统动作：${result.error || '未知错误'}`);
      $('toggle').disabled = false;
      return;
    }
    state.previousDryRun = Boolean(info.live_dry_run);
  }
  const result = await command('transcribe.set', {enabled: true});
  $('toggle').disabled = false;
  if (!result.ok) {
    showMessage(`无法开始转写：${result.error || '未知错误'}`);
    await restoreDryRun();
    return;
  }
  setActive(true);
}

async function restoreDryRun() {
  if (state.previousDryRun === null) return;
  const previous = state.previousDryRun;
  state.previousDryRun = null;
  await command('commands.live_dry_run', {enabled: previous});
}

async function stop() {
  $('toggle').disabled = true;
  await command('transcribe.set', {enabled: false});
  await restoreDryRun();
  setActive(false);
  $('toggle').disabled = false;
}

async function init() {
  connect();
  const info = await command('evaluation.info');
  if (!info.ok) {
    showMessage(`无法连接运行时：${info.error || '未知错误'}。请从调试台的链接打开本页。`);
    return;
  }
  $('model').textContent = info.final_model_dir || '未知';
  setDryRun(Boolean(info.live_dry_run));
  setActive(Boolean(info.live_transcribe));
  if (!info.transcribe_available) {
    showMessage('当前配置没有启用二级唤醒（kws.asr_probe）或离线整句识别模型，无法转写。');
    return;
  }
  $('toggle').disabled = false;
}

$('toggle').addEventListener('click', () => (state.active ? stop() : start()));
$('clear').addEventListener('click', () => {
  $('lines').replaceChildren();
  $('empty').hidden = false;
});
window.addEventListener('pagehide', () => {
  const send = (payload) => fetch(`/api/command?token=${encodeURIComponent(token)}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
    keepalive: true
  });
  if (state.active) send({action: 'transcribe.set', enabled: false});
  if (state.previousDryRun !== null) {
    send({action: 'commands.live_dry_run', enabled: state.previousDryRun});
  }
});

init();
