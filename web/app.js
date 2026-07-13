const token = new URLSearchParams(location.search).get('token') || '';
const state = { telemetry: [], vad: [], wakes: [], candidates: [], config: null, maxPoints: 400, source: 'processed' };
const $ = (id) => document.getElementById(id);

async function command(action, payload = {}) {
  const response = await fetch(`/api/command?token=${encodeURIComponent(token)}`, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({action, ...payload})
  });
  const result = await response.json();
  if (!result.ok) addEvent({type:'command_error', payload:{message:result.error || 'unknown error'}, seq:'—'});
  return result;
}

function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${location.host}/ws?token=${encodeURIComponent(token)}`);
  ws.onopen = () => { $('connection').textContent='在线'; $('connection').className='pill online'; command('state.get'); };
  ws.onclose = () => { $('connection').textContent='离线'; $('connection').className='pill offline'; setTimeout(connect, 1200); };
  ws.onmessage = ({data}) => handle(JSON.parse(data));
}

function handle(message) {
  const {type, payload} = message;
  if (type === 'telemetry') {
    state.telemetry.push(payload); state.vad.push(!!payload.vad);
    while (state.telemetry.length > state.maxPoints) state.telemetry.shift();
    while (state.vad.length > state.maxPoints) state.vad.shift();
    const cutoff = Number(payload.sample || 0) - 20 * 16000;
    state.wakes = state.wakes.filter(hit => Number(hit.wake_span?.end ?? hit.at ?? 0) >= cutoff);
    state.candidates = state.candidates.filter(item => Number(item.wake_span?.end ?? 0) >= cutoff);
    $('rms').textContent = Number(payload[state.source]?.rms ?? payload.processed?.rms ?? 0).toFixed(3);
    $('queue-depth').textContent = payload.queue_depth ?? 0; $('drops').textContent = payload.telemetry_dropped ?? 0;
    if (message.source === 'replay') $('replay-seek').value = Number(payload.sample || 0) / 16000;
  } else if (type === 'kws_hit') {
    state.wakes.push({at:message.timestamp_sample, ...payload}); addEvent(message);
  } else if (type === 'candidate') {
    state.candidates.push(payload); addEvent(message);
  } else if (type === 'config_state' || type === 'config_applied') {
    const pending = payload.restart_required || [];
    loadConfig(pending.length && payload.saved_config ? payload.saved_config : (payload.config || payload));
    $('restart-required').hidden = !pending.length;
    $('restart-required').textContent = pending.length ? `已保存，重启后生效：${pending.join('、')}` : '';
    addEvent(message);
  } else if (type === 'recording_state') {
    const active = !!payload.active; $('recording').classList.toggle('active', active);
    $('recording').lastChild.textContent = active ? ` 录制中 · ${payload.session || ''}` : ' 未录制'; addEvent(message);
  } else if (type === 'sessions') {
    const select=$('sessions'); const current=select.value; select.innerHTML='<option value="">选择回放会话</option>';
    for(const session of payload.sessions || []) select.add(new Option(session.name, session.path)); select.value=current;
  } else if (type === 'replay_state') {
    $('source').textContent = payload.playing ? 'REPLAY ▶' : 'REPLAY';
    $('replay-seek').max = Math.max(1, Number(payload.duration_seconds || 0));
    $('replay-seek').value = Number(payload.position_seconds || 0); addEvent(message);
  } else { addEvent(message); }
}

function addEvent(message) {
  const row=document.createElement('tr'); const at=message.timestamp_sample ? (message.timestamp_sample/16000).toFixed(2)+'s' : new Date().toLocaleTimeString();
  const detail=JSON.stringify(message.payload ?? {}).slice(0,500);
  const level=message.level || (String(message.type).includes('error') || message.type==='config_rejected' ? 'error' : 'info');
  row.innerHTML=`<td>${message.seq ?? '—'}</td><td>${at}</td><td class="level-${level}">${level}</td><td>${message.type}</td><td></td>`; row.lastChild.textContent=detail;
  $('events').prepend(row); while($('events').children.length>250) $('events').lastChild.remove();
}

function loadConfig(config) {
  state.config=structuredClone(config); state.maxPoints=Math.max(20,Number(config.web?.telemetry_hz||20)*20); $('config-revision').textContent=`rev ${config.revision ?? '—'}`;
  for(const input of $('config-form').elements) if(input.name){ const [group,key]=input.name.split('.'); if(config[group]?.[key]!==undefined) input.value=config[group][key]; }
}

function configPatch() {
  const patch={}; for(const input of $('config-form').elements) if(input.name){ const [group,key]=input.name.split('.'); patch[group]??={}; patch[group][key]=input.type==='number' ? Number(input.value) : input.value; } return patch;
}

function canvas(id) { const element=$(id); const dpr=devicePixelRatio||1; const rect=element.getBoundingClientRect(); const w=Math.max(1,Math.round(rect.width*dpr)),h=Math.max(1,Math.round(rect.height*dpr)); if(element.width!==w||element.height!==h){element.width=w;element.height=h} const c=element.getContext('2d'); c.setTransform(dpr,0,0,dpr,0,0); return {c,w:rect.width,h:rect.height}; }
function draw() {
  const wave=canvas('waveform'), vad=canvas('vad'), kws=canvas('kws'); wave.c.clearRect(0,0,wave.w,wave.h); vad.c.clearRect(0,0,vad.w,vad.h); kws.c.clearRect(0,0,kws.w,kws.h);
  const values=state.telemetry.map(x=>x[state.source]||x.processed||{min:0,max:0}); const step=wave.w/Math.max(1,state.maxPoints); const offset=state.maxPoints-values.length;
  wave.c.strokeStyle='#59e1d9'; wave.c.lineWidth=1; wave.c.beginPath(); values.forEach((v,i)=>{const x=(offset+i)*step,y1=wave.h/2-(v.max||0)*wave.h*.44,y2=wave.h/2-(v.min||0)*wave.h*.44; wave.c.moveTo(x,y1);wave.c.lineTo(x,y2)}); wave.c.stroke();
  vad.c.fillStyle='rgba(116,226,154,.72)'; state.vad.forEach((v,i)=>{if(v)vad.c.fillRect((offset+i)*step,8,Math.ceil(step)+1,vad.h-16)});
  kws.c.strokeStyle='#ff6b72'; kws.c.fillStyle='rgba(255,107,114,.18)'; const latest=state.telemetry.at(-1)?.sample ?? 0, windowSamples=20*16000; for(const hit of state.wakes){const x=(hit.at-(latest-windowSamples))/windowSamples*kws.w;if(x>=0&&x<=kws.w){kws.c.fillRect(x-2,0,4,kws.h);kws.c.fillStyle='#ff9ca1';kws.c.fillText(hit.keyword||'KWS',x+4,17)} for(const [index,sample] of (hit.token_samples||[]).entries()){const tx=(sample-(latest-windowSamples))/windowSamples*kws.w;if(tx>=0&&tx<=kws.w){kws.c.fillRect(tx,24,1,kws.h-24);kws.c.fillText(hit.tokens?.[index]||'',tx+2,kws.h-5)}} const wake=hit.wake_span;if(wake){const wx=(wake.start-(latest-windowSamples))/windowSamples*wave.w,ww=(wake.end-wake.start)/windowSamples*wave.w;wave.c.fillStyle='rgba(255,107,114,.22)';wave.c.fillRect(wx,0,ww,wave.h)}}
  wave.c.fillStyle='rgba(93,168,255,.14)'; for(const item of state.candidates){for(const span of item.source_spans||[]){const x=(span.start-(latest-windowSamples))/windowSamples*wave.w, width=(span.end-span.start)/windowSamples*wave.w;wave.c.fillRect(x,0,width,wave.h)} const first=item.source_spans?.[0];if(first){const x=(first.start-(latest-windowSamples))/windowSamples*wave.w;wave.c.fillStyle='#8bc1ff';wave.c.fillText(item.position||'candidate',x+4,16);wave.c.fillStyle='rgba(93,168,255,.14)'}}
  requestAnimationFrame(draw);
}

$('wave-source').onchange=(e)=>state.source=e.target.value;
$('clear-events').onclick=()=>$('events').innerHTML='';
$('record-start').onclick=()=>command('recording.start'); $('record-stop').onclick=()=>command('recording.stop');
$('replay-open').onclick=()=>command('replay.open',{session:$('sessions').value}); $('replay-play').onclick=()=>command('replay.play'); $('replay-pause').onclick=()=>command('replay.pause');
$('replay-speed').onchange=()=>command('replay.speed',{speed:Number($('replay-speed').value)}); $('replay-seek').onchange=()=>command('replay.seek',{seconds:Number($('replay-seek').value)});
$('config-form').onsubmit=(e)=>{e.preventDefault();command('config.apply',{patch:configPatch(),save:false})}; $('config-save').onclick=()=>command('config.apply',{patch:configPatch(),save:true}); $('config-reset').onclick=()=>state.config&&loadConfig(state.config);
connect(); draw();
