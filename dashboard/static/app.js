const $ = (id) => document.getElementById(id);
const colors = { Bus: '#ffa62b', Car: '#36d399', Lamp: '#facc15', Motorcycle: '#f472b6', People: '#60a5fa', Truck: '#c084fc' };
let lastFrame = 0;
let toastTimer;

function toast(message) {
  $('toast').textContent = message;
  $('toast').classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('toast').classList.remove('show'), 2800);
}

async function post(url, payload = {}) {
  const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function setSegment(id, value) {
  document.querySelectorAll(`#${id} button`).forEach((button) => button.classList.toggle('active', button.dataset.value === value));
}

function renderCounts(counts = {}) {
  const container = $('class-counts');
  const entries = Object.entries(counts);
  $('class-total').textContent = `${entries.length} ${entries.length === 1 ? 'class' : 'classes'}`;
  if (!entries.length) { container.innerHTML = '<p class="empty-copy">No detections in current frame</p>'; return; }
  container.innerHTML = entries.map(([name, count]) => `<div class="class-chip" style="--chip:${colors[name] || '#fff'}"><span><i></i>${name}</span><b>${count}</b></div>`).join('');
}

function render(status) {
  const latest = status.latest || {};
  const sources = status.sources || {};
  const rgb = sources.rgb || {}, thermal = sources.thermal || {};
  $('fps').textContent = status.fps?.toFixed?.(1) ?? status.fps ?? '—';
  $('latency').textContent = status.mean_inference_ms ?? '—';
  $('resolution').textContent = latest.resolution ?? '—';
  $('detection-total').textContent = latest.detections ?? '—';
  $('policy-summary').textContent = status.policy === 'sequence' ? 'Sequence controller' : 'Fixed baseline';
  $('model-name').textContent = status.model || '—';
  $('device-label').textContent = `${status.device || '—'} · ${status.model || 'no model'}`;
  $('frame-label').textContent = latest.frame ? `FRAME ${String(latest.frame).padStart(6, '0')} · ${latest.end_to_end_ms} MS END-TO-END` : 'Waiting for synchronized frames';
  $('updated').textContent = latest.timestamp ? `Updated ${latest.timestamp.replace('T', ' ')}` : 'No frames processed';
  $('skew').textContent = sources.skew_ms == null ? '—' : Number(sources.skew_ms).toFixed(1);
  $('dropped').textContent = (sources.dropped_rgb || 0) + (sources.dropped_thermal || 0);
  $('rgb-source').textContent = rgb.source || '—'; $('thermal-source').textContent = thermal.source || '—';
  $('rgb-dot').className = `mini-dot ${rgb.connected ? 'ok' : 'error'}`;
  $('thermal-dot').className = `mini-dot ${thermal.connected ? 'ok' : 'error'}`;
  $('confidence').value = status.confidence ?? .25;
  $('confidence-value').textContent = Number(status.confidence ?? .25).toFixed(2);
  $('confidence').style.background = `linear-gradient(90deg, var(--cyan) ${((status.confidence ?? .25) - .1) / .8 * 100}%, #27384b 0)`;
  setSegment('policy', status.policy); setSegment('view', status.view);
  renderCounts(latest.class_counts);

  const healthy = status.running && rgb.connected && thermal.connected && !status.error;
  $('state-dot').className = `state-dot ${healthy ? 'ok' : status.error ? 'error' : ''}`;
  $('state-label').textContent = status.loading ? 'Loading model' : healthy ? 'System online' : status.running ? 'Waiting for input' : 'Pipeline stopped';
  $('error-banner').classList.toggle('hidden', !status.error);
  $('error-text').textContent = status.error || '';
  $('stream-empty').classList.toggle('hidden', Boolean(latest.frame));
  const toggle = $('run-toggle');
  toggle.classList.toggle('stopped', !status.running);
  toggle.querySelector('.button-icon').textContent = status.running ? '■' : '▶';
  toggle.querySelector('span:last-child').textContent = status.running ? 'Stop pipeline' : 'Start pipeline';
  toggle.dataset.running = String(status.running);
  if (latest.frame && latest.frame !== lastFrame) lastFrame = latest.frame;
}

async function refresh() {
  try { const response = await fetch('/api/status', { cache: 'no-store' }); render(await response.json()); }
  catch (error) { $('state-label').textContent = 'Dashboard offline'; $('state-dot').className = 'state-dot error'; }
}

$('run-toggle').addEventListener('click', async () => {
  try { render(await post('/api/control', { running: $('run-toggle').dataset.running !== 'true' })); }
  catch (error) { toast(error.message); }
});

['policy', 'view'].forEach((id) => document.querySelectorAll(`#${id} button`).forEach((button) => button.addEventListener('click', async () => {
  try { render(await post('/api/control', { [id]: button.dataset.value })); }
  catch (error) { toast(error.message); }
})));

let confidenceTimer;
$('confidence').addEventListener('input', (event) => {
  const value = Number(event.target.value);
  $('confidence-value').textContent = value.toFixed(2);
  event.target.style.background = `linear-gradient(90deg, var(--cyan) ${(value - .1) / .8 * 100}%, #27384b 0)`;
  clearTimeout(confidenceTimer);
  confidenceTimer = setTimeout(() => post('/api/control', { confidence: value }).catch((error) => toast(error.message)), 120);
});

$('snapshot').addEventListener('click', async () => {
  try { const data = await post('/api/snapshot'); toast(`Saved ${data.image.split('/').pop()}`); }
  catch (error) { toast(error.message); }
});

$('stream').addEventListener('error', () => $('stream-empty').classList.remove('hidden'));
refresh();
setInterval(refresh, 700);
