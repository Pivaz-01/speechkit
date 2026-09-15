/* ------------------------------------------------------------------------
   The form is generated from /api/schema, which is generated from
   speechkit/settings.py. Adding a setting means adding one line there; there
   is no field list in this file to keep in step with it.
   ------------------------------------------------------------------------ */

const STAGES = [
  { id: 'acoustics', title: 'Acoustic analysis', run: 'Run acoustic analysis',
    blurb: 'Praat measures over whole recordings: pitch, perturbation, spectral ' +
           'and rhythm metrics, written as CSV files with plots.' },
  { id: 'phonemes', title: 'Phoneme recognition', run: 'Recognise phonemes',
    blurb: 'Labels every phoneme and pause with wav2vec2, writes one TSV per ' +
           'recording, and loads them into the editor for review.' },
  { id: 'alignment', title: 'Alignment to text', run: 'Align to passage text',
    blurb: 'Compares the phonemes produced against the canonical sequence for the ' +
           'passage, word by word, and reports accuracy.' },
  { id: 'server', title: 'Server', run: null,
    blurb: 'Where this interface listens, and how it loads native libraries.' },
];

const state = {
  schema: [],
  values: {},
  sections: {},
  presets: [],
  stage: 'acoustics',
  showAdvanced: false,
  showOtherTask: false,
  cursor: 0,
  dirty: false,
  env: null,
};

const TASK_LABELS = {
  reading: 'reading passages',
  sustained_vowel: 'sustained vowels',
  auto: 'a mixed folder',
};

// The acoustics panel is gated on the task, because the two tasks are measured
// by different code paths and most of the window and token settings do nothing
// on a reading passage. Showing all 41 at once invites people to tune
// parameters that are never read.
function currentTask() {
  return String(state.values.TASK || '');
}

function fieldAppliesToTask(f) {
  if (f.section !== 'acoustics' || !f.tasks || !f.tasks.length) return true;
  const task = currentTask();
  if (task === 'auto') return true;      // mixed folder: every path can run
  return f.tasks.includes(task);
}

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, props = {}, ...kids) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const k of kids.flat()) {
    if (k == null) continue;
    node.append(k.nodeType ? k : document.createTextNode(k));
  }
  return node;
};

/* ------------------------------------------------------------------ boot */

async function boot() {
  const [schema, env] = await Promise.all([
    fetch('/api/schema').then(r => r.json()),
    fetch('/api/env').then(r => r.json()),
  ]);
  state.schema = schema.schema;
  state.values = schema.values;
  state.sections = schema.sections;
  state.presets = schema.presets;
  state.env = env;
  $('#settingsPath').textContent = schema.settingsFile;

  renderEnv();
  renderRail();
  selectStage(location.hash.slice(1) || 'acoustics');
  pollJob();
  setInterval(pollJob, 1200);

  window.addEventListener('beforeunload', e => {
    if (state.dirty) { e.preventDefault(); e.returnValue = ''; }
  });
}

/* ----------------------------------------------------------- environment */

function renderEnv() {
  const wrap = $('#env');
  wrap.textContent = '';
  for (const s of ['acoustics', 'phonemes', 'alignment']) {
    const info = state.env.stages[s];
    const chip = el('span', {
      className: 'chip',
      title: info.ready
        ? `${s}: all dependencies present`
        : `${s} needs: ${info.missing.join(', ')}\npip install ${info.missing.join(' ')}`,
    }, `${s} ${info.ready ? 'ready' : 'missing ' + info.missing.length}`);
    chip.dataset.state = info.ready ? 'ok' : 'missing';
    wrap.append(chip);
  }
  const py = state.env.python_check;
  if (py) {
    const chip = el('span', { className: 'chip', title: py.message },
      'python ' + state.env.python);
    chip.dataset.state = py.ok === true ? 'ok' : py.ok === false ? 'missing' : '';
    wrap.append(chip);
  }
  const cuda = state.env.cuda;
  if (cuda) {
    const chip = el('span', {
      className: 'chip',
      title: cuda.device || 'no CUDA device visible to torch',
    }, cuda.available ? 'cuda' : 'cpu only');
    chip.dataset.state = cuda.available ? 'ok' : '';
    wrap.append(chip);
  }
}

/* ------------------------------------------------------------------ rail */

function renderRail() {
  const wrap = $('#stages');
  wrap.textContent = '';
  STAGES.forEach((s, i) => {
    const info = state.env.stages[s.id];
    const btn = el('button', { className: 'stage-btn', type: 'button' },
      el('span', { className: 'num' }, s.id === 'server' ? '·' : String(i + 1)),
      el('span', {}, s.title),
      el('span', { className: 'sub' },
        info ? (info.ready ? 'ready' : 'needs ' + info.missing.join(', ')) : 'local'));
    btn.dataset.stage = s.id;
    btn.onclick = () => selectStage(s.id);
    wrap.append(btn);
    if (s.id === 'phonemes') {
      wrap.append(el('a', { className: 'rail-link', href: '/editor', target: '_blank',
                            rel: 'noopener' }, 'Open the phoneme editor'));
    }
  });
}

function selectStage(id) {
  if (!STAGES.some(s => s.id === id)) id = 'acoustics';
  state.stage = id;
  location.hash = id;
  document.body.dataset.stage = id;
  for (const btn of document.querySelectorAll('.stage-btn')) {
    btn.setAttribute('aria-current', String(btn.dataset.stage === id));
  }
  renderPanel();
}

/* ----------------------------------------------------------------- panel */

function renderPanel() {
  const stage = STAGES.find(s => s.id === state.stage);
  const panel = $('#panel');
  panel.textContent = '';
  panel.scrollTop = 0;

  panel.append(
    el('div', { className: 'panel-head' },
      el('div', {},
        el('h1', {}, stage.title),
        el('p', { className: 'blurb' }, stage.blurb)),
      el('div', { className: 'panel-tools' },
        el('label', { className: 'switch' },
          Object.assign(el('input', { type: 'checkbox', checked: state.showAdvanced }),
            { onchange: e => { state.showAdvanced = e.target.checked; renderPanel(); } }),
          el('span', {}, 'Advanced settings')),
        button('Reset this section', () => resetSection(stage.id), 'quiet small'))));

  let fields = state.schema.filter(f => f.section === stage.id);
  if (stage.id === 'acoustics') {
    panel.append(taskPicker());
    // Until the task is known, nothing else is meaningful, so show nothing else.
    if (!currentTask()) {
      renderRunbar();
      return;
    }
    fields = fields.filter(f => f.kind !== 'task'
      && (state.showOtherTask || fieldAppliesToTask(f)));
  }
  const groups = [];
  for (const f of fields) {
    let g = groups.find(x => x.name === f.group);
    if (!g) groups.push(g = { name: f.group, fields: [] });
    g.fields.push(f);
  }

  for (const g of groups) {
    const visible = g.fields.filter(f => state.showAdvanced || !f.advanced);
    if (!visible.length) continue;
    const body = el('div', { className: 'fields' }, visible.map(renderField));
    const details = el('details', { className: 'group', open: true },
      el('summary', {}, g.name), body);
    panel.append(details);
  }

  panel.append(el('div', { className: 'results', id: 'results' }));
  renderRunbar();
  renderSummary();
}

function taskPicker() {
  const f = state.schema.find(x => x.name === 'TASK');
  const task = currentTask();
  const wrap = el('div', { className: 'taskpick' });

  wrap.append(el('h3', {}, f.label));
  const row = el('div', { className: 'taskopts' });
  for (const c of f.choices) {
    const b = el('button', { className: 'taskopt', type: 'button' },
      el('strong', {}, c.label));
    b.dataset.value = c.value;
    b.setAttribute('aria-pressed', String(c.value === task));
    b.onclick = () => {
      state.values.TASK = c.value;
      save('TASK');
      renderPanel();
    };
    row.append(b);
  }
  wrap.append(row);

  if (!task) {
    wrap.append(el('p', { className: 'taskhelp' }, f.help));
    return wrap;
  }

  const hidden = state.schema.filter(
    x => x.section === 'acoustics' && x.kind !== 'task' && !fieldAppliesToTask(x));
  const note = el('p', { className: 'taskhelp' });
  if (hidden.length) {
    note.append(`Showing the settings that apply to ${TASK_LABELS[task]}. `);
    const link = el('button', { className: 'linkish', type: 'button' },
      `${hidden.length} setting${hidden.length === 1 ? '' : 's'} that only affect the other task ` +
      (state.showOtherTask ? 'are shown below' : 'are hidden'));
    link.onclick = () => { state.showOtherTask = !state.showOtherTask; renderPanel(); };
    note.append(link);
    note.append('.');
  } else {
    note.append(`Every acoustic setting applies to ${TASK_LABELS[task]}.`);
  }
  wrap.append(note);
  return wrap;
}

function button(text, onclick, cls = '') {
  const b = el('button', { className: 'btn ' + cls, type: 'button' }, text);
  b.onclick = onclick;
  return b;
}

/* ---------------------------------------------------------------- fields */

function renderField(f) {
  const wrap = el('div', { className: 'field' });
  wrap.dataset.name = f.name;
  const label = el('div', { className: 'label' },
    el('label', { htmlFor: 'f_' + f.name }, f.label),
    el('code', { className: 'varname' }, f.name));
  if (f.tasks && f.tasks.length && !fieldAppliesToTask(f)) {
    wrap.classList.add('offtask');
    label.append(el('span', { className: 'tasktag' },
      f.tasks.includes('reading') ? 'reading only' : 'sustained vowel only'));
  }
  wrap.append(label);
  wrap.append(el('div', { className: 'control' }, control(f)));
  if (f.help) wrap.append(el('div', { className: 'help' }, f.help));
  applyDependency(wrap, f);
  return wrap;
}

function applyDependency(wrap, f) {
  if (!f.dependsOn) return;
  const current = state.values[f.dependsOn];
  const want = f.dependsValue;
  const shown = want === '__truthy__'
    ? Boolean(current) && String(current).length > 0
    : current === want;
  wrap.classList.toggle('hidden', !shown);
}

function refreshDependencies() {
  for (const f of state.schema) {
    if (!f.dependsOn) continue;
    const wrap = document.querySelector(`.field[data-name="${f.name}"]`);
    if (wrap) applyDependency(wrap, f);
  }
}

function control(f) {
  const id = 'f_' + f.name;
  const v = state.values[f.name];

  switch (f.kind) {
    case 'bool': {
      const input = el('input', { type: 'checkbox', id, checked: Boolean(v) });
      input.onchange = () => set(f.name, input.checked);
      return el('label', { className: 'switch' }, input,
        el('span', {}, input.checked ? 'on' : 'off'));
    }
    case 'task':
    case 'choice': {
      const sel = el('select', { id });
      for (const c of f.choices) {
        sel.append(el('option', { value: c.value, selected: c.value === v }, c.label));
      }
      sel.onchange = () => set(f.name, sel.value);
      return sel;
    }
    case 'int':
    case 'float':
    case 'float_or_none': {
      const input = el('input', {
        type: 'number', id,
        value: v === null || v === undefined ? '' : v,
        placeholder: f.kind === 'float_or_none' ? 'automatic' : '',
      });
      if (f.min !== null) input.min = f.min;
      if (f.max !== null) input.max = f.max;
      if (f.step !== null) input.step = f.step;
      input.onchange = () => set(f.name, input.value, input);
      return input;
    }
    case 'paths':
      return pathList(f);
    case 'path':
    case 'dir': {
      const input = el('input', { type: 'text', id, value: v || '',
                                  placeholder: f.kind === 'dir' ? 'folder' : 'file' });
      input.onchange = () => set(f.name, input.value, input);
      return el('div', { className: 'inputwrap' }, input,
        button('Browse', async () => {
          const picked = await pick(f.kind === 'dir' ? 'dir' : 'file', input.value);
          if (picked) { input.value = picked[0]; set(f.name, picked[0], input); }
        }, 'small'));
    }
    case 'json': {
      const ta = el('textarea', { id, spellcheck: false },
        JSON.stringify(v ?? (Array.isArray(f.default) ? [] : {}), null, 2));
      ta.onchange = () => {
        try {
          set(f.name, JSON.parse(ta.value || (Array.isArray(f.default) ? '[]' : '{}')), ta);
          ta.classList.remove('bad');
        } catch (err) {
          ta.classList.add('bad');
          toast('Not valid JSON: ' + err.message, true);
        }
      };
      return ta;
    }
    case 'passages': {
      const lines = Object.entries(v || {})
        .map(([k, n]) => `${k} = ${n === null ? 'none' : n}`).join('\n');
      const ta = el('textarea', { id, spellcheck: false, rows: 9 }, lines);
      ta.onchange = () => set(f.name, ta.value, ta);
      return ta;
    }
    default: {
      const input = el('input', { type: 'text', id, value: v ?? '' });
      input.onchange = () => set(f.name, input.value, input);
      return input;
    }
  }
}

function pathList(f) {
  const wrap = el('div', { className: 'pathlist' });
  const meta = el('div', { className: 'meta' }, '');
  const rows = el('div', {});

  const values = () => [...rows.querySelectorAll('input')].map(i => i.value.trim())
    .filter(Boolean);

  const commit = () => { set(f.name, values()); preview(); };

  const addRow = (value = '') => {
    const input = el('input', { type: 'text', value, placeholder: 'folder or .wav file' });
    input.onchange = commit;
    const row = el('div', { className: 'pathrow' }, input,
      button('Browse', async () => {
        const picked = await pick('multi', input.value);
        if (!picked) return;
        input.value = picked[0];
        picked.slice(1).forEach(p => addRow(p));
        commit();
      }, 'small'),
      button('Remove', () => { row.remove(); commit(); }, 'small quiet'));
    rows.append(row);
    return input;
  };

  const preview = async () => {
    const list = values();
    if (!list.length) { meta.textContent = 'nothing selected'; meta.className = 'meta'; return; }
    const recursiveField = f.name === 'WAV_INPUTS' ? 'RECURSIVE'
      : f.name === 'PHONEME_INPUTS' ? 'PHONEME_RECURSIVE' : null;
    const body = {
      inputs: list,
      recursive: recursiveField ? Boolean(state.values[recursiveField]) : false,
    };
    const r = await fetch('/api/preview_inputs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => r.json());
    if (r.missing && r.missing.length) {
      meta.className = 'meta warn';
      meta.textContent = `does not exist: ${r.missing.join(', ')}`;
      return;
    }
    meta.className = 'meta';
    meta.textContent = f.name === 'INPUT_DIRS'
      ? `${list.length} folder(s) selected`
      : `${r.count} recording(s)` + (r.sample.length ? ` — ${r.sample.slice(0, 4).join(', ')}${r.count > 4 ? ', …' : ''}` : '');
  };

  (state.values[f.name] || []).forEach(p => addRow(p));
  wrap.append(rows, el('div', { className: 'pathrow' },
    button('Add folder or file', () => addRow().focus(), 'small')), meta);
  preview();
  return wrap;
}

/* --------------------------------------------------------------- set/save */

let saveTimer;
function set(name, raw, input) {
  state.values[name] = raw;
  state.dirty = true;
  refreshDependencies();
  const sw = input && input.type === 'checkbox' && input.parentElement.querySelector('span');
  if (sw) sw.textContent = input.checked ? 'on' : 'off';
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => save(name, input), 400);
}

async function save(name, input) {
  const r = await fetch('/api/settings', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(state.values),
  });
  const body = await r.json();
  if (!r.ok) {
    if (input) input.classList.add('bad');
    toast(body.error || 'Could not save settings', true);
    return;
  }
  if (input) input.classList.remove('bad');
  state.values = body.values;
  state.dirty = false;
}

async function resetSection(section) {
  const r = await fetch('/api/settings/reset', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ section }),
  }).then(r => r.json());
  state.values = r.values;
  renderPanel();
  toast('Settings in this section are back to their defaults');
}

/* ------------------------------------------------------------- presets */

async function savePreset() {
  const name = prompt('Save the current settings as a preset called:');
  if (!name) return;
  const r = await fetch('/api/presets', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  const body = await r.json();
  if (!r.ok) { toast(body.error, true); return; }
  state.presets = body.presets;
  renderPresets();
  toast(`Saved the preset “${name}”`);
}

async function loadPreset(name) {
  if (!name) return;
  const r = await fetch('/api/presets?name=' + encodeURIComponent(name));
  const body = await r.json();
  if (!r.ok) { toast(body.error, true); return; }
  state.values = body.values;
  renderPanel();
  toast(`Loaded “${name}”`);
}

function renderPresets() {
  const wrap = $('#presets');
  wrap.textContent = '';
  const sel = el('select', { id: 'presetSel' }, el('option', { value: '' }, 'Presets…'));
  for (const p of state.presets) sel.append(el('option', { value: p }, p));
  sel.onchange = () => { loadPreset(sel.value); sel.value = ''; };
  wrap.append(sel, button('Save current', savePreset, 'small quiet'));
}

/* ------------------------------------------------------------- path picker */

let pickerResolve = null;

async function pick(mode, startPath) {
  const dlg = $('#picker');
  $('#pickerMode').textContent = mode === 'dir'
    ? 'Choose a folder'
    : mode === 'multi' ? 'Choose folders or files' : 'Choose a file';
  $('#pickUse').textContent = mode === 'dir' ? 'Use this folder' : 'Use selection';
  dlg.dataset.mode = mode;
  dlg.returnValue = '';
  await browse(startPath || '');
  dlg.showModal();
  return new Promise(res => { pickerResolve = res; });
}

async function browse(path) {
  const r = await fetch('/api/browse?path=' + encodeURIComponent(path));
  const data = await r.json();
  const listing = $('#pickerList');
  listing.textContent = '';
  $('#pickerPath').value = data.path || path;

  if (data.error) {
    listing.append(el('div', { className: 'entry file' }, data.error));
    return;
  }
  if (data.parent) {
    listing.append(entryButton('..', data.parent, null, 'dir'));
  }
  for (const root of data.roots || []) {
    listing.append(entryButton(root.name, root.path, null, 'dir'));
  }
  for (const d of data.dirs) {
    listing.append(entryButton(d.name + '/', d.path, d.wavs > 0 ? `${d.wavs} wav` : '', 'dir'));
  }
  const mode = $('#picker').dataset.mode;
  if (mode !== 'dir') {
    for (const f of data.files) {
      const b = entryButton(f.name, f.path, kb(f.size), 'file');
      if (f.kind === 'wav') b.classList.add('audio');
      listing.append(b);
    }
  }
}

function entryButton(label, path, note, kind) {
  const b = el('button', { className: 'entry ' + kind, type: 'button' },
    el('span', {}, label),
    note ? el('span', { className: 'count' }, note) : null);
  b.onclick = () => {
    if (kind === 'dir') browse(path);
    else finishPick([path]);
  };
  return b;
}

function finishPick(paths) {
  $('#picker').close();
  if (pickerResolve) { pickerResolve(paths); pickerResolve = null; }
}

function kb(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return Math.round(n / 1024) + ' kB';
  return (n / 1048576).toFixed(1) + ' MB';
}

/* --------------------------------------------------------------- run bar */

function renderRunbar() {
  const stage = STAGES.find(s => s.id === state.stage);
  const row = $('#runRow');
  row.textContent = '';
  if (!stage.run) {
    row.append(el('span', { className: 'state' },
      'Changes here apply the next time the interface starts.'));
    return;
  }
  const info = state.env.stages[stage.id];
  const runBtn = button(stage.run, () => runStage(stage.id), 'primary');
  runBtn.id = 'runBtn';
  if (!info.ready) {
    runBtn.disabled = true;
    runBtn.title = 'pip install ' + info.missing.join(' ');
  } else if (stage.id === 'acoustics' && !currentTask()) {
    runBtn.disabled = true;
    runBtn.title = 'Choose what is in these recordings first';
  }
  const stopBtn = button('Stop', cancelJob, 'danger');
  stopBtn.id = 'stopBtn';
  stopBtn.disabled = true;
  row.append(
    runBtn, stopBtn,
    el('span', { className: 'state', id: 'runState' }, 'idle'),
    el('span', { className: 'spacer' }),
    button('Log', () => $('#log').classList.toggle('open'), 'quiet small'));
}

async function runStage(stage) {
  clearTimeout(saveTimer);
  $('#log').textContent = '';
  state.cursor = 0;
  const r = await fetch('/api/run/' + stage, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(state.values),
  });
  const body = await r.json();
  if (!r.ok) { toast(body.error, true); return; }
  state.dirty = false;
  $('#log').classList.add('open');
  pollJob();
}

async function cancelJob() {
  const r = await fetch('/api/job/cancel', { method: 'POST' }).then(r => r.json());
  if (!r.cancelled) toast('Nothing is running');
}

async function pollJob() {
  let status;
  try {
    status = await fetch('/api/job?since=' + state.cursor).then(r => r.json());
  } catch { return; }
  state.cursor = status.cursor;

  const log = $('#log');
  for (const line of status.lines) {
    const cls = /^\[(error|stopped|cancel)/.test(line) ? 'err'
      : /^\[(done|model|input|editor|passages)/.test(line) ? 'hi' : '';
    log.append(el('span', { className: cls }, line + '\n'));
  }
  if (status.lines.length) log.scrollTop = log.scrollHeight;

  const stateEl = $('#runState');
  const runBtn = $('#runBtn');
  const stopBtn = $('#stopBtn');
  if (stateEl) {
    stateEl.dataset.state = status.state;
    const mins = Math.floor(status.elapsed / 60);
    const secs = String(Math.floor(status.elapsed % 60)).padStart(2, '0');
    stateEl.textContent = status.state === 'idle' ? 'idle'
      : `${status.state} · ${mins}:${secs}` +
        (status.stage && status.stage !== state.stage ? ` (${status.stage})` : '');
  }
  if (runBtn) runBtn.disabled = status.running
    || !state.env.stages[state.stage]?.ready
    || (state.stage === 'acoustics' && !currentTask());
  if (stopBtn) stopBtn.disabled = !status.running;

  if (status.summary) renderSummary(status.summary);
  if (status.state === 'failed' && status.error && state.lastError !== status.error) {
    state.lastError = status.error;
    toast(status.error, true);
  }
}

/* ---------------------------------------------------------------- results */

let lastSummary = null;

function renderSummary(summary) {
  if (summary) lastSummary = summary;
  const wrap = $('#results');
  if (!wrap) return;
  const s = lastSummary;
  wrap.textContent = '';
  if (!s || s.kind !== state.stage) return;

  if (s.kind === 'phonemes' && s.rows) {
    wrap.append(el('h3', {}, 'Recognised'));
    wrap.append(table(
      ['file', 'phonemes', 'seconds', 'speech rate', 'articulation rate'],
      s.rows.map(r => [r.file, r.phonemes, r.duration, r.speech_rate, r.articulation_rate]),
      [false, true, true, true, true]));
    if (s.editorReady) {
      wrap.append(el('div', { className: 'filelist' },
        el('a', { href: '/editor', target: '_blank', rel: 'noopener' },
          'review in the phoneme editor')));
    }
  }

  if (s.kind === 'alignment' && s.rows) {
    wrap.append(el('h3', {}, 'Phoneme accuracy'));
    wrap.append(table(
      ['passage', 'session', 'canonical', 'produced', 'accuracy %', 'sub', 'omit', 'ins'],
      s.rows.map(r => [r.passage, r.session, r.canonical, r.produced, r.accuracy,
                       r.substitutions, r.omissions, r.insertions]),
      [false, false, true, true, true, true, true, true]));
    wrap.append(fileLinks(s.outputs.map(p => ({ name: p.split(/[\\/]/).pop(), path: p }))));
  }

  if (s.kind === 'acoustics') {
    wrap.append(el('h3', {}, `Output (${s.files} recording${s.files === 1 ? '' : 's'})`));
    wrap.append(fileLinks(s.outputs || []));
  }
}

function table(headers, rows, numeric) {
  const t = el('table', { className: 'data' });
  t.append(el('thead', {}, el('tr', {}, headers.map(h => el('th', {}, h)))));
  t.append(el('tbody', {}, rows.map(r => el('tr', {},
    r.map((cell, i) => el('td', { className: numeric[i] ? 'num' : '' }, String(cell)))))));
  return t;
}

function fileLinks(files) {
  const wrap = el('div', { className: 'filelist' });
  for (const f of files) {
    wrap.append(el('a', { href: '/api/download?path=' + encodeURIComponent(f.path) }, f.name));
  }
  if (!files.length) wrap.append(el('span', { className: 'meta' }, 'no files written'));
  return wrap;
}

/* ----------------------------------------------------------------- toast */

let toastTimer;
function toast(msg, bad = false) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.toggle('bad', bad);
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), bad ? 6000 : 2600);
}

/* ------------------------------------------------------------------ wire */

document.addEventListener('DOMContentLoaded', () => {
  $('#pickerPath').addEventListener('change', e => browse(e.target.value));
  $('#pickUse').onclick = () => finishPick([$('#pickerPath').value]);
  $('#pickCancel').onclick = () => finishPick(null);
  $('#picker').addEventListener('close', () => {
    if (pickerResolve) { pickerResolve(null); pickerResolve = null; }
  });
  boot().then(renderPresets).catch(err => {
    document.body.append(el('div', { className: 'toast show bad' },
      'Could not load the interface: ' + err.message));
  });
});
