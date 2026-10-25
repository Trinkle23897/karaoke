// Room revisions order commands; progress and cache updates can share a revision.
export function isStaleRoomSnapshot(incoming, current) {
  return incoming.revision < current.revision ||
    (incoming.revision === current.revision && incoming.server_time < current.server_time);
}

export function applyRoomPatch(current, patch) {
  if (!current || current.stream !== patch.stream || current.sequence !== patch.base) return null;
  const next = {...current, ...patch.changes, sequence:patch.sequence,
    server_time:patch.server_time, playback_entry:patch.playback_entry};
  for (const [key, delta] of Object.entries(patch.collections)) {
    const items = new Map(current[key].map(item => [item.id, item]));
    for (const id of delta.remove) items.delete(id);
    for (const item of delta.upsert) items.set(item.id, item);
    next[key] = delta.order ? delta.order.map(id => items.get(id)) : [...items.values()];
    if (next[key].some(item => !item)) return null;
  }
  return next;
}

// A quiet room still sends a heartbeat. Reopen a dead stream without polling
// room state; the new subscription starts with the current complete snapshot.
export class RoomEvents {
  constructor(receive, connection, create = url => new EventSource(url), random = Math.random) {
    Object.assign(this, {receive, connection, create, random, attempts:0});
  }
  connect(job = '', playingEntry = '', force = false) {
    if (this.source && job === this.job && playingEntry === this.playingEntry && !force) return;
    if (job !== this.job || playingEntry !== this.playingEntry) this.attempts = 0;
    this.close(); this.job = job; this.playingEntry = playingEntry;
    this.retrying = false;
    let snapshot = null;
    const query = new URLSearchParams();
    if (job) query.set('job_id', job);
    if (playingEntry) query.set('playing_entry_id', playingEntry);
    const source = this.source = this.create('/v1/room/events' + (query.size ? `?${query}` : ''));
    const alive = () => {
      if (this.source !== source || this.retrying) return false;
      this.schedule(source, 45000);
      this.connection(true);
      return true;
    };
    const receive = (event, delta) => {
      if (this.source !== source || this.retrying) return;
      try {
        const value = JSON.parse(event.data);
        snapshot = delta ? applyRoomPatch(snapshot, value) : value;
        if (!snapshot) { this.retry(source); return; }
      } catch { this.retry(source); return; }
      this.attempts = 0;
      if (alive()) this.receive(snapshot);
    };
    source.addEventListener('room', event => receive(event, false));
    source.addEventListener('room_patch', event => receive(event, true));
    source.addEventListener('ping', () => { if (alive()) this.attempts = 0; });
    source.onopen = alive;
    source.onerror = () => {
      this.retry(source);
    };
    this.schedule(source, 45000);
  }
  retry(source) {
    if (this.source !== source || this.retrying) return;
    this.retrying = true;
    source.close(); // Own the retry so thousands of native retries cannot synchronize.
    this.connection(false);
    const delay = Math.min(30000, 1000 * 2 ** Math.min(this.attempts++, 5));
    this.schedule(source, delay * (.5 + .5 * this.random()), true);
  }
  schedule(source, delay, reconnect = false) {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      if (this.source !== source) return;
      if (reconnect) this.connect(this.job, this.playingEntry, true);
      else this.retry(source);
    }, delay);
  }
  close() {
    clearTimeout(this.timer);
    this.source?.close(); this.source = null;
  }
}

export function queueToTop(entries, id, playingIds = []) {
  const ids = entries.map(entry => entry.id);
  const playing = ids.filter(item => (Array.isArray(playingIds) ? playingIds : [playingIds]).includes(item));
  if (!ids.includes(id) || playing.includes(id)) return null;
  const first = [...playing, id];
  const order = [...first, ...ids.filter(item => !first.includes(item))];
  return order.some((item, index) => item !== ids[index]) ? order : null;
}

export function queueTopCandidates(entries, playingIds = []) {
  const playing = new Set(playingIds);
  const count = entries.filter(entry => playing.has(entry.id)).length;
  const pinned = entries.slice(0, count).every(entry => playing.has(entry.id));
  return new Set(entries.filter((entry, index) => !playing.has(entry.id) && (!pinned || index !== count))
    .map(entry => entry.id));
}

export function preparationRetry(entry) {
  if (entry?.status === 'failed' || (entry?.status === 'queued' && entry.next_retry_at))
    return {action:'retry', label:'重试'};
  return null;
}

const bounded = (value, min, max, fallback) => Number.isFinite(value) ? Math.max(min, Math.min(max, value)) : fallback;

// One tab owns audio. Saved progress from an older owner cannot overwrite a
// newer tab's commands; storage is scoped to this browser, never the room.
export class PlaybackTabs {
  constructor(storage, id, fallback = {}) {
    this.storage = storage; this.id = id; this.key = 'karaokeBrowserPlayback';
    this.sequence = 0;
    this.record = {revision:0, change:null, session:null, owner:null, playback:fallback};
    this.read();
  }
  read() {
    try {
      const saved = JSON.parse(this.storage?.getItem(this.key) || 'null');
      // Storage, not a locally incremented number, decides the last writer.
      // Two tabs can legitimately write the same revision after a concurrent read.
      if (saved && Number.isSafeInteger(saved.revision) && saved.playback)
        this.record = {...saved, change:saved.change || `${saved.owner}:${saved.revision}`,
          session:saved.session || saved.owner};
    } catch { /* Private storage restrictions must not disable local playback. */ }
    return this.record;
  }
  owns(session) {
    const current = this.read();
    return current.owner === this.id && (session === undefined || current.session === session);
  }
  write(playback, {claim = false, release = false, expectedSession} = {}) {
    const previous = this.record.change, current = this.read();
    if (expectedSession !== undefined && (current.owner || current.session !== expectedSession)) return false;
    if (!claim && (current.owner !== this.id || current.change !== previous)) return false;
    const change = `${this.id}:${++this.sequence}`;
    this.record = {revision:current.revision+1, change, session:claim ? change : current.session,
      owner:release ? null : this.id, playback};
    try { this.storage?.setItem(this.key, JSON.stringify(this.record)); } catch { /* Keep this tab's in-memory state when storage is unavailable. */ }
    return this.read().change === change;
  }
  suspend(playback) {
    this.suspendedSession = null;
    if (!this.owns() || !this.write(playback, {release:true})) return false;
    this.suspendedSession = this.record.session;
    return true;
  }
  resume() {
    const current = this.read(), session = this.suspendedSession;
    this.suspendedSession = null;
    // A restored page can resume only its own released session. Never steal
    // audio back after another tab has played, even if that tab is now closed.
    return Boolean(session && current.session === session && !current.owner &&
      this.write(current.playback, {claim:true, expectedSession:session}));
  }
}

// Only the playlist and preparation results belong to the room. Transport is
// private to this browser. Only an already-running performance may outlive a
// queue removal on another device; paused/restored selections must be current.
export class LocalPlayback {
  constructor(saved = {}) {
    this.entry = saved?.entry?.id && saved.entry.job_id ? saved.entry : null;
    this.state = {playing:saved?.playing === true, position:bounded(saved?.position, 0, Infinity, 0), started_at:null,
      levels:{instrumental:1, backing_vocals:1, lead_vocals:bounded(saved?.levels?.lead_vocals, 0, 1, 0)},
      pitch_semitones:Math.round(bounded(saved?.pitch_semitones, -6, 6, 0))};
    this.playedAt = saved?.played_at || null;
    this.skipped = new Set(saved?.skipped_ids || []);
    this.leadRestore = this.state.levels.lead_vocals || bounded(saved?.lead_restore, 0, 1, 1) || 1;
    // Older unsung operations did not distinguish skipping a waiting song from
    // removing it. Never replay an ambiguous deletion after an upgrade.
    this.pendingFinishes = (saved?.pending_finishes || []).filter(item => item.action || item.playback_started_at != null)
      .map(item => ({action:'finish', ...item}));
  }

  reconcile(room) {
    if (this.skipped.size) this.skipped = new Set(room.entries.filter(e => this.skipped.has(e.id)).map(e => e.id));
    const available = entry => !this.skipped.has(entry.id) && !this.isFinished(entry.id);
    const first = room.entries.find(available) || null;
    // An idle preselection follows the queue. Pause, buffering and a saved seek
    // position do not end an existing performance.
    const idle = !this.state.playing && this.playedAt === null && this.state.position === 0;
    if (!this.entry || (idle && this.entry.id !== first?.id)) this.select(first, this.state.playing);
    else {
      const queued = room.entries.find(entry => entry.id === this.entry.id);
      const watched = room.playback_entry?.job_id === this.entry.job_id ? room.playback_entry : null;
      const updated = queued || watched || room.library?.find(entry => entry.job_id === this.entry.job_id);
      if (!queued && (!this.state.playing || this.state.started_at === null))
        this.select(first, this.state.playing);
      else if (updated) this.entry = {...updated, id:this.entry.id};
    }
    if (this.state.playing && this.entry?.status !== 'completed') {
      const ready = room.entries.find(entry => entry.status === 'completed' && available(entry));
      if (ready) this.select(ready, true);
    }
    return {...room, ...this.state, current_id:this.entry?.id || null};
  }

  select(entry, playing) {
    this.entry = entry;
    // Playing is this browser's intent. An empty queue waits without a clock;
    // finishing the last song must not discard the request to keep singing.
    Object.assign(this.state, {position:0, playing, started_at:null, pitch_semitones:0});
    this.playedAt = null;
  }

  position() {
    const {position, playing, started_at} = this.state;
    return position + (playing && started_at !== null ? Math.max(0, Date.now()/1000 - started_at) : 0);
  }

  markPlayed() {
    if (this.state.playing && this.state.started_at !== null && Date.now()/1000 > this.state.started_at)
      this.playedAt ||= this.state.started_at;
  }

  isFinished(id) { return this.pendingFinishes.some(item => item.entry_id === id); }

  // Entry IDs make finish idempotent on the server. Persist the original command
  // until acknowledged, including across refresh or an audio-owner handoff.
  async flushFinishes(send, accept, owns = () => true) {
    if (this.flushing) return this.flushing;
    this.flushing = (async () => {
      while (this.pendingFinishes.length && owns()) {
        const finished = this.pendingFinishes[0];
        const updated = await send(finished);
        if (!owns()) return;
        this.pendingFinishes = this.pendingFinishes.filter(item => item !== finished);
        accept(updated);
      }
    })();
    try { await this.flushing; } finally { this.flushing = null; }
  }

  command(action, extra, entries, clockOffset = 0) {
    const state = this.state;
    if (extra.entry_id && !['lead','toggle_lead','pitch'].includes(action) && extra.entry_id !== this.entry?.id) return {};
    this.markPlayed();
    let finished;
    if (['next','remove'].includes(action)) {
      if (action === 'next' && this.entry && this.entry.status !== 'completed') this.skipped.add(this.entry.id);
      const target = entries.find(entry => entry.id !== this.entry?.id && !this.skipped.has(entry.id) && !this.isFinished(entry.id));
      if (this.entry && (action === 'remove' || this.entry.status === 'completed'))
        finished = {action:action === 'remove' ? 'remove' : 'finish', entry_id:this.entry.id,
        playback_started_at:this.playedAt === null ? null : this.playedAt + clockOffset};
      if (finished) this.pendingFinishes.push(finished);
      this.select(target || null, action === 'remove' ? state.playing : true);
    } else if (action === 'lead' || action === 'toggle_lead') {
      const level = action === 'toggle_lead' ? (state.levels.lead_vocals > 0 ? 0 : this.leadRestore) : extra.level;
      state.levels.lead_vocals = bounded(level, 0, 1, state.levels.lead_vocals);
      if (state.levels.lead_vocals > 0) this.leadRestore = state.levels.lead_vocals;
    } else if (action === 'pitch') {
      state.pitch_semitones = extra.pitch_semitones;
    } else if (action === 'play') {
      if (!this.entry) this.skipped.clear();
      state.playing = true;
    }
    else if (action === 'pause') Object.assign(state, {position:this.position(), playing:false, started_at:null});
    else if (this.entry) {
      if (action === 'restart') Object.assign(state, {position:0, started_at:null, pitch_semitones:0});
      else if (action === 'seek') Object.assign(state, {position:bounded(extra.position, 0, Math.max(0, (this.entry.duration || 0)-.05), 0), started_at:null});
      else if (action === 'ready' && state.playing && state.started_at === null && this.entry.status === 'completed') {
        this.skipped.clear(); // After the next song starts, skipped entries can be selected again.
        const delay = state.position === 0 ? Math.max(0, -(this.entry.opening_countdown?.start || 0)) : 0;
        state.started_at = Date.now()/1000 + delay;
      } else if (action === 'buffering' && state.playing && state.started_at !== null && extra.playback_started_at === state.started_at) {
        Object.assign(state, {position:extra.position, started_at:null});
      }
    }
    return {finished};
  }

  save(position = this.position()) {
    this.markPlayed();
    return {entry:this.entry, playing:this.state.playing, position:Math.max(0, position), levels:this.state.levels,
      pitch_semitones:this.state.pitch_semitones, lead_restore:this.leadRestore,
      played_at:this.playedAt, pending_finishes:this.pendingFinishes, skipped_ids:[...this.skipped]};
  }
}
