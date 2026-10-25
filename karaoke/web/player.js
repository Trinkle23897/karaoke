// Use one AudioContext clock for all stems and latency-adjusted visuals.
import {createPitchShift} from './pitch-shift.js';
import {StemStream} from './stem-stream.js';

function videoReady(video, signal, position) {
  return new Promise((resolve, reject) => {
    const events = ['loadedmetadata', 'loadeddata', 'canplay', 'seeked'];
    let positioned = false;
    const finish = error => {
      clearTimeout(timer);
      events.forEach(name => video.removeEventListener(name, check));
      video.removeEventListener('error', failed);
      signal.removeEventListener('abort', aborted);
      error ? reject(error) : resolve();
    };
    const check = () => {
      if (video.readyState >= 1 && !positioned) {
        positioned = true;
        if (Math.abs(video.currentTime - position) > .04) video.currentTime = position;
      }
      if (video.readyState >= 2 && !video.seeking) finish();
    };
    const failed = () => finish(new Error('视频加载失败，请重试'));
    const aborted = () => finish();
    const timer = setTimeout(() => finish(new Error('视频加载超时，请重试')), 30000);
    events.forEach(name => video.addEventListener(name, check));
    video.addEventListener('error', failed);
    signal.addEventListener('abort', aborted, {once:true});
    if (signal.aborted) aborted();
    else if (video.error) failed();
    else check();
  });
}

export class Player {
  constructor(video, onTick, onEnd, pitchFactory = createPitchShift) {
    this.video = video;
    video.muted = true;
    video.playsInline = true;
    this.onTick = onTick;
    this.onEnd = onEnd;
    this.sources = [];
    this.volume = 1;
    this.anchor = null;
    this.unlocked = false;
    this.mode = 'karaoke';
    this.levels = {instrumental:1, backing_vocals:1, lead_vocals:0};
    this.pitch = 0;
    this.pitchFactory = pitchFactory;
  }

  scheduleTick(delay) {
    this.tickTimer = setTimeout(() => this.tick(), delay);
  }

  tick() {
    clearTimeout(this.tickTimer);
    this.tickTimer = null;
    if (this.disposed || this.anchor === null || this.context.state !== 'running') return;
    if (this.pendingStart) {
      const pending = this.pendingStart;
      this.onTick(pending.position === 0 ? Math.min(0, this.outputTime() - this.anchor) : pending.position);
    } else {
      this.pump();
      if (this.anchor === null) return;
      const outputTime = this.outputTime();
      const elapsed = outputTime - this.anchor;
      const position = Math.min(this.scheduledUntil,
        this.startPosition === 0 ? elapsed : Math.max(this.startPosition, elapsed));
      if (this.videoPending && outputTime >= this.audibleWhen) {
        this.videoPending = false;
        this.lastVideoTime = this.video.currentTime;
        this.lastVideoAdvance = this.context.currentTime;
        const run = this.videoRun;
        this.video.play().catch(error => {
          if (this.videoRun === run) this.failPlayback(error, position);
        });
      }
      if (position >= this.entry.duration - 0.025) {
        const id = this.entry.id;
        this.stop();
        this.onEnd(id);
      } else {
        this.syncVideo(Math.max(0, position));
        if (this.anchor !== null) this.onTick(position);
      }
    }
    // Audio is already scheduled on its own clock; the baked video needs no
    // per-frame JavaScript. Wake for tenths on the controls and exactly at the
    // audible start, rather than tying media maintenance to display refreshes.
    if (this.anchor !== null) this.scheduleTick(this.videoPending
      ? Math.min(100, Math.max(1, (this.audibleWhen - this.outputTime()) * 1000)) : 100);
  }

  outputTime() {
    // Keep visuals on the clock that actually schedules audio. Some WebKit
    // devices report a fresh getOutputTimestamp() with a nearly frozen time,
    // which would leave the video waiting forever while audio keeps playing.
    const context = this.context;
    return Math.max(0, context.currentTime - (context.baseLatency || 0) - (context.outputLatency || 0));
  }

  syncVideo(position) {
    const video = this.video;
    if (this.videoPending) return;
    // A presentation sample only exists after a suspicious media-clock reading.
    // Healthy native playback must not keep a per-frame observer running.
    const fresh = this.presentedTime != null && this.context.currentTime - this.presentedAt < .25;
    const videoTime = fresh
      ? this.presentedTime : video.currentTime;
    if (videoTime !== this.lastVideoTime) {
      this.lastVideoTime = videoTime;
      this.lastVideoAdvance = this.context.currentTime;
    }
    // readyState can briefly drop while frames still advance. Freeze only for
    // an actual pause/stall, not a momentary change in the decoder's buffer.
    if (video.paused) {
      this.waitForMedia(position);
      return;
    }
    const stalled = this.context.currentTime - this.lastVideoAdvance > 1;
    const drifted = !video.seeking && Math.abs(position - videoTime) > 1;
    if (!stalled && !drifted) {
      if (this.videoFrameRequest != null) video.cancelVideoFrameCallback(this.videoFrameRequest);
      this.videoFrameRequest = null;
      return;
    }
    if (!fresh && video.requestVideoFrameCallback) {
      if (this.videoFrameRequest == null) {
        const run = this.videoRun;
        this.videoCheckStarted = this.context.currentTime;
        this.videoFrameRequest = video.requestVideoFrameCallback((_, metadata) => {
          if (this.videoRun !== run) return;
          this.videoFrameRequest = null;
          this.presentedTime = metadata.mediaTime;
          this.presentedAt = this.context.currentTime;
        });
        return;
      }
      // Off-screen or genuinely stalled video may never deliver a callback.
      // A valid media clock above remains sufficient; otherwise bound the wait.
      if (this.context.currentTime - this.videoCheckStarted < .25) return;
    }
    // Never chase a moving target. Stop and prepare a fixed position only
    // after a frame confirms the mismatch, or the confirmation times out.
    this.waitForMedia(position);
  }

  waitForMedia(position) {
    const startedAt = this.startedAt;
    this.stop();
    this.onTick(position);
    this.onBuffering?.(position, startedAt);
  }

  failPlayback(error, position) {
    this.unlocked = false;
    this.waitForMedia(position);
    this.onError?.(error);
  }

  prepareVideo(position, signal = this.abort.signal) {
    const existing = this.videoPreparation;
    if (existing?.signal === signal && Math.abs(existing.position - position) < .04) return existing.promise;
    const preparation = this.videoPreparation = {position, signal};
    preparation.promise = this.primeVideo(position, signal).finally(() => {
      if (this.videoPreparation === preparation) this.videoPreparation = null;
    });
    return preparation.promise;
  }

  async primeVideo(position, signal) {
    await videoReady(this.video, signal, position);
    if (signal.aborted || (this.videoPrepared != null && this.video.paused &&
        Math.abs(this.videoPrepared - position) < .04)) return;
    let timer;
    try {
      await Promise.race([this.video.play(), new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('视频启动超时，请重试')), 15000);
      })]);
      if (signal.aborted) {
        if (!this.signature) this.video.pause();
        return;
      }
      if (this.video.paused) throw new Error('视频未能开始播放，请重试');
      this.video.pause();
      await videoReady(this.video, signal, position);
      if (!signal.aborted) this.videoPrepared = position;
    } finally { clearTimeout(timer); }
  }

  async startVideo(pending) {
    try {
      if (this.unlocked) await this.resumeAudio();
      if (this.pendingStart !== pending) return;
      // Normal startup primes before the room's ready command. Also protect
      // direct seek/resume callers before scheduling any audio.
      await this.prepareVideo(pending.position, pending.abort.signal);
      if (this.pendingStart !== pending) return;
      if (this.context.state !== 'running') throw new Error('声音尚未启动，请点播放重试');
      this.pendingStart = null;
      this.videoRun = pending;
      this.videoPending = true;
      this.audibleWhen = this.context.currentTime + Math.max(pending.when - this.outputTime(),
        .02 + (this.pitchChain?.latency || 0));
      this.anchor = this.audibleWhen - pending.position;
      this.lastVideoTime = this.video.currentTime;
      this.lastVideoAdvance = this.context.currentTime;
      this.pump();
      this.pitchChain?.start(this.audibleWhen);
      this.tick();
    } catch (error) {
      if (this.pendingStart !== pending) return;
      this.failPlayback(error, pending.position);
    }
  }

  ensureContext() {
    if (this.context) return;
    this.context = new (window.AudioContext || window.webkitAudioContext)();
    this.context.onstatechange = () => {
      if (this.context.state !== 'running' && this.anchor !== null) {
        const position = Math.max(0, Math.min(this.scheduledUntil, this.outputTime() - this.anchor));
        // An output-route/system interruption does not revoke the user's play
        // permission. Freeze both tracks and let the current intent resume them.
        if (this.context.state === 'closed') this.failPlayback(new Error('声音设备已关闭，请刷新播放页'), position);
        else this.waitForMedia(position);
      }
    };
    this.master = this.context.createGain();
    this.master.gain.value = this.volume;
    this.master.connect(this.context.destination);
    this.gains = [0, 1, 2].map(() => {
      const gain = this.context.createGain();
      gain.connect(this.master);
      return gain;
    });
    this.setLevels(this.levels);
  }

  resumeAudio() {
    const context = this.context;
    if (context.state === 'running') return Promise.resolve();
    // resume() may settle before an iOS interruption has ended. Readiness means
    // running, not merely a fulfilled promise. Share one event wait across SSE
    // updates; callers still check their own song/command generation afterward.
    if (!this.resumingAudio) this.resumingAudio = new Promise((resolve, reject) => {
      const finish = error => {
        context.removeEventListener('statechange', changed);
        error ? reject(error) : resolve();
      };
      const changed = () => {
        if (context.state === 'running') finish();
        else if (context.state === 'closed') finish(new Error('声音设备已关闭，请刷新播放页'));
      };
      context.addEventListener('statechange', changed);
      Promise.resolve().then(() => context.resume()).then(changed, finish);
      changed();
    }).finally(() => { this.resumingAudio = null; });
    return this.resumingAudio;
  }

  async unlock() {
    if (this.unlocked && this.context.state === 'running') return;
    // iOS otherwise treats Web Audio as ambient audio, which the ringer switch mutes.
    try { if (navigator.audioSession) navigator.audioSession.type = 'playback'; } catch { /* Optional API. */ }
    this.ensureContext();
    this.unlocked = false;
    await this.context.resume();
    await this.preparePitch();
    this.unlocked = this.context.state === 'running';
    if (!this.unlocked) throw new Error('声音尚未启动，请点播放重试');
  }

  preparePitch() {
    if (!this.pitchLoading) {
      this.pitchLoading = this.pitchFactory(this.context, this.master).then(chain => {
        this.pitchChain = chain;
        for (const gain of this.gains) { gain.disconnect(); gain.connect(chain.input); }
        chain.setPitch(this.pitch);
      }).catch(error => { this.pitchError = error; });
    }
    return this.pitchLoading;
  }

  setPitch(value) {
    this.pitch = Math.max(-6, Math.min(6, Math.round(value || 0)));
    if (this.pitch && this.pitchError) {
      this.stop();
      throw new Error(this.pitchError.message || '升降调初始化失败，请刷新播放页');
    }
    this.pitchChain?.setPitch(this.pitch);
  }

  setVolume(value) {
    this.volume = value;
    if (this.master) this.master.gain.setTargetAtTime(value, this.context.currentTime, 0.02);
  }

  setMix(mode) {
    this.mode = mode;
    this.setLevels({instrumental:1, backing_vocals:mode === 'instrumental' ? 0 : 1, lead_vocals:mode === 'original' ? 1 : 0});
  }

  setLevels(levels) {
    for (const name of ['instrumental', 'backing_vocals', 'lead_vocals']) {
      if (Number.isFinite(levels[name])) this.levels[name] = Math.min(1, Math.max(0, levels[name]));
    }
    if (!this.gains) return;
    const values = ['instrumental', 'backing_vocals', 'lead_vocals'].map(name => this.levels[name]);
    values.forEach((value, i) => this.gains[i].gain.setTargetAtTime(value, this.context.currentTime, 0.02));
  }

  stop() {
    const wasPlaying = this.anchor !== null;
    clearTimeout(this.tickTimer);
    this.tickTimer = null;
    for (const source of this.sources) {
      try { source.stop(); } catch { /* Already ended. */ }
      source.disconnect();
    }
    this.sources = [];
    this.anchor = null;
    this.signature = null;
    this.pendingStart?.abort.abort();
    this.pendingStart = null;
    this.videoRun = null;
    if (this.videoFrameRequest != null) this.video.cancelVideoFrameCallback(this.videoFrameRequest);
    this.videoFrameRequest = null;
    this.presentedTime = null;
    this.videoPending = false;
    if (this.video.playbackRate !== 1) this.video.playbackRate = 1;
    if (wasPlaying) this.pitchChain?.stop();
    this.video.pause();
  }

  suspend() {
    this.stop();
    this.abort?.abort();
    this.ready = false;
    this.videoPrepared = null;
    // pause() alone permits the browser to keep downloading the video.
    if (this.video.src) {
      this.video.removeAttribute('src');
      this.video.load();
    }
  }

  clear() {
    this.suspend();
    this.key = null;
    this.entry = null;
    this.introStart = null;
    this.stream = null;
    this.ready = false;
  }

  dispose() {
    this.disposed = true;
    this.clear();
    this.context?.close();
  }

  matches(entry) {
    return Boolean(entry && this.key === entry.id && this.entry?.job_id === entry.job_id &&
      this.entry.playback_version === entry.playback_version &&
      ['mv', 'accompaniment', 'instrumental', 'backing_vocals', 'lead_vocals'].every(
        name => this.entry.artifacts[name] === entry.artifacts[name]));
  }

  async loadIntro(entry, signal) {
    this.introStart = null;
    try {
      const version = new URL(entry.artifacts.mv, 'http://localhost').searchParams.get('v');
      if (!version) return;
      const response = await fetch(`/v1/jobs/${entry.job_id}`, {signal});
      if (!response.ok) return;
      const job = await response.json();
      const opening = job.result?.quality?.countdown_events?.[0];
      // Use the countdown baked into this exact video, never infer an onset.
      if (signal.aborted || !this.matches(entry) || job.status !== 'completed' ||
          job.result?.artifacts?.mv?.sha256 !== version) return;
      if (opening?.line === 0 && opening.unit === 0 && Number.isFinite(opening.start) &&
          opening.start > 0 && Number.isFinite(opening.onset) &&
          opening.onset > opening.start && opening.onset < entry.duration)
        this.introStart = opening.start;
    } catch { /* Optional metadata must not prevent playback. */ }
  }

  introSkipTarget(position) {
    return Number.isFinite(this.introStart) && Number.isFinite(position) &&
      this.introStart >= 5 && position >= 0 && position < this.introStart ? this.introStart : null;
  }

  async load(entry, position = 0) {
    const generation = this.loadGeneration = (this.loadGeneration || 0) + 1;
    const cached = this.matches(entry) ? this.stream : null;
    if (cached && !cached.signal.aborted) {
      const stream = this.stream;
      try {
        await this.loading;
        if (!stream.signal.aborted) await Promise.all([stream.prepare(position),
          this.anchor === null ? videoReady(this.video, stream.signal, position) : Promise.resolve()]);
      } catch (error) {
        if (stream.signal.aborted) return;
        if (generation === this.loadGeneration) { this.key = null; this.suspend(); }
        throw error;
      }
      return;
    }
    this.stop();
    this.abort?.abort();
    this.abort = new AbortController();
    const {signal} = this.abort;
    this.key = entry.id;
    this.entry = entry;
    this.ready = false;
    this.videoPrepared = null;
    this.ensureContext();
    this.video.preload = 'auto';
    this.video.src = entry.artifacts.mv;
    this.video.load();
    this.loadIntro(entry, signal);
    this.stream = new StemStream(this.context, `/v1/jobs/${entry.job_id}/playback`, signal, () => this.pump(), cached, entry.playback_version);
    this.loading = Promise.all([this.stream.prepare(position), videoReady(this.video, signal, position)]).then(() => {
      if (signal.aborted) return;
      this.ready = true;
    }).catch(error => {
      if (signal.aborted) return;
      this.key = null;
      this.suspend();
      throw error;
    });
    return this.loading;
  }

  apply(state, clockOffset) {
    this.setPitch(state.pitch_semitones || 0);
    if (state.levels) this.setLevels(state.levels);
    else this.setMix(state.mix);
    if (!this.ready || this.entry?.id !== state.current_id) return;
    const position = Math.min(this.entry.duration, state.position +
      (state.playing && state.started_at ? Math.max(0, Date.now()/1000 + clockOffset - state.started_at) : 0));
    if (!state.playing || !state.started_at || !this.unlocked) {
      this.stop();
      // An unopened playback tab may receive another device's playing clock.
      // Keep its preview still instead of seeking on every preparation event.
      const preview = this.unlocked ? position : state.position;
      if (Math.abs(this.video.currentTime - preview) > .04) this.video.currentTime = preview;
      this.onTick(position);
      return;
    }
    const signature = [state.current_id, state.position, state.started_at].join(':');
    if (signature === this.signature) return;
    this.stop();
    if (!this.hasAudio(position)) { this.onBuffering?.(position, state.started_at); return; }
    this.signature = signature;
    this.startedAt = state.started_at;
    const delay = Math.max(0, state.started_at - (Date.now()/1000 + clockOffset));
    this.audibleWhen = this.outputTime() + delay;
    this.startPosition = position;
    this.anchor = this.audibleWhen - position;
    this.nextWindow = this.stream.index(position);
    this.scheduledUntil = position;
    const pending = this.pendingStart = {position, when:this.audibleWhen, abort:new AbortController()};
    this.tick();
    return this.startVideo(pending);
  }

  hasAudio(position) { return this.ready && Boolean(this.stream?.at(position)); }

  pump() {
    if (this.anchor === null || !this.ready || this.pendingStart) return;
    const stream = this.stream;
    const position = Math.max(this.startPosition, this.outputTime() - this.anchor);
    const inputPosition = Math.max(this.startPosition,
      this.context.currentTime - this.anchor + (this.pitchChain?.latency || 0));
    stream.around(inputPosition);
    while (this.nextWindow < stream.count) {
      const audio = stream.windows.get(this.nextWindow)?.audio;
      if (!audio) break;
      const {start, end, decodedStart} = stream.bounds(this.nextWindow++);
      const first = Math.max(start, this.startPosition);
      const when = this.anchor - (this.pitchChain?.latency || 0) + first;
      // Never start a late window partway through a song. Freeze the shared
      // room clock below and resume all stems from the same buffered position.
      if (when < this.context.currentTime - .005) { this.nextWindow--; break; }
      for (const [i, buffer] of audio.entries()) {
        const source = this.context.createBufferSource();
        source.buffer = buffer;
        const name = stream.stems[i];
        source.connect(this.gains[name === 'lead_vocals' ? 2 : name === 'backing_vocals' ? 1 : 0]);
        source.onended = () => {
          source.disconnect();
          this.sources = this.sources.filter(item => item !== source);
          if (stream === this.stream) this.pump();
        };
        source.start(when, first - decodedStart, end - first);
        this.sources.push(source);
      }
      this.scheduledUntil = end;
    }
    if (position >= this.scheduledUntil && this.scheduledUntil < stream.duration) {
      const pausedAt = this.scheduledUntil, startedAt = this.startedAt;
      this.stop();
      this.onTick(pausedAt);
      this.onBuffering?.(pausedAt, startedAt);
    }
  }
}
