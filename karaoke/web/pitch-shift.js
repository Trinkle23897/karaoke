export const pitchConfiguration = {blockMs:160, intervalMs:40, splitComputation:true};

export async function createPitchShift(context, destination) {
  if (!context.audioWorklet) return bufferedPitch(context, destination);
  const {default:SignalsmithStretch} = await import('./vendor/signalsmith-stretch.mjs');
  const processor = await SignalsmithStretch(context);
  await processor.configure(pitchConfiguration);
  const latency = await processor.latency();
  const input = context.createGain(), dry = context.createGain(), wet = context.createGain();
  const delay = context.createDelay(1), gate = context.createGain();
  delay.delayTime.value = latency;
  dry.gain.value = 1; wet.gain.value = 0; gate.gain.value = 0;
  input.connect(delay).connect(dry).connect(gate);
  input.connect(processor).connect(wet).connect(gate);
  gate.connect(destination);
  let pitch = 0;
  let playing = false;
  let active = false;
  let idleTimer;
  const updateProcessor = () => {
    clearTimeout(idleTimer);
    if (playing && pitch) {
      if (!active) processor.configure(pitchConfiguration);
      processor.schedule({semitones:pitch, active:true});
      active = true;
    } else {
      // Let the wet-to-dry crossfade finish before suspending expensive DSP.
      idleTimer = setTimeout(() => { processor.stop(); active = false; }, (latency + .12) * 1000);
    }
  };
  return {
    input, latency,
    setPitch(value) {
      if (value === pitch) return;
      pitch = value;
      if (playing) updateProcessor();
      const when = context.currentTime + latency;
      for (const [node, level] of [[dry, value === 0 ? 1 : 0], [wet, value === 0 ? 0 : 1]]) {
        node.gain.cancelScheduledValues(context.currentTime);
        node.gain.setTargetAtTime(level, when, .015);
      }
    },
    start(when) {
      playing = true;
      updateProcessor();
      gate.gain.cancelScheduledValues(context.currentTime);
      gate.gain.setValueAtTime(0, context.currentTime);
      gate.gain.setValueAtTime(0, when);
      gate.gain.linearRampToValueAtTime(1, when+.005);
    },
    stop() {
      playing = false;
      active = false;
      clearTimeout(idleTimer);
      processor.stop();
      gate.gain.cancelScheduledValues(context.currentTime);
      gate.gain.setTargetAtTime(0, context.currentTime, .005);
      // Clear the previous song/seek's delayed audio before opening the gate.
      processor.configure(pitchConfiguration);
    },
  };
}

// Mix only the frames needed by the worker. Decoded windows have padding
// for resampling. Skip it at each join and leave silence after the last frame.
function readMix(stream, frame, length, until, rate, gains) {
  const channels = [new Float32Array(length), new Float32Array(length)];
  for (let i = 0; i < length && frame + i < until;) {
    const index = stream.index((frame + i) / rate);
    const {decodedStart, end} = stream.bounds(index);
    const offset = frame + i - Math.round(decodedStart * rate);
    const count = Math.min(length - i, Math.round(end * rate) - frame - i);
    stream.windows.get(index).audio.forEach((buffer, stem) => {
      if (!gains[stem]) return;
      channels.forEach((out, c) => {
        const samples = buffer.getChannelData(c % buffer.numberOfChannels).subarray(offset, offset + count);
        if (!stem && gains[stem] === 1) out.set(samples, i);
        else for (let j = 0; j < samples.length; j++) out[i + j] += samples[j] * gains[stem];
      });
    });
    i += count;
  }
  return channels;
}

async function bufferedPitch(context, destination) {
  if (!globalThis.Worker || !globalThis.WebAssembly) throw new Error('当前浏览器不支持升降调，请换用新版浏览器');
  const worker = new Worker(new URL('./pitch-worker.js', import.meta.url), {type:'module'});
  const requests = [];
  let failure;
  function close() {
    failure = new Error('升降调处理失败，请刷新播放页');
    worker.terminate();
    for (const request of requests.splice(0)) request.reject(failure);
  }
  worker.onerror = worker.onmessageerror = close;
  worker.onmessage = ({data}) => data ? requests.shift()?.resolve(data) : close();
  function send(message) {
    return new Promise((resolve, reject) => {
      if (failure) throw failure;
      worker.postMessage(message, message.samples?.map(ch => ch.buffer) || []);
      requests.push({resolve, reject});
    });
  }
  const rate = context.sampleRate, delay = await send({rate, config:pitchConfiguration});
  const input = context.createGain();
  input.connect(destination);
  const sources = new Map();
  let pitch = 0, active = null, queued = null;
  const gainsOf = (stream, levels) => stream.stems.map(name => levels[name === 'accompaniment' ? 'instrumental' : name]);

  async function render(stream, mix, end) {
    const length = Math.min(2 * delay, end - mix.frame);
    const history = Math.min(delay, mix.frame - Math.round(stream.bounds(stream.first).start * rate));
    // Prime a new mix with nearby context. Discard that and the DSP's
    // lookahead, so the returned samples start exactly at mix.frame.
    const trim = mix.reset ? delay + history : 0;
    const samples = readMix(stream, mix.frame + delay - trim, length + trim, end, rate, mix.gains);
    const task = queued = {mix, padded:mix.frame + length + delay > end};
    const channels = await send({pitch:mix.pitch, trim, samples});
    if (queued !== task) return; // Paused, sought or switched songs.
    task.buffer = context.createBuffer(2, length, rate);
    channels.forEach((samples, c) => task.buffer.copyToChannel(samples, c));
  }

  function switchTo(mix, when, starting) {
    function fade(node, level) {
      node.gain.cancelScheduledValues(when);
      if (starting) node.gain.setValueAtTime(level, when);
      else node.gain.setTargetAtTime(level, when, .015);
    }
    fade(active?.output || input, 0);
    if (active) {
      for (const [source, output] of sources) {
        if (output === active.output) source.stop(starting ? context.currentTime : when + .12);
      }
    }
    if (mix) {
      mix.output = context.createGain();
      mix.output.gain.value = 0;
      mix.output.connect(destination);
    }
    fade(mix?.output || input, 1);
    active = mix;
  }

  return {
    input, latency:0,
    setPitch(value) { pitch = value; },
    prepare(stream, start, levels) {
      if (!pitch) return;
      let last = stream.index(start);
      while (stream.windows.get(last + 1)?.audio) last++;
      return render(stream, {pitch, gains:gainsOf(stream, levels), frame:Math.round(start * rate), reset:true},
        Math.round(stream.bounds(last).end * rate));
    },
    pump(stream, anchor, start, until, levels) {
      if (!pitch && !active && !queued) return;
      const now = context.currentTime - anchor, starting = now <= start;
      const end = Math.round(until * rate);
      // A delayed tick must not silently skip unheard samples.
      if (active && active.frame < end && active.frame / rate < now) return {missing:active.frame / rate};
      const changeAt = starting ? Math.round(start * rate) : Math.ceil((now + .02) * rate);
      if (!pitch) {
        queued = null;
        if (active) switchTo(null, anchor + changeAt / rate, starting);
        return;
      }
      const gains = gainsOf(stream, levels);
      const matches = mix => pitch === mix?.pitch && gains.every((gain, i) => gain === mix.gains[i]);
      if (queued) {
        if (!queued.buffer) return;
        const {mix, buffer, padded} = queued;
        queued = null;
        // During a live change the previous mix keeps playing while the
        // worker computes. Join it at the same sample on the audio clock.
        const frame = mix === active ? mix.frame : Math.max(mix.frame, changeAt);
        if ((mix === active || matches(mix)) && frame < mix.frame + buffer.length) {
          const when = anchor + frame / rate;
          if (mix !== active) switchTo(mix, when, starting);
          const source = context.createBufferSource();
          source.buffer = buffer;
          source.connect(mix.output);
          sources.set(source, mix.output);
          source.onended = () => { sources.delete(source); source.disconnect(); };
          source.start(when, (frame - mix.frame) / rate);
          mix.frame += buffer.length;
          mix.reset = padded;
        }
      }
      if (!matches(active)) {
        if (changeAt >= end) return;
        // Rapid taps can supersede several renders in a row. Keep one
        // block of the audible mix in reserve before rendering a change.
        if (active && active.frame < end && active.frame - changeAt < 2 * delay)
          return {pending:render(stream, active, end)};
        if (active) active.reset = true;
        return {pending:render(stream, {pitch, gains, frame:changeAt, reset:true}, end)};
      }
      if (active.frame < end && active.frame / rate < Math.max(start, now) + 1.2)
        return {pending:render(stream, active, end)};
    },
    stop() {
      for (const source of sources.keys()) { source.stop(); source.disconnect(); }
      sources.clear();
      active = queued = null;
      input.gain.cancelScheduledValues(context.currentTime);
      input.gain.setValueAtTime(1, context.currentTime);
    },
    dispose:close,
  };
}
