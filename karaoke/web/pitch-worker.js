import SignalsmithStretch from './vendor/signalsmith-stretch.mjs';

// A regular worker can run the bundled WASM on LAN HTTP. Only small stereo
// blocks cross this boundary. The player owns the clock and all audio nodes.
self.onmessage = async ({data:{rate, config}}) => {
  try {
    const wasm = await SignalsmithStretch.createModule();
    wasm._main();
    wasm._configure(2, Math.round(rate * config.blockMs / 1000),
      Math.round(rate * config.intervalMs / 1000), config.splitComputation);
    const delay = wasm._inputLatency() + wasm._outputLatency();
    // Two delays of output, plus up to one of history and one of lookahead.
    const size = 4 * delay, base = wasm._setBuffers(2, size);
    const channel = c => new Float32Array(wasm.HEAP8.buffer, base + c * size * 4, size);
    self.onmessage = ({data:{pitch, trim, samples}}) => {
      if (trim) {
        wasm._reset();
        wasm._setTransposeSemitones(pitch, 8000 / rate);
      }
      const length = samples[0].length;
      samples.forEach((input, c) => channel(c).set(input));
      wasm._process(length, length);
      const output = samples.map((input, c) => {
        input.set(channel(c + 2).subarray(trim, length));
        return input.subarray(0, length - trim);
      });
      self.postMessage(output, output.map(ch => ch.buffer));
    };
    self.postMessage(delay);
  } catch {
    self.postMessage(null);
  }
};
