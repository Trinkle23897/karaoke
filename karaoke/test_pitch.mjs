import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {Worker} from 'node:worker_threads';
import {createPitchShift, pitchConfiguration} from './web/pitch-shift.js';
import {StemStream} from './web/stem-stream.js';

async function processor() {
  let Processor, ready;
  const loaded = new Promise(resolve => { ready = resolve; });
  const replies = new Map();
  const context = vm.createContext({
    sampleRate:48000, currentTime:0, WebAssembly, console, atob, setTimeout, clearTimeout, performance,
    AudioWorkletProcessor:class {
      constructor() { this.port = {postMessage:([id, value]) => {
        if (id === 'ready') ready();
        else replies.get(id)?.(value);
      }}; }
    },
    registerProcessor:(name, implementation) => { Processor = implementation; },
  });
  const code = await readFile(new URL('./web/vendor/signalsmith-stretch.mjs', import.meta.url), 'utf8');
  vm.runInContext(code.replace('export default _export;', ''), context);
  const node = new Processor({numberOfOutputs:1, outputChannelCount:[2]});
  await loaded;
  let counter = 0;
  const call = (method, ...args) => new Promise(resolve => {
    const id = counter++; replies.set(id, resolve);
    node.port.onmessage({data:[id, method, ...args]}); replies.delete(id);
  });
  await call('configure', pitchConfiguration);
  await call('schedule', {active:true, semitones:0});
  return {node, context, call};
}

function peakNear(samples, expected) {
  let best = {frequency:0, amplitude:0};
  for (let f = expected-4; f <= expected+4; f += .25) {
    let real=0, imaginary=0;
    for (let i=0; i<samples.length; i++) {
      const weight = .5-.5*Math.cos(2*Math.PI*i/(samples.length-1));
      const angle = 2*Math.PI*f*i/48000;
      real += samples[i]*weight*Math.cos(angle);
      imaginary += samples[i]*weight*Math.sin(angle);
    }
    const amplitude = 4*Math.hypot(real,imaginary)/samples.length;
    if (amplitude > best.amplitude) best={frequency:f, amplitude};
  }
  return best;
}

test('real WASM shifts a stereo three-tone mix dynamically without changing the time scale', async t => {
  const {node, context, call} = await processor();
  const latency = await call('latency');
  assert.ok(latency > 0 && latency < .25);
  let maxCents = 0;
  const tones=[[220,330,550],[261.6256,392,659.2551]];
  const output=[new Float32Array(480000),new Float32Array(480000)];
  let previous;
  for (let first=0; first<output[0].length; first+=128) {
    context.currentTime=first/48000;
    const shift=[0,3,-2,6,-6][Math.floor(first/96000)];
    if (shift !== previous) { await call('schedule', {semitones:shift}); previous=shift; }
    const input=tones.map(frequencies => Float32Array.from({length:128}, (_,i) =>
      frequencies.reduce((sum,f) => sum+.1*Math.sin(2*Math.PI*f*(first+i)/48000),0)));
    const block=[new Float32Array(128),new Float32Array(128)];
    node.process([input],[block],{});
    block.forEach((data,c) => output[c].set(data,first));
  }
  for (const [time,shift] of [[1,0],[3,3],[5,-2],[7,6],[9,-6]]) {
    for (let channel=0;channel<2;channel++) {
      const segment=output[channel].subarray(time*48000,(time+.4)*48000);
      assert.ok(segment.every(Number.isFinite));
      for (const tone of tones[channel]) {
        const expected=tone*2**(shift/12), peak=peakNear(segment,expected);
        const cents = Math.abs(1200*Math.log2(peak.frequency/expected));
        maxCents = Math.max(maxCents, cents);
        assert.ok(cents < 10, `${tone}Hz shifted ${shift}: ${cents.toFixed(2)} cents off`);
        assert.ok(peak.amplitude>.04 && peak.amplitude<.2, `Missing/clipped tone: ${JSON.stringify(peak)}`);
      }
    }
  }
  t.diagnostic(`Latency ${(latency*1000).toFixed(1)} ms; maximum pitch error ${maxCents.toFixed(2)} cents`);
});

test('reported processor delay matches the zero-shift audio timeline', async () => {
  const {node, context, call}=await processor();
  const delay=Math.round(await call('latency')*48000);
  const input=new Float32Array(48000), output=new Float32Array(96000);
  let seed=42;
  for (let i=0;i<input.length;i++) { seed=(Math.imul(seed,1664525)+1013904223)>>>0; input[i]=.2*(seed/2**32-.5); }
  for (let first=0;first<output.length;first+=128) {
    context.currentTime=first/48000;
    const block=new Float32Array(128), rendered=[new Float32Array(128),new Float32Array(128)];
    if (first<input.length) block.set(input.subarray(first,Math.min(first+128,input.length)));
    node.process([[block,block]],[rendered],{});output.set(rendered[0],first);
  }
  let error=0, energy=0;
  for (let i=10000;i<40000;i++) { error+=(input[i]-output[i+delay])**2; energy+=input[i]**2; }
  assert.ok(error/energy<1e-6, `Latency mismatch: relative error ${error/energy}`);
});

test('inactive pitch path skips DSP and resumes processing when enabled', async () => {
  const {node, call}=await processor();
  const original=node.wasmModule._process;
  let calls=0;
  node.wasmModule._process=(...args)=>{calls++;return original(...args);};
  const input=[new Float32Array(128).fill(.1),new Float32Array(128).fill(.2)];
  const output=input.map(a=>a.slice());
  await call('stop');
  for (let i=0;i<100;i++) node.process([input],[output],{});
  assert.equal(calls,0);
  assert.ok(output.every(channel=>channel.every(value=>value===0)));
  await call('configure',pitchConfiguration);
  await call('schedule',{active:true,semitones:3});
  node.process([input],[output],{});
  assert.equal(calls,1);
});

test('real WASM continues producing audio when the output channel count changes', async () => {
  const {node, context}=await processor();
  let frame=0;
  for (const channels of [1,2]) {
    let peak=0;
    for (let block=0;block<100;block++,frame+=128) {
      context.currentTime=frame/48000;
      const input=Array.from({length:channels}, () => Float32Array.from({length:128},
        (_,i)=>.1*Math.sin(2*Math.PI*440*(frame+i)/48000)));
      const output=input.map(()=>new Float32Array(128));
      assert.equal(node.process([input],[output],{}),true);
      for (const channel of output) {
        assert.ok(channel.every(Number.isFinite));
        peak=Math.max(peak,...channel.map(Math.abs));
      }
    }
    assert.ok(peak>.05,'Reconfigured processor must emit the input tone');
  }
});

async function bufferedPlayer(t) {
  const rate=48000, scheduled=[];
  const context={
    sampleRate:rate, currentTime:0,
    createGain:()=>({connect(){}, gain:{value:1, cancelScheduledValues(){},
      setTargetAtTime(v){this.value=v;}, setValueAtTime(v){this.value=v;}}}),
    createBuffer(channels, length) {
      const samples=Array.from({length:channels},()=>new Float32Array(length));
      return {numberOfChannels:channels, length, getChannelData:c=>samples[c],
        copyToChannel:(data,c)=>samples[c].set(data)};
    },
    createBufferSource:()=>({connect(node){this.output=node;}, disconnect(){},
      start(t){this.time=t; scheduled.push(this);}, stop(){this.stopped=true;}}),
  };
  // Run the browser's actual module and transferable PCM in a Node worker.
  const previous=globalThis.Worker;
  globalThis.Worker=class extends Worker {
    constructor(url) {
      super(`const {parentPort, workerData} = require('node:worker_threads');
        globalThis.self = parentPort; import(workerData);`, {eval:true,workerData:url.href});
      this.on('message',data=>this.onmessage({data}));
      this.on('error',error=>this.onerror(error));
    }
  };
  t.after(()=>{globalThis.Worker=previous;});
  const chain=await createPitchShift(context, {});
  t.after(()=>chain.dispose());
  // Two padded 44.1 kHz windows, decoded at the device's 48 kHz. The tone
  // starts exactly at the join. The padding itself must never be heard.
  const stream=Object.assign(Object.create(StemStream.prototype), {
    stems:['accompaniment','lead_vocals'], first:0, windows:new Map(), count:2,
    info:{sample_rate:44100, frames:88200, chunk_frames:44100, padding_frames:882},
  });
  for (let k=0;k<2;k++) {
    const audio=[220,440].map(tone=>{
      const track=context.createBuffer(2,rate*1.02);
      for (let c=0;c<2;c++) {
        const data=track.getChannelData(c);
        data.fill(1);
        for (let i=0;i<rate;i++)
          data[k*960+i]=k*.1*Math.sin(2*Math.PI*(tone+110*c)*i/rate);
      }
      return track;
    });
    stream.windows.set(k,{audio});
  }
  const levels={instrumental:1,lead_vocals:.5};
  async function pump(anchor,start=0) {
    let mix;
    while ((mix=chain.pump(stream,anchor,start,2,levels))?.pending) await mix.pending;
    return mix?.missing;
  }
  return {context,rate,stream,scheduled,chain,levels,pump};
}

test('without a worklet, shift the decoded mix on time across padded windows and discard it on stop', async t => {
  const {context,rate,stream,scheduled,chain,levels,pump}=await bufferedPlayer(t), anchor=.1;
  await pump(anchor);
  assert.equal(scheduled.length,0);
  // Ignore a late result from the position we just left.
  chain.setPitch(-4);
  const previous=chain.prepare(stream,1.1,levels);
  chain.stop();
  chain.setPitch(3);
  await Promise.all([previous,chain.prepare(stream,0,levels)]);
  for (let i=0;i<21;i++) {context.currentTime=i/10; assert.equal(await pump(anchor),undefined);}
  for (let c=0;c<2;c++) {
    let frame=0;
    const output=new Float32Array(2*rate);
    for (const {time,buffer} of scheduled) {
      assert.equal(Math.round((time-anchor)*rate),frame);
      output.set(buffer.getChannelData(c),frame);
      frame+=buffer.length;
    }
    assert.equal(frame,output.length);
    assert.ok(Math.abs(output.findIndex(v=>Math.abs(v)>.04)/rate-1)<.05);
    const segment=output.subarray(1.3*rate,1.9*rate);
    for (const [tone,level] of [[220,1],[440,.5]]) {
      const expected=(tone+110*c)*2**(3/12), peak=peakNear(segment,expected);
      assert.ok(Math.abs(1200*Math.log2(peak.frequency/expected))<10);
      assert.ok(Math.abs(peak.amplitude-.1*level)<.02);
    }
  }
  chain.setPitch(0); await pump(anchor);
  assert.equal(chain.input.gain.value,1);
  chain.stop();
  assert.ok(scheduled.every(source=>source.stopped));
});

test('a late key change resumes at the first unheard sample with the new mix', async t => {
  const {context,rate,stream,scheduled,chain,levels,pump}=await bufferedPlayer(t), anchor=3;
  chain.setPitch(2);
  await pump(anchor);
  assert.equal(scheduled[0].time,anchor);
  const until=scheduled.reduce((frames,source)=>frames+source.buffer.length,0)/rate;
  assert.ok(until<2);
  context.currentTime=anchor+until+.1;
  chain.setPitch(-2);
  levels.lead_vocals=0;
  assert.equal(await pump(anchor),until);
  chain.stop();
  assert.ok(scheduled.every(source=>source.stopped));
  scheduled.length=0;
  stream.first=1;
  stream.windows.delete(0);
  await chain.prepare(stream,until,levels);
  await pump(context.currentTime+.1-until,until);
  assert.ok(Math.abs(scheduled[0].time-context.currentTime-.1)<1/rate);
  const output=scheduled[1].buffer.getChannelData(0);
  const expected=220*2**(-2/12), peak=peakNear(output,expected);
  assert.ok(Math.abs(1200*Math.log2(peak.frequency/expected))<10 && peak.amplitude>.07);
  assert.ok(peakNear(output,2*expected).amplitude<.01);
});

test('rapid key changes keep audio queued while waiting for the last requested key', async t => {
  const {context,stream,scheduled,chain,levels,pump}=await bufferedPlayer(t), anchor=.1;
  chain.setPitch(1);
  await pump(anchor);
  for (let i=0;i<5;i++) {
    context.currentTime=anchor+.62+i*.16;
    chain.setPitch(i+2);
    const mix=chain.pump(stream,anchor,0,2,levels);
    assert.equal(mix?.missing,undefined);
    await mix?.pending;
  }
  assert.equal(await pump(anchor),undefined);
  const expected=220*2**(6/12), peak=peakNear(scheduled.at(-1).buffer.getChannelData(0),expected);
  assert.ok(Math.abs(1200*Math.log2(peak.frequency/expected))<10 && peak.amplitude>.07);
  chain.stop();
});
