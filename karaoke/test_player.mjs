import test, {beforeEach} from 'node:test';
import assert from 'node:assert/strict';
import {Player} from './web/player.js';
import {isStaleRoomSnapshot, applyRoomPatch, LocalPlayback, PlaybackTabs, RoomEvents, queueToTop, queueTopCandidates, preparationRetry} from './web/room-state.js';

test('room subscriptions recover closed and silent connections without polling or accepting old events', t => {
  t.mock.timers.enable({apis:['setTimeout']});
  const streams=[], received=[], connected=[];
  class Stream extends EventTarget {
    readyState=0;
    close() { this.readyState=2; }
    room(value) { this.dispatchEvent(new MessageEvent('room', {data:JSON.stringify(value)})); }
  }
  const events=new RoomEvents(value=>received.push(value), value=>connected.push(value), url=>{
    const stream=new Stream(); stream.url=url; streams.push(stream); return stream;
  }, () => 1);
  try {
    events.connect();
    const first=streams[0]; first.readyState=1; first.onopen(); first.room({entries:[]});
    for (let i=0;i<5;i++) {t.mock.timers.tick(15000); first.dispatchEvent(new Event('ping'));}
    events.connect(); assert.equal(streams.length,1);
    first.close(); first.onerror();
    t.mock.timers.tick(1000);
    assert.equal(streams.length,2);
    first.room({entries:['stale']});
    streams[1].room({entries:['new song']});
    assert.deepEqual(received,[{entries:[]},{entries:['new song']}]);
    // A half-open connection can stop delivering data without firing onerror.
    t.mock.timers.tick(45000);
    t.mock.timers.tick(1000);
    assert.equal(streams.length,3); assert.equal(streams[1].readyState,2);
    events.connect('singing job');
    assert.equal(new URL(streams[3].url,'http://localhost').searchParams.get('job_id'),'singing job');
    streams[2].room({entries:['old subscription']});
    assert.equal(received.length,2);
    streams[3].readyState=0; streams[3].onerror();
    t.mock.timers.tick(1000); assert.equal(streams.length,5); // Close native retry and back off ourselves.
    events.connect('singing job','playing entry');
    const playingQuery = stream => new URL(stream.url,'http://localhost').searchParams.get('playing_entry_id');
    assert.equal(playingQuery(streams[5]),'playing entry');
    events.connect('singing job','playing entry');
    assert.equal(streams.length,6);
    t.mock.timers.tick(45000);
    t.mock.timers.tick(1000);
    assert.equal(playingQuery(streams[6]),'playing entry'); // Reconnect retains presence, not a play command.
    events.connect('singing job');
    assert.equal(playingQuery(streams[7]),null);
    events.close(); t.mock.timers.tick(60000);
    assert.equal(streams.length,8); assert.equal(connected.at(-1),false);
  } finally { events.close(); }
});

test('room patches retain unchanged collections and apply progress, removal and reordering', () => {
  const a={id:'a',status:'running'}, b={id:'b',status:'completed'};
  const initial={stream:'session',sequence:4,revision:2,entries:[a,b],library:[a,b],history:[]};
  const progress=applyRoomPatch(initial,{stream:'session',base:4,sequence:5,server_time:100,
    changes:{},collections:{entries:{upsert:[{...a,progress:50}],remove:[]}}});
  assert.equal(progress.library,initial.library);
  assert.equal(progress.history,initial.history);
  assert.equal(progress.entries[1],b);
  assert.equal(progress.entries[0].progress,50);
  assert.equal(initial.entries[0].progress,undefined);
  const removed=applyRoomPatch(progress,{stream:'session',base:5,sequence:9,server_time:101,
    changes:{revision:3},collections:{entries:{upsert:[{id:'c'}],remove:['a'],order:['c','b']}}});
  assert.deepEqual(removed.entries,[{id:'c'},b]);
  assert.equal(removed.revision,3);
  assert.equal(applyRoomPatch(initial,{stream:'other',base:4}),null);
  assert.equal(applyRoomPatch(progress,{stream:'session',base:4}),null);
});

test('stream retry backs off with jitter and a missing patch base forces a full resubscription', t => {
  t.mock.timers.enable({apis:['setTimeout']});
  const streams=[], received=[];
  class Stream extends EventTarget { close() { this.closed=true; } }
  const events=new RoomEvents(value=>received.push(value),()=>{},()=>{
    const stream=new Stream(); streams.push(stream); return stream;
  },()=>0);
  try {
    events.connect();
    const fail=()=>streams.at(-1).onerror();
    fail(); assert.equal(streams[0].closed,true);
    t.mock.timers.tick(499); assert.equal(streams.length,1);
    t.mock.timers.tick(1); assert.equal(streams.length,2);
    fail(); t.mock.timers.tick(999); assert.equal(streams.length,2);
    t.mock.timers.tick(1); assert.equal(streams.length,3);
    streams.at(-1).dispatchEvent(new MessageEvent('room',{data:JSON.stringify({stream:'s',sequence:3,entries:[]})}));
    streams.at(-1).dispatchEvent(new MessageEvent('room_patch',{data:JSON.stringify({stream:'s',sequence:5,base:4})}));
    assert.equal(received.length,1);
    t.mock.timers.tick(500); assert.equal(streams.length,4); // Valid snapshot reset the backoff.
  } finally { events.close(); }
});

test('queue move availability preserves singing pins without constructing an order for every row', () => {
  const entries=['a','b','c','d'].map(id=>({id}));
  assert.deepEqual([...queueTopCandidates(entries)],['b','c','d']);
  assert.deepEqual([...queueTopCandidates(entries,['a','b'])],['d']);
  assert.deepEqual([...queueTopCandidates(entries,['b'])],['a','c','d']);
  assert.deepEqual([...queueTopCandidates(entries,['missing'])],['b','c','d']);
  assert.deepEqual([...queueTopCandidates([],['missing'])],[]);
});

test('a restored page resumes its released session without undoing pause or taking audio from another tab', () => {
  const values=new Map(), storage={getItem:key=>values.get(key),setItem:(key,value)=>values.set(key,value)};
  for (const playing of [true,false]) {
    values.clear();
    const first=new PlaybackTabs(storage,'first'), other=new PlaybackTabs(storage,'other');
    const waiting={entry:null,playing,position:0};
    first.write(waiting,{claim:true});
    assert.equal(first.suspend(waiting),true);
    assert.equal(first.owns(),false);
    assert.equal(first.resume(),true);
    assert.equal(first.owns(),true);
    assert.equal(first.record.playback.playing,playing);
    assert.equal(first.resume(),false);
    first.suspend(waiting);
    other.write({playing:true,position:5},{claim:true});
    assert.equal(first.resume(),false); assert.equal(other.owns(),true);
    // Even a now-closed replacement session must not revive the old one.
    first.write(waiting,{claim:true}); first.suspend(waiting);
    other.write({playing:false,position:10},{claim:true}); other.suspend(other.record.playback);
    assert.equal(first.resume(),false);
    assert.equal(first.read().playback.position,10);
  }
});

test('automatic resume rechecks ownership when another tab claims between storage reads', () => {
  const values=new Map();
  let interleave;
  const storage={getItem(key) {
    const value=values.get(key), action=interleave;
    interleave=null; action?.(); return value;
  },setItem:(key,value)=>values.set(key,value)};
  const first=new PlaybackTabs(storage,'first'), other=new PlaybackTabs(storage,'other');
  first.write({playing:true},{claim:true}); first.suspend(first.record.playback);
  interleave=()=>other.write({playing:true,position:10},{claim:true});
  assert.equal(first.resume(),false);
  assert.equal(other.owns(),true);
  assert.equal(first.read().playback.position,10);
});

test('late HTTP replies cannot undo a newer room command or same-revision progress update', () => {
  const current = {revision:4,server_time:100,entries:[{status:'completed'}]};
  assert.equal(isStaleRoomSnapshot({revision:3,server_time:101},current),true);
  assert.equal(isStaleRoomSnapshot({revision:4,server_time:99,entries:[{status:'running'}]},current),true);
  assert.equal(isStaleRoomSnapshot({revision:4,server_time:101},current),false);
  assert.equal(isStaleRoomSnapshot(current,current),false);
  assert.equal(isStaleRoomSnapshot({revision:5,server_time:99},current),false);
  assert.equal(isStaleRoomSnapshot({revision:0,server_time:100},{revision:0}),false);
});

test('browser tabs share a single audio owner and reject saves from the previous tab', () => {
  const values = new Map(), storage = {getItem:key=>values.get(key), setItem:(key,value)=>values.set(key,value)};
  const first = new PlaybackTabs(storage,'first'), second = new PlaybackTabs(storage,'second');
  const before = {entry:{id:'old',job_id:'old'},position:12,playing:true};
  assert.equal(first.write(before,{claim:true}),true);
  assert.equal(first.owns(),true); assert.equal(second.owns(),false);
  assert.deepEqual(second.read().playback,before);
  const after = {entry:{id:'new',job_id:'new'},position:0,playing:true};
  assert.equal(second.write(after,{claim:true}),true);
  assert.equal(first.write({...before,position:99}),false);
  assert.equal(first.owns(),false); assert.equal(second.owns(),true);
  assert.deepEqual(first.read().playback,after);
  const reopened = new PlaybackTabs(storage,'reopened',before);
  assert.deepEqual(reopened.read().playback,after);
  assert.equal(first.write(before,{release:true}),false);
  assert.equal(second.write(after,{release:true}),true);
  assert.equal(reopened.read().owner,null);
  const phoneValues = new Map(), phoneStorage = {getItem:key=>phoneValues.get(key), setItem:(key,value)=>phoneValues.set(key,value)};
  const anotherBrowser = new PlaybackTabs(phoneStorage,'phone',before);
  assert.deepEqual(anotherBrowser.read().playback,before);
  assert.equal(anotherBrowser.write(before,{claim:true}),true);
  assert.equal(anotherBrowser.owns(),true);
  assert.equal(anotherBrowser.write({...before,position:30,levels:{lead_vocals:0.5}}),true);
  assert.deepEqual(reopened.read().playback,after);
  reopened.write(after,{claim:true});
  assert.equal(anotherBrowser.owns(),true);
  assert.equal(anotherBrowser.read().playback.position,30);
});

test('interleaved tab claims and delayed writes cannot hide an ownership change behind a revision', () => {
  const values = new Map();
  let interleave;
  const storage = {getItem(key) {
    const value = values.get(key), action = interleave;
    interleave = null; action?.();
    return value;
  }, setItem:(key,value)=>values.set(key,value)};
  const first = new PlaybackTabs(storage,'first'), second = new PlaybackTabs(storage,'second');
  first.write({position:5},{claim:true});
  let firstSession, firstChange;
  interleave = () => {
    assert.equal(first.write({position:10},{claim:true}),true);
    firstSession = first.record.session; firstChange = first.record.change;
  };
  assert.equal(second.write({position:20},{claim:true}),true);
  assert.equal(first.record.revision,second.record.revision);
  assert.notEqual(firstChange,second.record.change);
  assert.equal(first.owns(firstSession),false);
  assert.equal(first.write({position:99}),false);
  assert.equal(first.read().playback.position,20);
  const delayed = values.get(first.key);
  first.write({position:25},{claim:true});
  const latestSession = first.record.session;
  values.set(first.key,delayed); // A previously-started lower-revision write lands last.
  assert.equal(first.owns(latestSession),false);
  assert.equal(first.read().owner,'second');
  first.write({position:30},{claim:true});
  assert.equal(first.owns(firstSession),false); // A callback from its old session is fenced too.
  assert.equal(first.owns(),true);
});

test('restored or paused deleted songs return to the queue head instead of the library copy', () => {
  const old={id:'removed',job_id:'old',status:'completed'}, first={id:'head',job_id:'new',status:'queued'};
  for (const playing of [false,true]) {
    const local=new LocalPlayback({entry:old,position:120,playing,pitch_semitones:3});
    local.reconcile({entries:[first],library:[old],playback_entry:old});
    assert.equal(local.entry.id,'head'); assert.equal(local.state.position,0);
    assert.equal(local.state.started_at,null); assert.equal(local.state.pitch_semitones,0);
    assert.equal(local.state.playing,playing);
    local.reconcile({entries:[],library:[old,first]});
    assert.equal(local.entry,null);
  }
});

test('removing the current song preserves pause or continuous-play intent', () => {
  const entries=[{id:'one',job_id:'one',status:'completed'},{id:'two',job_id:'two',status:'completed'}];
  for (const playing of [false,true]) {
    const local=new LocalPlayback(); local.select(entries[0],playing);
    assert.equal(local.command('remove',{entry_id:'one'},entries).finished.entry_id,'one');
    assert.equal(local.entry.id,'two'); assert.equal(local.state.playing,playing);
    local.command('remove',{entry_id:'two'},[entries[1]]);
    assert.equal(local.entry,null); assert.equal(local.state.playing,playing);
  }
});

test('skipping a waiting entry stays local after reload; an explicit removal is replayed as remove', async () => {
  const entries=['one','two'].map(id=>({id,job_id:id,status:'queued'}));
  let local=new LocalPlayback(); local.reconcile({entries});
  local.command('next',{entry_id:'one'},entries);
  local=new LocalPlayback(local.save());
  entries[0].status='completed';
  local.reconcile({entries});
  assert.equal(local.entry.id,'two');
  await local.flushFinishes(()=>assert.fail('a local skip must not delete a queue entry'),()=>{});
  local.command('remove',{entry_id:'two'},entries);
  local=new LocalPlayback(local.save()); const sent=[];
  await local.flushFinishes(async command=>{sent.push(command);return {entries:[entries[0]]};},room=>local.reconcile(room));
  assert.deepEqual(sent,[{action:'remove',entry_id:'two',playback_started_at:null}]);
  local.command('play',{},[entries[0]]); local.reconcile({entries:[entries[0]]});
  assert.equal(local.entry.id,'one');
});

let frame;
class Context extends EventTarget {
  sampleRate = 48000;
  currentTime = 10;
  state = 'running';
  destination = {};
  createGain() { return {gain:{value:0,setTargetAtTime(value){this.value=value;}},connect(target){this.target=target;},disconnect(){this.target=null;}}; }
  createBufferSource() {
    return {playbackRate:{value:1},connect(target){this.target=target;},start(time,offset,duration){this.started=[time,offset];this.length=duration;},stop(){this.stopped=true;},disconnect(){}};
  }
  resume() { return Promise.resolve(); }
  close() { this.state='closed'; return Promise.resolve(); }
  decodeAudioData(data) { return Promise.resolve({duration:20, tag:data}); }
}
globalThis.window = {AudioContext:Context};
const metadata = {sample_rate:48000, channels:2, frames:960000, chunk_frames:960000, padding_frames:256};
const fetchAudio = async url => ({ok:true,json:async()=>metadata,arrayBuffer:async()=>url});
globalThis.fetch = fetchAudio;
const song = {id:'one',job_id:'one',duration:20,artifacts:{mv:'video.mp4',instrumental_preview:'instrumental',backing_vocals_preview:'backing',lead_vocals_preview:'lead'}};
const state = {current_id:'one',position:3,started_at:100,playing:true,mix:'karaoke'};
function setup(onEnd=()=>{}) {
  let position=0, since=10, p;
  const video = Object.assign(new EventTarget(), {readyState:4,load(){},removeAttribute(name){delete this[name];},
    pause(){position=this.currentTime;this.paused=true;},
    play(){since=p.context.currentTime;this.paused=false;return Promise.resolve();}});
  Object.defineProperty(video,'currentTime',{configurable:true,
    get:()=>position+(video.paused ? 0 : (p?.context?.currentTime ?? 10)-since),
    set:value=>{position=value;since=p?.context?.currentTime ?? 10;}});
  p=new Player(video,()=>{},onEnd);
  // Most tests advance the audio clock explicitly. Scheduling itself is
  // exercised with fake timers below, independent of monitor refresh rate.
  p.scheduleTick=()=>{};
  frame=()=>p.tick();
  return p;
}
const clockOffset = () => 100-Date.now()/1000;
test('media maintenance sleeps while idle and preserves a precise start without per-frame work', async t => {
  t.mock.timers.enable({apis:['setTimeout', 'Date'], now:100000});
  const p=setup(), ticks=[], ended=[];
  delete p.scheduleTick;
  p.onTick=value=>ticks.push(value);
  p.onEnd=id=>ended.push(id);
  await p.unlock(); await p.load(song);
  Object.defineProperty(p.context,'currentTime',{get:()=>10+(Date.now()-100000)/1000});
  const advance = milliseconds => {
    for(let remaining=milliseconds; remaining>0;) {
      const step=Math.min(25,remaining); t.mock.timers.tick(step); remaining-=step;
    }
  };
  try {
    advance(1000);
    assert.deepEqual(ticks,[]);
    await p.apply({...state,position:0,started_at:Date.now()/1000+.125},0);
    const sources=[...p.sources], audible=p.audibleWhen;
    advance(100); assert.equal(p.video.paused,true);
    advance(25); assert.equal(p.video.paused,false);
    assert.ok(Math.abs(p.context.currentTime-audible)<1e-8);
    ticks.length=0;
    advance(1000);
    assert.equal(ticks.length,10);
    assert.ok(Math.abs(ticks.at(-1)-1)<1e-8);
    assert.deepEqual(p.sources,sources);
    p.stop(); const pausedTicks=ticks.length;
    advance(1000);
    assert.equal(ticks.length,pausedTicks);
    assert.ok(sources.every(source=>source.stopped));
    assert.deepEqual(ended,[]);
    await p.apply({...state,position:19.5,started_at:Date.now()/1000},0);
    advance(750);
    assert.deepEqual(ended,['one']);
    const finalTicks=ticks.length;
    advance(1000);
    assert.equal(ticks.length,finalTicks);
    assert.deepEqual(ended,['one']);
  } finally {p.dispose();}
});

test('two-track songs load two streams, retain the bed and adjust only the lead', async () => {
  const previous = globalThis.fetch, urls = [];
  globalThis.fetch = async url => {
    urls.push(url);
    return {ok:true, json:async()=>({...metadata, stems:['accompaniment','lead_vocals']}), arrayBuffer:async()=>url};
  };
  const p=setup();
  try {
    await p.unlock();
    const entry={...song,artifacts:{mv:'silent.mp4',accompaniment:'bed.flac',lead_vocals:'lead.flac'}};
    await p.load(entry); await p.apply(state,clockOffset());
    assert.equal(p.sources.length,2);
    assert.equal(p.sources[0].target,p.gains[0]); assert.equal(p.sources[1].target,p.gains[2]);
    assert.ok(!urls.some(url=>url.includes('/backing_vocals/') || url.includes('/instrumental/')));
    const sources=[...p.sources];
    for (const lead of [0,.35,1]) {
      p.setLevels({lead_vocals:lead});
      assert.equal(p.gains[0].gain.value,1); assert.equal(p.gains[2].gain.value,lead);
      assert.deepEqual(p.sources,sources);
    }
    assert.equal(p.matches({...entry,artifacts:{...entry.artifacts,accompaniment:'new.flac'}}),false);
    await p.load(entry,12); await p.apply({...state,position:12,started_at:101},clockOffset());
    assert.ok(p.sources.every(source=>source.started[1]===12));
    p.clear(); assert.equal(p.sources.length,0);
  } finally {p.clear();globalThis.fetch=previous;}
});
// The wall clock and simulated AudioContext advance only when the test asks.
beforeEach(t => t.mock.method(Date, 'now', () => 100_000));

test('two devices play, pause, seek and mix independently across shared queue updates', async () => {
  const room={revision:1,entries:[{...song,status:'completed'},{...song,id:'two',status:'queued'}]};
  const computer=new LocalPlayback(), phone=new LocalPlayback(), desktop=setup(), mobile=setup();
  const snapshot=device=>device.reconcile(room);
  const command=(device,action,extra={})=>device.command(action,extra,room.entries);
  try {
    for (const [device,p] of [[computer,desktop],[phone,mobile]]) {
      snapshot(device); await p.unlock(); await p.load(device.entry);
      command(device,'play'); command(device,'ready'); await p.apply(snapshot(device),0);
    }
    const desktopSources=[...desktop.sources];
    assert.equal(desktopSources.length,3); assert.equal(mobile.sources.length,3);
    command(phone,'seek',{position:9}); command(phone,'ready'); command(phone,'lead',{level:1});
    await mobile.apply(snapshot(phone),0);
    command(phone,'pause'); await mobile.apply(snapshot(phone),0);
    // Even a legacy client changing server transport is only a queue update here.
    Object.assign(room,{revision:2,current_id:'two',playing:false,position:19,levels:{lead_vocals:0}});
    await desktop.apply(snapshot(computer),0);
    assert.equal(computer.state.playing,true); assert.equal(computer.state.position,0);
    assert.equal(computer.state.levels.lead_vocals,0);
    assert.equal(phone.state.position,9); assert.equal(phone.state.levels.lead_vocals,1);
    assert.ok(desktopSources.every(s=>!s.stopped)); assert.deepEqual(desktop.sources,desktopSources);
    assert.equal(mobile.sources.length,0);
    command(phone,'next',{entry_id:'one'});
    assert.equal(phone.entry.id,'two'); assert.equal(phone.state.started_at,null);
    assert.equal(computer.entry.id,'one');
    room.entries[1].status='completed'; snapshot(phone); command(phone,'ready');
    assert.notEqual(phone.state.started_at,null);
  } finally { desktop.clear(); mobile.clear(); }
});

test('finishing on one device does not interrupt the other or resurrect a removed queue item', t => {
  let now=100; t.mock.method(Date,'now',()=>now*1000);
  const room={entries:[{...song,status:'completed'},{...song,id:'two',status:'completed'}]};
  const computer=new LocalPlayback(), phone=new LocalPlayback();
  for (const device of [computer,phone]) {
    device.reconcile(room); device.command('play',{},room.entries); device.command('ready',{},room.entries);
  }
  now+=5;
  const {finished}=phone.command('next',{entry_id:'one'},room.entries);
  assert.deepEqual(finished,{action:'finish',entry_id:'one',playback_started_at:100});
  room.entries.shift();
  computer.reconcile(room); phone.reconcile(room);
  assert.equal(computer.entry.id,'one'); assert.equal(computer.position(),5);
  assert.equal(computer.state.playing,true); assert.equal(phone.entry.id,'two');
  const unchanged=phone.save();
  phone.command('next',{entry_id:'one'},room.entries);
  assert.deepEqual(phone.save(),unchanged);
  phone.command('next',{entry_id:'two'},room.entries);
  phone.reconcile(room); // HTTP finish acknowledgement has not arrived yet.
  assert.equal(phone.entry,null);
  room.entries.push({...song,id:'three'}); phone.reconcile(room);
  assert.equal(phone.entry.id,'three'); assert.equal(phone.state.playing,true);
});

test('queue to top inserts after the playing song without changing either device transport', () => {
  const entries = ['one', 'two', 'three', 'four'].map(id => ({...song,id,status:'completed'}));
  assert.deepEqual(queueToTop(entries,'four'),['four','one','two','three']);
  assert.deepEqual(queueToTop(entries,'four','one'),['one','four','two','three']);
  assert.deepEqual(queueToTop(entries,'four','two'),['two','four','one','three']);
  assert.equal(queueToTop(entries,'one'),null);
  assert.equal(queueToTop(entries,'one','one'),null);
  assert.equal(queueToTop(entries,'two','one'),null);
  assert.equal(queueToTop(entries,'deleted'),null);
  assert.deepEqual(queueToTop(entries,'four','removed-playing-entry'),['four','one','two','three']);
  assert.deepEqual(queueToTop(entries,'four',['one']),['one','four','two','three']);
  assert.deepEqual(queueToTop(entries,'four',['two','one']),['one','two','four','three']);
  assert.equal(queueToTop(entries,'two',['one','two']),null);
  assert.deepEqual(entries.map(e=>e.id),['one','two','three','four']);
  const computer = new LocalPlayback(), phone = new LocalPlayback();
  computer.select(entries[1],true); phone.select(entries[0],true);
  const before = [computer,phone].map(p=>p.save());
  const room = {entries:queueToTop(entries,'four',computer.entry.id).map(id=>entries.find(e=>e.id===id))};
  for (const device of [computer,phone]) device.reconcile(room);
  assert.deepEqual([computer,phone].map(p=>p.save()),before);
  assert.ok(computer.state.playing && phone.state.playing);
  // An idle browser still receives other devices' active entry IDs.
  const orderingPhone = new LocalPlayback();
  const shared = {entries, playing_entry_ids:['one']};
  orderingPhone.reconcile(shared);
  assert.equal(orderingPhone.state.playing,false);
  assert.deepEqual(queueToTop(shared.entries,'three',shared.playing_entry_ids),['one','three','two','four']);
});

test('only idle preselection follows queue order; pause, refresh and seek preserve a performance', t => {
  let now=100; t.mock.method(Date,'now',()=>now*1000);
  const entries = ['one','two','three'].map(id => ({...song,id,status:'completed'}));
  const idle = new LocalPlayback({levels:{lead_vocals:.35}});
  idle.reconcile({entries});
  const singing = new LocalPlayback(idle.save());
  singing.command('play',{},entries); singing.command('ready',{},entries);
  singing.command('pitch',{pitch_semitones:2},entries);
  now+=12;
  const before = singing.save();
  const reordered = queueToTop(entries,'three').map(id => entries.find(e => e.id === id));
  idle.reconcile({entries:reordered}); singing.reconcile({entries:reordered});
  assert.equal(idle.entry.id,'three'); assert.equal(idle.state.position,0);
  assert.equal(idle.state.playing,false); assert.equal(idle.state.pitch_semitones,0);
  assert.equal(idle.state.levels.lead_vocals,.35);
  assert.deepEqual(singing.save(),before);
  singing.command('pause',{},reordered);
  for (const paused of [singing,new LocalPlayback(singing.save())]) {
    paused.reconcile({entries:reordered});
    assert.equal(paused.entry.id,'one'); assert.equal(paused.position(),12);
    assert.equal(paused.state.playing,false); assert.equal(paused.state.pitch_semitones,2);
    paused.command('play',{},reordered); paused.command('ready',{},reordered);
    assert.equal(paused.position(),12);
  }
  const seeked = new LocalPlayback(); seeked.reconcile({entries});
  seeked.command('seek',{position:8},entries); seeked.reconcile({entries:reordered});
  assert.equal(seeked.entry.id,'one'); assert.equal(seeked.position(),8);
  idle.command('play',{},reordered); idle.command('ready',{},reordered);
  assert.equal(idle.entry.id,'three');
  idle.command('next',{},reordered); idle.reconcile({entries:reordered});
  assert.equal(idle.entry.id,'one');
});

test('a ready song behind a preparing head retains its place when paused and resumed', t => {
  let now=100; t.mock.method(Date,'now',()=>now*1000);
  const entries=[{...song,status:'queued'},{...song,id:'ready',status:'completed'}];
  const local=new LocalPlayback(); local.reconcile({entries});
  local.command('play',{},entries); local.reconcile({entries}); local.command('ready',{},entries);
  now+=30;
  local.command('pause',{},entries); local.reconcile({entries});
  assert.equal(local.entry.id,'ready'); assert.equal(local.position(),30);
  local.command('play',{},entries); local.reconcile({entries}); local.command('ready',{},entries);
  now+=2;
  assert.equal(local.entry.id,'ready'); assert.equal(local.position(),32);
});

test('failed finishes survive reload, retry idempotently and do not repeat or skip the next song', async t => {
  let now=100; t.mock.method(Date,'now',()=>now*1000);
  const entries=['one','two','three'].map(id=>({...song,id,status:'completed'}));
  const room={entries:[...entries],history:[]};
  let local=new LocalPlayback(); local.reconcile(room);
  local.command('play',{},entries); local.command('ready',{},entries); now+=5;
  local.command('next',{entry_id:'one'},entries,60);
  const finish={action:'finish',entry_id:'one',playback_started_at:160};
  await assert.rejects(local.flushFinishes(async body=>{
    assert.deepEqual(body,finish); throw new Error('offline');
  },()=>assert.fail('unconfirmed operation')));
  local=new LocalPlayback(JSON.parse(JSON.stringify(local.save()))); local.reconcile(room);
  assert.equal(local.entry.id,'two'); assert.equal(local.pendingFinishes.length,1);
  const send=async body=>{
    const entry=room.entries.find(e=>e.id===body.entry_id);
    if(entry) { room.entries=room.entries.filter(e=>e!==entry); room.history.push(entry.id); }
    return {...room};
  };
  await assert.rejects(local.flushFinishes(async body=>{
    await send(body); throw new Error('response lost');
  },()=>assert.fail('lost response')));
  local.command('ready',{},room.entries); now+=5;
  local.command('next',{entry_id:'two'},room.entries,60);
  assert.equal(local.pendingFinishes.length,2);
  const sent=[];
  await local.flushFinishes(async body=>{sent.push(body); return send(body);},updated=>local.reconcile(updated));
  assert.deepEqual(sent,[finish,{action:'finish',entry_id:'two',playback_started_at:165}]);
  assert.deepEqual(room.history,['one','two']); assert.deepEqual(room.entries.map(e=>e.id),['three']);
  assert.equal(local.entry.id,'three'); assert.equal(local.state.playing,true);
  assert.equal(local.pendingFinishes.length,0);
  assert.equal(new LocalPlayback(local.save()).pendingFinishes.length,0);
});

test('an in-flight finish transfers with the audio session; the old response cannot overwrite its successor', async () => {
  const entries=['one','two','three'].map(id=>({...song,id,status:'completed'}));
  const first=new LocalPlayback(); first.reconcile({entries}); first.command('next',{},entries);
  let resolve, owns=true, accepted=0;
  const pending=first.flushFinishes(()=>new Promise(done=>resolve=done),()=>accepted++,()=>owns);
  const second=new LocalPlayback(JSON.parse(JSON.stringify(first.save())));
  owns=false;
  second.command('next',{},entries);
  resolve({entries:entries.slice(1)}); await pending;
  assert.equal(accepted,0); assert.equal(first.pendingFinishes.length,1);
  assert.equal(second.entry.id,'three'); assert.equal(second.pendingFinishes.length,2);
  const sent=[];
  await second.flushFinishes(async body=>{sent.push(body.entry_id);return {entries:entries.slice(2)};},()=>{});
  assert.deepEqual(sent,['one','two']); assert.equal(second.pendingFinishes.length,0);
});

test('manual preparation actions cover failures and backoff without interrupting an older performance', () => {
  assert.deepEqual(preparationRetry({status:'failed'}),{action:'retry',label:'重试'});
  assert.deepEqual(preparationRetry({status:'queued',next_retry_at:123}),{action:'retry',label:'重试'});
  for (const status of ['completed','queued','running','unprepared']) assert.equal(preparationRetry({status}),null);
  const original={...song,status:'completed'}, local=new LocalPlayback();
  local.select(original,true);
  local.command('ready',{},[original]);
  const before=local.save();
  const replacement={...song,id:'new-entry',job_id:'new-job',status:'queued',artifacts:{}};
  local.reconcile({entries:[replacement],library:[original]});
  assert.deepEqual(local.save(),before);
  assert.equal(local.command('next',{},[replacement]).finished.entry_id,original.id);
  assert.equal(local.entry.id,replacement.id);
});

test('a browser waiting to sing starts the first ready song without arming another browser or overriding pause', () => {
  const pending={...song,status:'queued'}, ready={...song,id:'ready',status:'completed'};
  const local=new LocalPlayback(), other=new LocalPlayback();
  const room={entries:[pending]};
  local.reconcile(room); other.reconcile(room);
  local.command('play',{},room.entries);
  assert.equal(local.state.playing,true);
  assert.equal(local.state.started_at,null);
  room.entries.push(ready);
  local.reconcile(room); other.reconcile(room);
  assert.equal(local.entry.id,'ready'); assert.equal(local.state.playing,true);
  assert.equal(other.state.playing,false);
  assert.equal(other.entry.id,pending.id);
  assert.equal(local.entry.id,'ready');
  local.command('pause',{},room.entries);
  room.entries[0]={...pending,status:'completed'};
  local.reconcile(room);
  assert.equal(local.state.playing,false);
  assert.equal(local.entry.id,pending.id);
});

test('browser refresh restores only its own playback intent, with no clock before audio is ready', t => {
  let now=100; t.mock.method(Date,'now',()=>now*1000);
  const room={entries:[{...song,status:'completed'},{...song,id:'two',status:'completed'}]};
  const computer=new LocalPlayback(); computer.reconcile(room);
  computer.command('next',{entry_id:'one'},room.entries); computer.command('ready',{},room.entries);
  computer.command('lead',{level:1},room.entries); computer.command('pitch',{pitch_semitones:2},room.entries);
  now+=7;
  const restored=new LocalPlayback(JSON.parse(JSON.stringify(computer.save())));
  restored.reconcile({...room,playing:true,position:99,current_id:'one'});
  assert.equal(restored.entry.id,'two'); assert.equal(restored.position(),7);
  assert.deepEqual(restored.state.levels,{instrumental:1,backing_vocals:1,lead_vocals:1}); assert.equal(restored.state.pitch_semitones,2);
  assert.equal(restored.state.playing,true); assert.equal(restored.state.started_at,null);
  assert.equal(restored.command('next',{entry_id:'two'},room.entries).finished.playback_started_at,100);
  const other=new LocalPlayback(); other.reconcile(room);
  assert.equal(other.entry.id,'one'); assert.equal(other.position(),0); assert.equal(other.state.levels.lead_vocals,0);
  computer.command('pause',{},room.entries);
  assert.equal(new LocalPlayback(computer.save()).state.playing,false);
});

test('an empty playback queue keeps waiting for a ready song until this browser pauses', () => {
  const local=new LocalPlayback(), other=new LocalPlayback();
  const room={entries:[{...song,status:'completed'}]};
  local.reconcile(room); local.command('play',{},room.entries);
  local.command('next',{entry_id:song.id},room.entries);
  room.entries=[]; local.reconcile(room);
  assert.equal(local.entry,null); assert.equal(local.state.playing,true);
  assert.equal(local.state.started_at,null); assert.equal(local.position(),0);
  const pending={...song,id:'new-song',status:'queued'};
  room.entries.push(pending); local.reconcile(room); other.reconcile(room);
  assert.equal(local.entry.id,pending.id); assert.equal(local.state.started_at,null);
  room.entries[0]={...pending,status:'completed'};
  local.reconcile(room); other.reconcile(room);
  assert.equal(local.state.playing,true); assert.equal(other.state.playing,false);
  local.command('next',{entry_id:pending.id},room.entries);
  room.entries=[]; local.reconcile(room); local.command('pause',{},room.entries);
  room.entries.push({...pending,id:'after-pause',status:'completed'}); local.reconcile(room);
  assert.equal(local.state.playing,false);
  assert.equal(new LocalPlayback(local.save()).state.playing,false);
});

test('each new performance resets pitch while pause, seek, refresh and other devices preserve it', () => {
  const entries=[{...song,status:'completed'},{...song,id:'two',status:'completed'}];
  const local=new LocalPlayback(), other=new LocalPlayback();
  for (const device of [local,other]) {
    device.reconcile({entries}); device.command('play',{},entries);
    device.command('pitch',{pitch_semitones:3},entries);
  }
  local.command('lead',{level:.65},entries);
  for (const [action,extra] of [['pause',{}],['play',{}],['seek',{position:0}],['ready',{}]]) {
    local.command(action,extra,entries);
    assert.equal(local.state.pitch_semitones,3);
  }
  const restored=new LocalPlayback(local.save()); restored.reconcile({entries});
  assert.equal(restored.state.pitch_semitones,3);
  local.command('next',{entry_id:song.id},entries);
  assert.equal(local.entry.id,'two'); assert.equal(local.state.pitch_semitones,0);
  assert.equal(local.state.levels.lead_vocals,.65);
  local.command('pitch',{pitch_semitones:-2},entries);
  local.command('seek',{position:8},entries);
  local.command('restart',{entry_id:song.id},entries);
  assert.equal(local.state.pitch_semitones,-2); assert.equal(local.state.position,8);
  local.command('restart',{},entries);
  assert.equal(local.state.pitch_semitones,0); assert.equal(local.state.position,0);
  assert.equal(local.state.started_at,null); assert.equal(local.state.playing,true);
  local.command('pitch',{pitch_semitones:2},entries);
  local.command('next',{entry_id:'two'},[...entries,{...song,id:'three',status:'completed'}]);
  assert.equal(local.state.pitch_semitones,0);
  local.command('pitch',{pitch_semitones:4},entries);
  local.select({...song,status:'queued'},true);
  local.reconcile({entries:[{...entries[1],id:'new-request'}]});
  assert.equal(local.entry.id,'new-request'); assert.equal(local.state.pitch_semitones,0);
  other.reconcile({entries}); assert.equal(other.state.pitch_semitones,3);
});

test('lead percentage preserves the mix and playback through adjustment, refresh and song changes', async () => {
  const entries=[{...song,status:'completed'}];
  const local=new LocalPlayback({levels:{instrumental:0,backing_vocals:.2,lead_vocals:.4}}), p=setup();
  local.reconcile({entries}); await p.unlock(); await p.load(local.entry);
  local.command('play',{},entries); local.command('ready',{},entries);
  await p.apply(local.reconcile({entries}),0);
  const sources=[...p.sources], position=p.video.currentTime;
  assert.equal(p.master.gain.value,1);
  assert.deepEqual(p.gains.map(g=>g.gain.value),[1,1,.4]);
  for (const level of [0,.25,.73,1]) {
    local.command('lead',{level},entries); await p.apply(local.reconcile({entries}),0);
    assert.deepEqual(p.gains.map(g=>g.gain.value),[1,1,level]);
    assert.equal(p.master.gain.value,1);
    assert.deepEqual(p.sources,sources); assert.equal(p.video.currentTime,position);
    assert.deepEqual(new LocalPlayback(local.save()).state.levels,local.state.levels);
  }
  for (const [level, expected] of [[-.1,0],[2,1],[.35,.35],[NaN,.35],[undefined,.35]]) {
    local.command('lead',{level},entries);
    assert.equal(local.state.levels.lead_vocals,expected);
  }
  local.command('toggle_lead',{},entries); await p.apply(local.reconcile({entries}),0);
  assert.equal(local.state.levels.lead_vocals,0);
  assert.deepEqual(p.gains.map(g=>g.gain.value),[1,1,0]);
  assert.deepEqual(p.sources,sources); // Muting never reloads or restarts the song.
  const restored=new LocalPlayback(JSON.parse(JSON.stringify(local.save())));
  restored.command('toggle_lead',{},entries);
  assert.equal(restored.state.levels.lead_vocals,.35);
  assert.equal(local.state.levels.lead_vocals,0);
  local.command('next',{entry_id:'one'},[...entries,{...song,id:'two',status:'completed'}]);
  local.command('toggle_lead',{},entries);
  assert.equal(local.state.levels.lead_vocals,.35); // Song changes retain the restore level.
  local.command('lead',{level:0},entries); local.command('lead',{level:.05},entries);
  local.command('toggle_lead',{},entries); local.command('toggle_lead',{},entries);
  assert.equal(local.state.levels.lead_vocals,.05); // +/- replaces the remembered nonzero level.
  local.command('lead',{level:.35},entries);
  assert.deepEqual(local.state.levels,{instrumental:1,backing_vocals:1,lead_vocals:.35});
  assert.equal(new LocalPlayback(local.save()).state.levels.lead_vocals,.35);
  const initial=new LocalPlayback(); initial.command('toggle_lead',{},[]);
  assert.equal(initial.state.levels.lead_vocals,1); // No previous nonzero setting yet.
  p.clear();
});

test('skip intro lands at the baked count-in and seeks all stems without resetting sound controls', async t => {
  const entry={...song,status:'completed',artifacts:{...song.artifacts,mv:'video.mp4?v=first'}};
  const opening={line:0,unit:0,start:12.25,onset:16.25,dots:3,step_seconds:1,settle_seconds:1};
  const job={status:'completed',result:{artifacts:{mv:{sha256:'first'}},quality:{countdown_events:[opening]}}};
  let requests=0;
  t.mock.method(globalThis,'fetch',async url=>{
    if (url==='/v1/jobs/one') {requests++;return {ok:true,json:async()=>job};}
    return fetchAudio(url);
  });
  let now=100; t.mock.method(Date,'now',()=>now*1000);
  const p=setup(), local=new LocalPlayback();
  p.pitchFactory=async()=>({input:{},latency:0,setPitch(){},start(){},stop(){}});
  try {
    local.reconcile({entries:[entry]}); local.command('play',{},[entry]);
    local.command('lead',{level:.35},[entry]); local.command('pitch',{pitch_semitones:2},[entry]);
    await p.unlock(); await p.load(entry);
    local.command('ready',{},[entry]);
    await p.apply({...local.state,current_id:entry.id},0);
    assert.equal(p.introSkipTarget(0),12.25);
    assert.equal(p.introSkipTarget(7.25),12.25);
    assert.equal(p.introSkipTarget(12.24),12.25); // Stay available until the count-in starts.
    for (const position of [-1,12.25,16.25,19,NaN]) assert.equal(p.introSkipTarget(position),null);
    const oldSources=[...p.sources], target=p.introSkipTarget(local.position());
    local.command('seek',{entry_id:entry.id,position:target},[entry]);
    await p.apply({...local.state,current_id:entry.id},0);
    assert.ok(oldSources.every(source=>source.stopped));
    now+=2;
    assert.equal(local.position(),12.25); // Loading the target consumes no count-in time.
    await p.load(entry,target); local.command('ready',{},[entry]);
    await p.apply({...local.state,current_id:entry.id},0);
    assert.equal(p.video.currentTime,target);
    assert.ok(p.sources.every(source=>source.started[1]===target));
    assert.equal(local.state.levels.lead_vocals,.35); assert.equal(local.state.pitch_semitones,2);
    assert.equal(p.introSkipTarget(target),null); assert.equal(requests,1);
    local.command('seek',{entry_id:'old song',position:0},[entry]);
    assert.equal(local.position(),target);
  } finally {p.dispose();}
});

test('intro metadata never blocks playback or leaks from an older song or video revision', async t => {
  const pending=[];
  t.mock.method(globalThis,'fetch',url=>url==='/v1/jobs/one'
    ? new Promise(resolve=>pending.push(resolve)) : fetchAudio(url));
  const entry={...song,artifacts:{...song.artifacts,mv:'video.mp4?v=first'}};
  const job={status:'completed',result:{artifacts:{mv:{sha256:'first'}},
    quality:{countdown_events:[{line:0,unit:0,start:12,onset:16}]}}};
  const p=setup();
  try {
    await p.load(entry); assert.equal(p.ready,true); assert.equal(p.introSkipTarget(0),null);
    await p.load({...entry,artifacts:{...entry.artifacts,mv:'video.mp4?v=repaired'}});
    pending.shift()({ok:true,json:async()=>job}); await settled();
    assert.equal(p.introSkipTarget(0),null);
    pending.shift()({ok:true,json:async()=>job}); await settled();
    assert.equal(p.introSkipTarget(0),null); // Metadata for the wrong video version.
    await p.load(entry); p.clear();
    pending.shift()({ok:true,json:async()=>job}); await settled();
    assert.equal(p.introSkipTarget(0),null);
    await p.load(entry);
    pending.shift()({ok:false}); await settled();
    assert.equal(p.ready,true); assert.equal(p.introSkipTarget(0),null);
  } finally {p.dispose();}
});

test('missing, short or invalid opening cues never skip lyrics or a later instrumental break', async t => {
  let opening;
  t.mock.method(globalThis,'fetch',async url=>url==='/v1/jobs/one'
    ? {ok:true,json:async()=>({status:'completed',result:{artifacts:{mv:{sha256:'first'}},quality:{countdown_events:opening ? [opening] : []}}})}
    : fetchAudio(url));
  for (const cue of [null,{line:0,unit:0,start:-3,onset:1},{line:0,unit:0,start:4.9,onset:8.9},
    {line:3,unit:0,start:12,onset:16},{line:0,unit:1,start:12,onset:16},
    {line:0,unit:0,start:12,onset:24},{line:0,unit:0,start:'12',onset:16}]) {
    opening=cue; const p=setup();
    try {
      await p.load({...song,artifacts:{...song.artifacts,mv:'video.mp4?v=first'}});
      assert.equal(p.introSkipTarget(0),null); assert.equal(p.ready,true);
    } finally {p.dispose();}
  }
});

test('local readiness, count-in and stale buffering never start or rewind the wrong clock', t => {
  let now=100; t.mock.method(Date,'now',()=>now*1000);
  const room={entries:[{...song,status:'completed',opening_countdown:{start:-3}}]};
  const p=new LocalPlayback(); p.reconcile(room); p.command('play',{},room.entries);
  now+=2; assert.equal(p.position(),0); assert.equal(p.state.started_at,null);
  p.command('ready',{},room.entries); assert.equal(p.state.started_at,105);
  now=104; assert.equal(p.position(),0);
  now=109; assert.equal(p.position(),4);
  p.command('buffering',{entry_id:'one',position:3.5,playback_started_at:105},room.entries);
  now=112; assert.equal(p.position(),3.5); assert.equal(p.state.started_at,null);
  p.command('ready',{},room.entries); assert.equal(p.state.started_at,112);
  p.command('buffering',{entry_id:'one',position:1,playback_started_at:105},room.entries);
  assert.equal(p.state.started_at,112); assert.equal(p.position(),3.5);
});

test('preparing playback loads its first frame before starting the clock', async () => {
  const p=setup(); p.video.preload='metadata';
  p.video.load=function () {
    this.readyState=1;
    if (this.preload === 'auto') queueMicrotask(() => {
      this.readyState=3; this.dispatchEvent(new Event('canplay'));
    });
  };
  let loaded=false;
  const pending=p.load(song).then(()=>{loaded=true;});
  try {
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(loaded,true);
    assert.equal(p.video.paused,true);
    assert.equal(p.sources.length,0);
  } finally { p.clear(); await pending; }
});

test('media loading does not wait for an audio processor that needs a user gesture', async () => {
  const p=setup(); p.ensureContext(); p.context.state='suspended';
  let activateProcessor;
  const processor=new Promise(resolve=>{activateProcessor=resolve;});
  p.pitchFactory=()=>processor;
  p.context.resume=async()=>{p.context.state='running';};
  let loaded=false;
  const pending=p.load(song).then(()=>{loaded=true;});
  try {
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(loaded,true);
    await p.apply(state,clockOffset());
    assert.equal(p.sources.length,0);
    let unlocked=false;
    const unlocking=p.unlock().then(()=>{unlocked=true;});
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(unlocked,false);
    activateProcessor({input:{},setPitch(){},stop(){},start(){}});
    await unlocking; await p.apply(state,clockOffset());
    assert.equal(p.sources.length,3);
  } finally { activateProcessor({input:{},setPitch(){},stop(){},start(){}}); await pending; p.clear(); }
});

test('music playback selects the media audio session and requires a running audio context', async () => {
  const session={type:'auto'}, p=setup();
  Object.defineProperty(navigator,'audioSession',{configurable:true,value:session});
  try {
    p.ensureContext(); p.context.state='suspended';
    p.context.resume=async()=>{if (session.type==='playback') p.context.state='running';};
    await p.unlock(); await p.load(song); await p.apply(state,clockOffset());
    assert.equal(p.sources.length,3);
    p.stop(); p.unlocked=false; p.context.state='interrupted'; p.context.resume=async()=>{};
    await assert.rejects(p.unlock(),/声音尚未启动/);
    assert.equal(p.unlocked,false);
    await p.apply(state,clockOffset()); assert.equal(p.sources.length,0);
  } finally { delete navigator.audioSession; p.clear(); }
});

test('transient audio interruptions freeze playback without revoking permission or starting the clock early', async t => {
  t.mock.timers.enable({apis:['Date'],now:100000});
  const p=setup(), waiting=[], errors=[], local=new LocalPlayback();
  const entry={...song,status:'completed'}, room={entries:[entry]};
  local.select(entry,true); local.command('ready',{},room.entries);
  p.onBuffering=(position,startedAt)=>{
    waiting.push(position);
    local.command('buffering',{position,playback_started_at:startedAt},room.entries);
  };
  p.onError=error=>errors.push(error.message);
  try {
    await p.unlock(); await p.load(entry); await p.apply(local.reconcile(room),0);
    let attempts=0;
    // Some browsers fulfill resume() while the external interruption persists.
    p.context.resume=async()=>{attempts++;};
    for (const [index, interrupted] of ['interrupted','suspended','interrupted'].entries()) {
      const position=4+index;
      p.context.currentTime=p.anchor+position;
      const sources=[...p.sources];
      p.context.state=interrupted; p.context.onstatechange();
      assert.equal(p.anchor,null); assert.equal(p.video.paused,true); assert.equal(p.unlocked,true);
      assert.ok(sources.every(s=>s.stopped)); assert.equal(local.state.started_at,null);
      const ready=p.resumeAudio();
      assert.equal(p.resumeAudio(),ready);
      let resumed=false; ready.then(()=>{resumed=true;});
      await new Promise(resolve=>setImmediate(resolve));
      t.mock.timers.tick(3000);
      assert.equal(resumed,false); assert.equal(attempts,index+1);
      assert.equal(local.position(),position); assert.equal(p.sources.length,0);
      p.context.onstatechange(); assert.equal(waiting.length,index+1);
      p.context.state='running'; p.context.dispatchEvent(new Event('statechange'));
      await ready;
      local.command('ready',{},room.entries);
      await p.apply(local.reconcile(room),0);
      assert.equal(p.sources.length,3); assert.equal(p.sources[0].started[1],position);
      p.context.currentTime=p.audibleWhen; p.tick();
      assert.equal(p.video.paused,false);
    }
    assert.deepEqual(errors,[]);
  } finally { p.clear(); }
});

test('late audio recovery cannot undo pause, seek, song replacement, or disposal', async () => {
  for (const action of ['pause','seek','replace','dispose']) {
    const p=setup();
    try {
      await p.unlock(); await p.load(song);
      p.context.state='interrupted';
      const stale=p.apply(state,clockOffset());
      await new Promise(resolve=>setImmediate(resolve));
      assert.equal(p.sources.length,0);
      if (action==='pause') await p.apply({...state,playing:false,started_at:null,position:5},clockOffset());
      if (action==='replace') await p.load({...song,id:'two',job_id:'two'});
      if (action==='dispose') p.dispose();
      let current;
      if (action==='seek') current=p.apply({...state,position:12},clockOffset());
      p.context.state=action==='dispose'?'closed':'running';
      p.context.dispatchEvent(new Event('statechange'));
      await stale; await current;
      if (action==='seek') {
        assert.equal(p.sources.length,3); assert.equal(p.sources[0].started[1],12);
      } else {
        assert.equal(p.sources.length,0,action); assert.equal(p.anchor,null,action);
      }
    } finally { p.clear(); }
  }
});

test('a genuine audio resume rejection remains retryable with a new play gesture', async () => {
  const p=setup(), errors=[];
  p.onError=error=>errors.push(error.message);
  try {
    await p.unlock(); await p.load(song);
    p.context.state='suspended';
    p.context.resume=async()=>{throw new DOMException('User gesture required','NotAllowedError');};
    await p.apply(state,clockOffset());
    assert.equal(p.unlocked,false); assert.equal(p.sources.length,0);
    assert.deepEqual(errors,['User gesture required']);
    p.context.resume=async()=>{p.context.state='running';};
    await p.unlock(); await p.apply(state,clockOffset());
    assert.equal(p.sources.length,3);
  } finally {p.clear();}
});

test('a fresh play gesture can release an automatic resume that is still waiting', async () => {
  const p=setup();
  try {
    await p.unlock(); await p.load(song);
    p.context.state='suspended';
    const ready=p.resumeAudio();
    await new Promise(resolve=>setImmediate(resolve));
    let resumed=false; ready.then(()=>{resumed=true;});
    let gestures=0;
    p.context.resume=async()=>{
      gestures++; p.context.state='running'; p.context.dispatchEvent(new Event('statechange'));
    };
    await p.unlock();
    assert.equal(gestures,1); await ready; assert.equal(resumed,true);
    await p.apply(state,clockOffset()); assert.equal(p.sources.length,3);
  } finally {p.clear();}
});

test('decoded audio cannot start the clock before the video is ready', async () => {
  const p=setup(); await p.unlock(); p.video.readyState=0;
  let loaded=false;
  const pending=p.load(song).then(()=>{loaded=true;});
  await new Promise(resolve=>setImmediate(resolve));
  await p.apply(state,clockOffset());
  assert.equal(loaded,false); assert.equal(p.sources.length,0); assert.equal(p.ready,false);
  p.video.readyState=1; p.video.dispatchEvent(new Event('loadeddata'));
  await new Promise(resolve=>setImmediate(resolve)); assert.equal(loaded,false);
  p.video.readyState=3; p.video.dispatchEvent(new Event('canplay'));
  await pending; await p.apply(state,clockOffset()); assert.equal(p.sources.length,3);
  p.clear();
});

test('a failed or superseded video never releases stale audio', async () => {
  const p=setup(); await p.unlock(); p.video.readyState=0;
  const failed=p.load(song);
  p.video.dispatchEvent(new Event('error'));
  await assert.rejects(failed,/视频加载失败/); assert.equal(p.ready,false);
  const old=p.load(song);
  const current=p.load({...song,id:'two'});
  await old; assert.equal(p.ready,false);
  p.video.readyState=3; p.video.dispatchEvent(new Event('canplay'));
  await current; assert.equal(p.entry.id,'two');
  p.clear();
});

test('native video startup must resolve before scheduling any audio; stale startup cannot restart a song', async () => {
  const p=setup(), ticks=[];
  p.onTick=value=>ticks.push(value);
  await p.unlock(); await p.load(song);
  const play=p.video.play.bind(p.video);
  let release;
  p.video.play=()=>new Promise(resolve=>{release=()=>{play();resolve();};});
  const starting=p.apply(state,clockOffset());
  await settled();
  p.context.currentTime+=3; frame();
  assert.equal(p.sources.length,0); assert.equal(ticks.at(-1),3);
  release(); await starting;
  assert.equal(p.sources.length,3);
  assert.ok(p.sources.every(source=>source.started[1]===3));
  assert.equal(p.video.paused,true); // Hold until the shared audible start.
  const stale=p.apply({...state,position:8},clockOffset());
  await settled();
  p.stop();
  await p.load({...song,id:'two',job_id:'two'});
  release(); await stale;
  assert.equal(p.sources.length,0); assert.equal(p.entry.id,'two');
  assert.equal(p.video.paused,true);
  p.clear();
});

test('seeking waits for the requested frame and does not retarget a slow decoder', async () => {
  const p=setup(); await p.unlock(); await p.load(song);
  let position=0, seeks=0, finished=false;
  Object.defineProperty(p.video,'currentTime',{configurable:true,get:()=>position,
    set:value=>{position=value;seeks++;p.video.seeking=true;}});
  const loading=p.load(song,12).then(()=>{finished=true;});
  await settled();
  for (let i=0;i<5;i++) p.video.dispatchEvent(new Event('canplay'));
  await settled();
  assert.equal(seeks,1); assert.equal(finished,false); assert.equal(p.sources.length,0);
  p.video.seeking=false; p.video.dispatchEvent(new Event('seeked'));
  await loading; await p.apply({...state,position:12},clockOffset());
  assert.equal(seeks,1); assert.equal(p.sources.length,3);
  p.clear();
});

test('video startup rejection or timeout freezes the room and exposes a retryable error', async t => {
  for (const failure of ['rejected','timeout']) {
    const p=setup(), waiting=[], errors=[];
    p.onBuffering=(...args)=>waiting.push(args); p.onError=error=>errors.push(error.message);
    await p.unlock(); await p.load(song);
    if (failure==='timeout') t.mock.timers.enable({apis:['setTimeout']});
    p.video.play=()=>failure==='timeout' ? new Promise(()=>{}) : Promise.reject(new Error('blocked'));
    const starting=p.apply(state,clockOffset());
    await settled();
    if (failure==='timeout') t.mock.timers.tick(15000);
    await starting;
    assert.equal(p.sources.length,0); assert.equal(p.anchor,null); assert.equal(p.unlocked,false);
    assert.deepEqual(waiting,[[3,100]]); assert.equal(errors.length,1);
    t.mock.timers.reset(); p.clear();
  }
});

test('readiness priming is shared across state updates and reused when the room starts', async () => {
  const p=setup(); await p.unlock(); await p.load(song);
  const play=p.video.play.bind(p.video); let calls=0, release;
  p.video.play=()=>{calls++;return new Promise(resolve=>{release=()=>{play();resolve();};});};
  const first=p.prepareVideo(3), repeated=p.prepareVideo(3);
  assert.equal(first,repeated);
  await settled(); assert.equal(calls,1); assert.equal(p.sources.length,0);
  release(); await first;
  await p.apply(state,clockOffset());
  assert.equal(calls,1); assert.equal(p.sources.length,3); assert.equal(p.video.paused,true);
  p.clear();
});

test('stems begin together; mode and volume changes do not restart or seek', async () => {
  const p=setup();await p.unlock();await p.load(song);await p.apply(state,clockOffset());
  assert.equal(p.sources.length,3);
  assert.deepEqual(p.sources.map(s=>s.started),[p.sources[0].started,p.sources[0].started,p.sources[0].started]);
  assert.deepEqual(p.gains.map(g=>g.gain.value),[1,1,0]);
  const originalSources=[...p.sources], originalPosition=p.video.currentTime;
  await p.apply({...state,mix:'instrumental'},clockOffset());p.setVolume(.4);
  assert.deepEqual(p.sources,originalSources);assert.equal(p.video.currentTime,originalPosition);
  assert.deepEqual(p.gains.map(g=>g.gain.value),[1,0,0]);assert.equal(p.master.gain.value,.4);
  await p.apply({...state,mix:'original'},clockOffset());
  assert.deepEqual(p.gains.map(g=>g.gain.value),[1,1,1]);assert.deepEqual(p.sources,originalSources);
  await p.apply({...state,mix:'custom',levels:{instrumental:.8,backing_vocals:.35,lead_vocals:.2}},clockOffset());
  assert.deepEqual(p.gains.map(g=>g.gain.value),[.8,.35,.2]);
  assert.deepEqual(p.sources,originalSources);assert.equal(p.video.currentTime,originalPosition);
  assert.equal(p.master.gain.value,.4);
  p.setLevels({backing_vocals:0});
  assert.deepEqual(p.gains.map(g=>g.gain.value),[.8,0,.2]);
  await p.apply({...state,levels:{instrumental:0,backing_vocals:0,lead_vocals:0}},clockOffset());
  assert.ok(p.gains.every(g=>g.gain.value===0));assert.deepEqual(p.sources,originalSources);
  await p.apply({...state,started_at:98},clockOffset());
  assert.equal(p.video.currentTime,5);
  assert.ok(p.sources.every(s=>s.started[1]===5));
  p.clear();
});

test('opening pre-roll holds video and all stems on a future shared start; pause and seek cancel it', async () => {
  const p=setup(), ticks=[];
  p.onTick=value=>ticks.push(value);
  p.pitchFactory=async()=>({input:{},latency:.12,setPitch(){},start(){},stop(){}});
  await p.unlock(); await p.load(song);
  const start={...state,position:0,started_at:103.369};
  await p.apply(start,clockOffset());
  const sources=[...p.sources], audible=p.audibleWhen;
  assert.ok(Math.abs(audible-13.369)<.01);
  assert.ok(sources.every(s=>Math.abs(s.started[0]+.12-audible)<1e-9 && s.started[1]===0));
  frame(); assert.equal(p.video.paused,true); assert.ok(ticks.at(-1)<-3.3);
  await p.apply({...start,levels:{lead_vocals:.5}},clockOffset());
  assert.deepEqual(p.sources,sources); assert.equal(p.audibleWhen,audible);
  p.context.currentTime=audible-.01; frame(); assert.equal(p.video.paused,true);
  p.context.currentTime=audible; frame(); assert.equal(p.video.paused,false);
  assert.ok(Math.abs(ticks.at(-1))<1e-9); assert.equal(p.video.currentTime,0);
  await p.apply({...start,started_at:107.},clockOffset());
  const waiting=[...p.sources]; frame(); assert.equal(p.video.paused,true);
  await p.apply({...start,playing:false,started_at:null},clockOffset());
  assert.ok(waiting.every(s=>s.stopped)); assert.equal(p.anchor,null); assert.equal(ticks.at(-1),0);
  await p.apply({...state,position:5},clockOffset());
  assert.ok(p.sources.every(s=>Math.abs(s.started[1]-5)<.01));
  await p.load({...song,id:'two'});
  assert.equal(p.anchor,null); assert.equal(p.video.paused,true);
  p.clear();
});

test('seek and pause stop all previous sources; end notification fires once', async () => {
  const ended=[];const p=setup(id=>ended.push(id));await p.unlock();await p.load(song);
  await p.apply(state,clockOffset());const previous=[...p.sources];
  await p.apply({...state,position:12},clockOffset());
  assert.ok(previous.every(s=>s.stopped));assert.ok(p.sources.every(s=>Math.abs(s.started[1]-12)<.01));
  const seeked=[...p.sources];await p.apply({...state,playing:false,started_at:null,position:14},clockOffset());
  assert.ok(seeked.every(s=>s.stopped));assert.equal(p.sources.length,0);assert.equal(p.video.currentTime,14);
  p.video.seeking=true;
  await p.apply({...state,playing:false,started_at:null,position:15},clockOffset());
  assert.equal(p.video.currentTime,15); // A new explicit seek supersedes the old request.
  p.video.seeking=false;
  await p.apply({...state,position:19},clockOffset());p.context.currentTime+=2;
  frame();frame();assert.deepEqual(ended,['one']);assert.equal(p.sources.length,0);
});

test('an idle mobile audio context resumes between songs without a second play gesture', async () => {
  const ended = [], p = setup(id => ended.push(id));
  try {
    await p.unlock(); await p.load(song);
    await p.apply({...state,position:19},clockOffset());
    p.context.currentTime += 2; frame();
    assert.deepEqual(ended,['one']); assert.equal(p.unlocked,true);
    p.context.state = 'suspended'; p.context.onstatechange();
    let resumed = 0;
    p.context.resume = async () => {resumed++; p.context.state='running';};
    const next = {...song,id:'two',job_id:'two'};
    await p.load(next);
    await p.apply({...state,current_id:'two',position:0},clockOffset());
    assert.equal(resumed,1); assert.equal(p.unlocked,true);
    assert.notEqual(p.anchor,null); assert.equal(p.sources.length,3);
    p.context.currentTime += 1; frame();
    assert.equal(p.video.paused,false);
    await p.apply({...state,current_id:'two',playing:false,started_at:null},clockOffset());
    assert.equal(p.anchor,null); assert.equal(p.sources.length,0);
    assert.equal(resumed,1);
  } finally {p.clear();}
});

test('a late download from the previous selection cannot replace current audio', async () => {
  const p=setup();await p.unlock();
  const originalFetch=globalThis.fetch;const pending=[];
  globalThis.fetch=url=>new Promise(resolve=>pending.push(()=>resolve({ok:true,json:async()=>metadata,arrayBuffer:async()=>url})));
  const oldLoad=p.load(song);
  globalThis.fetch=originalFetch;
  await p.load({...song,id:'two',job_id:'two'});
  pending.forEach(resolve=>resolve());await oldLoad;
  assert.equal(p.entry.id,'two');assert.equal(p.stream.at(0)[2].tag,'/v1/jobs/two/playback/lead_vocals/0?sample_rate=48000');
  p.clear();assert.equal(p.entry,null);assert.equal(p.ready,false);
});

test('a repaired asset reloads the same song; metadata updates keep the current media', async () => {
  const p=setup(); await p.unlock(); await p.load(song); await p.apply(state,clockOffset());
  try {
    const initialStream=p.stream, initialSources=[...p.sources];
    const renamed={...song,title:'Updated title',artifacts:{...song.artifacts,poster:'new-cover.jpg'}};
    assert.equal(p.matches(renamed),true);
    await p.load(renamed,3);
    assert.equal(p.stream,initialStream); assert.deepEqual(p.sources,initialSources);
    for (const artifact of ['mv','lead_vocals']) {
      const repaired={...p.entry,artifacts:{...p.entry.artifacts,[artifact]:`${artifact}?v=repaired`}};
      const previous=p.stream, playing=[...p.sources];
      assert.equal(p.matches(repaired),false);
      await p.load(repaired,3);
      assert.ok(previous.signal.aborted); assert.notEqual(p.stream,previous);
      assert.ok(playing.every(source=>source.stopped));
      assert.equal(p.video.src,repaired.artifacts.mv);
      assert.equal(p.matches(repaired),true);
      await p.apply({...state,playing:false,started_at:null},clockOffset());
      assert.equal(p.video.currentTime,3); assert.equal(p.video.paused,true);
      await p.apply(state,clockOffset());
      assert.deepEqual(p.gains.map(g=>g.gain.value),[1,1,0]);
    }
  } finally { p.clear(); }
});

test('manifest and every stem window load the room version, including after a gain-only update', async t => {
  const urls=[];
  t.mock.method(globalThis,'fetch',async url=>{urls.push(new URL(url,'http://localhost'));return fetchAudio(url);});
  const p=setup(); await p.unlock();
  try {
    for (const version of ['first','new-gain']) {
      urls.length=0;
      await p.load({...song,playback_version:version});
      const requested=urls.filter(url=>url.pathname.includes('/playback'));
      assert.equal(requested.length,4);
      assert.ok(requested.every(url=>url.searchParams.get('v')===version && url.searchParams.get('sample_rate')==='48000'));
    }
  } finally { p.clear(); }
});

test('late loading of an old file version cannot replace a repaired version of the same song', async () => {
  const p=setup(), pending=[], originalFetch=globalThis.fetch;
  await p.unlock();
  globalThis.fetch=url=>new Promise(resolve=>pending.push(()=>resolve({ok:true,json:async()=>metadata,arrayBuffer:async()=>url})));
  const old=p.load(song);
  try {
    const stale=p.stream;
    globalThis.fetch=originalFetch;
    const repaired={...song,artifacts:{...song.artifacts,mv:'video.mp4?v=repaired'}};
    await p.load(repaired,3);
    pending.forEach(resolve=>resolve()); await old;
    assert.ok(stale.signal.aborted);
    assert.equal(p.matches(song),false); assert.equal(p.matches(repaired),true);
    assert.equal(p.video.src,repaired.artifacts.mv); assert.equal(p.ready,true);
  } finally { globalThis.fetch=originalFetch; pending.forEach(resolve=>resolve()); p.clear(); }
});

test('live key changes affect the common mix without restarting; video follows audible time', async () => {
  const p=setup(), shifts=[], starts=[];
  p.pitchFactory=async () => ({input:{},latency:.12,setPitch(value){shifts.push(value);},start(when){starts.push(when);},stop(){}});
  await p.unlock();await p.load(song);await p.apply(state,clockOffset());
  const sources=[...p.sources], anchor=p.anchor, initial=p.video.currentTime;
  assert.ok(p.gains.every(g=>g.target===p.pitchChain.input));
  assert.ok(Math.abs(starts[0]-sources[0].started[0]-.12)<1e-9);
  await p.apply({...state,pitch_semitones:3},clockOffset());
  await p.apply({...state,pitch_semitones:-2,levels:{instrumental:.7,backing_vocals:.5,lead_vocals:.3}},clockOffset());
  assert.deepEqual(p.sources,sources);assert.equal(p.anchor,anchor);
  assert.ok(sources.every(s=>s.playbackRate.value===1));
  assert.deepEqual(shifts.slice(-2),[3,-2]);
  frame();assert.equal(p.video.currentTime,initial); // Wait through processing latency.
  p.context.currentTime=starts[0]; frame();
  p.context.currentTime=starts[0]+2;
  frame();assert.ok(Math.abs(p.video.currentTime-initial-2)<1e-6);
  await p.apply({...state,position:8,pitch_semitones:-2},clockOffset());
  assert.ok(sources.every(s=>s.stopped));assert.equal(p.pitch,-2);
  await p.apply({...state,position:8,pitch_semitones:0},clockOffset());
  assert.equal(p.pitch,0);
});

test('video, progress and song end follow device output rather than the ahead-of-time render clock', async t => {
  t.mock.method(performance,'now',()=>100_000);
  for (const timestamp of ['available','starting','unsupported']) {
    const ended=[], ticks=[], p=setup(id=>ended.push(id));
    p.onTick=position=>ticks.push(position);
    p.pitchFactory=async()=>({input:{},latency:.12,setPitch(){},start(){},stop(){}});
    await p.unlock(); await p.load(song);
    p.context.baseLatency=.02; p.context.outputLatency=.16;
    if (timestamp==='available') p.context.getOutputTimestamp=()=>({contextTime:p.context.currentTime-.20,performanceTime:99_980});
    if (timestamp==='starting') p.context.getOutputTimestamp=()=>({contextTime:0,performanceTime:0});
    try {
      await p.apply({...state,position:0},clockOffset());
      const sources=[...p.sources], outputStart=p.audibleWhen;
      p.context.currentTime=outputStart+.1; frame();
      assert.equal(p.video.paused,true,timestamp);
      p.context.currentTime=outputStart+.18; frame();
      assert.equal(p.video.paused,false,timestamp);
      assert.ok(Math.abs(ticks.at(-1))<1e-8,timestamp);
      p.context.currentTime=outputStart+2.18; p.video.currentTime=2; frame();
      assert.equal(p.video.playbackRate,1,timestamp);
      assert.ok(Math.abs(ticks.at(-1)-2)<1e-8,timestamp);
      assert.deepEqual(p.sources,sources);
      p.context.currentTime=outputStart+song.duration; frame();
      assert.deepEqual(ended,[],timestamp);
      p.context.currentTime+=.18; frame(); frame();
      assert.deepEqual(ended,['one'],timestamp);
    } finally { p.clear(); }
  }
});

test('an output timestamp that barely advances cannot hold video behind playing audio', async t => {
  const p=setup(), ticks=[], waiting=[];
  p.onTick=position=>ticks.push(position);
  p.onBuffering=(...args)=>waiting.push(args);
  await p.unlock(); await p.load(song);
  // Actual iPad trace: render time 5.83 -> 19.83 s, timestamp .00983 -> .01015 s.
  p.context.currentTime=5.831111111111111;
  p.context.baseLatency=.0029024943310657597;
  const initial=p.context.currentTime;
  t.mock.method(performance,'now',()=>8089+(p.context.currentTime-initial)*1000);
  p.context.getOutputTimestamp=()=>({
    contextTime:.009834585383662157+(p.context.currentTime-initial)/44100,
    performanceTime:performance.now()-2,
  });
  try {
    await p.apply(state,clockOffset());
    const sources=[...p.sources], start=p.audibleWhen+p.context.baseLatency;
    p.context.currentTime=start; frame();
    assert.equal(p.video.paused,false);
    for (let i=1;i<=840;i++) { p.context.currentTime=start+i/60; frame(); }
    assert.ok(Math.abs(ticks.at(-1)-17)<1e-8);
    assert.ok(Math.abs(p.video.currentTime-17)<1e-8);
    assert.deepEqual(waiting,[]);
    assert.deepEqual(p.sources,sources);
    assert.ok(sources.every(source=>!source.stopped));
  } finally { p.clear(); }
});

test('a brief video readyState drop does not interrupt already scheduled audio', async () => {
  const p=setup(), waiting=[];
  p.onBuffering=(...args)=>waiting.push(args);
  try {
    await p.unlock(); await p.load(song); await p.apply(state,clockOffset());
    p.context.currentTime=p.audibleWhen; frame();
    const sources=[...p.sources];
    p.video.readyState=2;
    for (let i=0;i<15;i++) { p.context.currentTime+=.02; frame(); }
    p.video.readyState=4; p.context.currentTime+=.02; frame();
    assert.deepEqual(waiting,[]);
    assert.notEqual(p.anchor,null);
    assert.ok(sources.every(source=>!source.stopped));
  } finally { p.clear(); }
});

test('a stalled video freezes the shared clock instead of chasing it with seeks', async () => {
  const p=setup(), waiting=[];
  p.onBuffering=(...args)=>waiting.push(args);
  await p.unlock(); await p.load(song); await p.apply(state,clockOffset());
  p.context.currentTime=p.audibleWhen; frame();
  const sources=[...p.sources];
  let seeks=0, videoTime=p.video.currentTime;
  Object.defineProperty(p.video,'currentTime',{configurable:true,get:()=>videoTime,set:value=>{videoTime=value;seeks++;p.video.seeking=true;}});
  p.video.seeking=true;
  for (let i=0;i<120;i++) { p.context.currentTime+=1/60; frame(); }
  assert.equal(seeks,0);
  assert.equal(waiting.length,1); assert.equal(p.anchor,null);
  assert.ok(sources.every(s=>s.stopped));
  const loading=p.load(song,waiting[0][0]);
  await settled(); assert.equal(seeks,1);
  assert.equal(p.sources.length,0);
  p.video.seeking=false; p.video.dispatchEvent(new Event('seeked'));
  await loading;
  await p.apply({...state,position:videoTime,started_at:101},clockOffset());
  assert.equal(p.sources.length,3);
  p.clear();
});

test('only suspicious video clocks request a frame; healthy playback and off-screen clocks need no observer', async () => {
  const p=setup(), waiting=[], callbacks=new Map(); let request=0;
  p.video.requestVideoFrameCallback=callback=>{callbacks.set(++request,callback);return request;};
  p.video.cancelVideoFrameCallback=id=>callbacks.delete(id);
  const present=time=>{const next=callbacks.entries().next().value;if(next){const [id,callback]=next;callbacks.delete(id);callback(0,{mediaTime:time});}};
  p.onBuffering=(...args)=>waiting.push(args);
  try {
    await p.unlock(); await p.load(song); await p.apply(state,clockOffset());
    p.context.currentTime=p.audibleWhen; frame();
    const sources=[...p.sources];
    for(let i=0;i<20;i++) {p.context.currentTime+=.1;frame();}
    assert.equal(request,0);
    // Frames keep advancing, but the media element periodically reports a
    // stale startup timestamp, as seen in the physical iPad trace.
    Object.defineProperty(p.video,'currentTime',{configurable:true,get:()=>.011,set(){}});
    for(let i=0;i<20;i++) {
      p.context.currentTime+=.1;
      frame();
      assert.ok(callbacks.size<=1);
      present(p.outputTime()-p.anchor);
      assert.equal(callbacks.size,0,'a frame sample must not subscribe to the next frame');
    }
    assert.deepEqual(waiting,[]); assert.deepEqual(p.sources,sources);
    assert.ok(sources.every(source=>!source.stopped));
    // Browsing away from the picture may suspend presentation callbacks while
    // the media clock keeps advancing. This must not interrupt the audio.
    Object.defineProperty(p.video,'currentTime',{configurable:true,get:()=>p.outputTime()-p.anchor,set(){}});
    for(let i=0;i<120;i++) {p.context.currentTime+=1/60;frame();}
    assert.deepEqual(waiting,[]); assert.deepEqual(p.sources,sources);
    Object.defineProperty(p.video,'currentTime',{configurable:true,get:()=>.011,set(){}});
    p.context.currentTime+=.3;frame();
    assert.equal(callbacks.size,1);
    // If the native clock recovers before an off-screen callback arrives,
    // cancel that probe instead of reusing its expired deadline later.
    Object.defineProperty(p.video,'currentTime',{configurable:true,get:()=>p.outputTime()-p.anchor,set(){}});
    p.context.currentTime+=.1;frame();
    assert.equal(callbacks.size,0);
    const stalledAt=p.video.currentTime;
    Object.defineProperty(p.video,'currentTime',{configurable:true,get:()=>stalledAt,set(){}});
    for(let i=0;i<16;i++) {p.context.currentTime+=.1;frame();}
    assert.equal(waiting.length,1); assert.equal(p.anchor,null);
    assert.ok(sources.every(source=>source.stopped)); assert.equal(callbacks.size,0);
  } finally {p.clear();}
});

test('video frame callbacks from before a seek or song change cannot replace the current presentation', async () => {
  const p=setup(), callbacks=new Map(); let request=0;
  p.video.requestVideoFrameCallback=callback=>{callbacks.set(++request,callback);return request;};
  p.video.cancelVideoFrameCallback=id=>callbacks.delete(id);
  try {
    await p.unlock(); await p.load(song); await p.apply(state,clockOffset());
    p.context.currentTime=p.audibleWhen; frame();
    p.video.currentTime=.011; frame();
    const stale=callbacks.values().next().value;
    assert.equal(typeof stale,'function');
    await p.apply({...state,position:12},clockOffset());
    p.context.currentTime=p.audibleWhen; frame();
    p.video.currentTime=.011; frame();
    const latest=callbacks.values().next().value;
    assert.equal(typeof latest,'function');
    latest?.(0,{mediaTime:12});
    stale?.(0,{mediaTime:3});
    p.video.currentTime=.011;
    p.context.currentTime=p.audibleWhen; frame();
    assert.notEqual(p.anchor,null); assert.equal(p.video.paused,false);
    await p.load({...song,id:'two',job_id:'two'});
    await p.apply({...state,current_id:'two',position:0},clockOffset());
    latest?.(0,{mediaTime:19});
    p.context.currentTime=p.audibleWhen; frame();
    assert.notEqual(p.anchor,null); assert.equal(p.video.paused,false);
  } finally {p.clear();}
});

test('small video drift leaves native playback uninterrupted; genuine desync still buffers', async () => {
  const p=setup(); await p.unlock(); await p.load(song); await p.apply(state,clockOffset());
  p.context.currentTime=p.audibleWhen; frame();
  const sources=[...p.sources], anchor=p.anchor, waiting=[];
  p.onBuffering=(...args)=>waiting.push(args);
  for(const drift of [.05,-.05,.12,-.12]) {
    p.context.currentTime+=1;
    p.video.currentTime=p.outputTime()-p.anchor-drift;
    const before=p.video.currentTime;
    frame();
    assert.equal(p.video.currentTime,before); assert.equal(p.video.playbackRate,1);
    assert.equal(p.anchor,anchor); assert.deepEqual(p.sources,sources);
  }
  p.video.currentTime=p.outputTime()-p.anchor-1.2; frame();
  assert.equal(waiting.length,1); assert.equal(p.anchor,null);
  assert.ok(sources.every(source=>source.stopped));
  p.clear();
});

test('a WebKit rate-change clock setback cannot cause a repeating video stall', async () => {
  const p=setup(); await p.unlock(); await p.load(song); await p.apply(state,clockOffset());
  p.context.currentTime=p.audibleWhen; frame();
  const sources=[...p.sources];
  // The device trace has ~52 ms stable native skew. A rate change then
  // stalls BOTH presented frames and the video clock by about 250 ms.
  let position=p.video.currentTime-.052, last=p.context.currentTime, rate=1;
  let blockedUntil=last, changes=0;
  const advance=()=>{
    const now=p.context.currentTime;
    if(!p.video.paused) position+=Math.max(0,now-Math.max(last,blockedUntil))*rate;
    last=now;
  };
  Object.defineProperty(p.video,'currentTime',{configurable:true,
    get(){advance();return position;},set(value){advance();position=value;}});
  Object.defineProperty(p.video,'playbackRate',{configurable:true,
    get(){return rate;},set(value){
      advance();
      if(value!==rate){blockedUntil=p.context.currentTime+.25;changes++;}
      rate=value;
    }});
  try {
    let previous=position;
    for(let i=0;i<720;i++) {
      p.context.currentTime+=1/60; frame();
      assert.ok(p.video.currentTime>previous,'buffered video must keep presenting frames');
      assert.ok(Math.abs(p.video.currentTime-(p.outputTime()-p.anchor))<.06);
      previous=p.video.currentTime;
    }
    assert.equal(changes,0);
    assert.ok(sources.every(source=>!source.stopped));
    assert.deepEqual(p.sources,sources);
  } finally {p.clear();}
});

test('an unopened playback tab does not seek through another device\'s playing clock', async () => {
  const p=setup(); await p.load(song);
  let seeks=0, videoTime=0;
  Object.defineProperty(p.video,'currentTime',{get:()=>videoTime,set:value=>{videoTime=value;seeks++;}});
  await p.apply(state,clockOffset());
  for (let i=1;i<=20;i++) await p.apply(state,clockOffset()+i);
  assert.equal(seeks,1); assert.equal(videoTime,state.position); assert.equal(p.sources.length,0);
  await p.apply({...state,position:12},clockOffset()); assert.equal(videoTime,12);
  p.clear();
});

const settled = () => new Promise(resolve => setImmediate(resolve));
function streamingFetch(intercept = () => null) {
  const info = {...metadata, frames:48000*80, chunk_frames:48000*8};
  return async (url, options) => intercept(url.split('?')[0], options) || {
    ok:true, json:async()=>info, arrayBuffer:async()=>url,
  };
}

test('audio lookahead survives a ten-second delayed segment without pausing', async () => {
  let elapsed=0;
  const pending=[], waiting=[];
  globalThis.fetch=streamingFetch(url=>url.endsWith('/lead_vocals/2') ? new Promise(resolve=>{
    pending.push({due:elapsed+10,resolve:()=>resolve({ok:true,arrayBuffer:async()=>url})});
  }) : null);
  const p=setup(); p.onBuffering=(...args)=>waiting.push(args);
  try {
    await p.unlock(); await p.load({...song,duration:80}); await settled();
    await p.apply({...state,position:0},clockOffset());
    const anchor=p.anchor;
    for (elapsed=0;elapsed<32;elapsed+=.25) {
      for (const item of pending.splice(0)) {
        if (item.due<=elapsed) item.resolve(); else pending.push(item);
      }
      await settled();
      p.context.currentTime=anchor+elapsed;
      for (const source of [...p.sources]) if (source.started[0]+source.length<=p.context.currentTime) source.onended();
      frame(); await settled();
      if (p.anchor===null) break;
    }
    assert.deepEqual(waiting,[]);
    assert.notEqual(p.anchor,null);
  } finally { for (const item of pending) item.resolve(); p.clear(); globalThis.fetch=fetchAudio; }
});

test('prefetch prioritizes the current window and bounds concurrent network requests', async () => {
  const pending=[];
  globalThis.fetch=streamingFetch(url=>/\/(instrumental|backing_vocals|lead_vocals)\/\d+$/.test(url) ?
    new Promise(resolve=>pending.push({url,resolve:()=>resolve({ok:true,arrayBuffer:async()=>url})})) : null);
  const p=setup();
  try {
    const loading=p.load({...song,duration:80}); await settled();
    assert.equal(pending.length,3);
    assert.ok(pending.every(item=>item.url.endsWith('/0')));
    for (const item of pending.splice(0)) item.resolve();
    await loading;
    assert.equal(pending.length,3);
    assert.ok(pending.every(item=>item.url.endsWith('/1')));
    for (const item of pending.splice(0)) item.resolve();
    await settled();
    assert.equal(pending.length,3);
    assert.ok(pending.every(item=>item.url.endsWith('/2')));
  } finally { p.clear(); for (const item of pending) item.resolve(); globalThis.fetch=fetchAudio; }
});

test('audio downloads stay near playback and fetched audio is reused on rewind', async () => {
  const requests=[];
  globalThis.fetch=streamingFetch(url=>{requests.push(url);return null;});
  const p=setup();
  try {
    await p.unlock(); await p.load({...song,duration:80}); await settled();
    assert.equal(requests.filter(url=>/\/(instrumental|backing_vocals|lead_vocals)\/\d+$/.test(url)).length,12);
    assert.ok(!requests.includes('/v1/jobs/one/playback/lead_vocals/9'));
    assert.equal(p.sources.length,0);
    await p.apply({...state,position:0},clockOffset());
    assert.equal(p.sources.length,12);
    const first=p.sources.slice(0,3), next=p.sources.slice(3,6);
    assert.ok(first.every(s=>s.length===8 && s.started[1]===0));
    assert.ok(next.every(s=>Math.abs(s.started[0]-first[0].started[0]-8)<1e-9));
    assert.ok(next.every(s=>Math.abs(s.started[1]-256/48000)<1e-9));
    p.context.currentTime=p.audibleWhen; frame();
    for (let position=1;position<79;position++) {
      p.context.currentTime=p.anchor+position;
      for (const source of [...p.sources]) {
        if (source.started[0]+source.length<=p.context.currentTime) source.onended();
      }
      frame(); await settled();
      assert.notEqual(p.anchor,null);
      assert.ok(p.stream.windows.size<=4);
      assert.ok(p.sources.length<=12);
      assert.ok([...p.stream.windows.keys()].every(i=>i>=Math.floor(position/8)));
    }
    const fetched=requests.length;
    p.stop();
    await p.load({...song,duration:80},0); await settled();
    await p.apply({...state,position:0},clockOffset());
    assert.ok(p.hasAudio(0));
    assert.equal(requests.length,fetched);
    assert.ok(p.stream.windows.size<=4);
  } finally { p.clear(); globalThis.fetch=fetchAudio; }
});

test('pausing cancels media transfers and retry timers; resume reuses completed audio', async t => {
  t.mock.timers.enable({apis:['setTimeout']});
  const requests=[], pending=[];
  let attempts=0;
  globalThis.fetch=streamingFetch((url,{signal})=>{
    requests.push(url);
    if (url.endsWith('/lead_vocals/4')) return new Promise((resolve,reject)=>{
      pending.push({signal,resolve});
      signal.addEventListener('abort',()=>reject(new Error('cancelled')),{once:true});
    });
    if (url.endsWith('/backing_vocals/4')) {
      attempts++;
      return {ok:false,status:503,headers:{get:()=> '2'}};
    }
    return null;
  });
  const p=setup();
  try {
    await p.unlock(); await p.load({...song,duration:80}); await settled();
    await p.apply({...state,position:0},clockOffset());
    p.context.currentTime=p.anchor+8; frame(); await settled();
    assert.equal(pending.length,1); assert.equal(attempts,1);
    const previous=p.stream;
    p.suspend(); await settled();
    assert.equal(pending[0].signal.aborted,true);
    assert.equal(p.video.src,undefined); assert.equal(p.video.paused,true);
    assert.equal(p.ready,false); assert.equal(p.anchor,null);
    const count=requests.length;
    t.mock.timers.tick(10000); await settled();
    assert.equal(requests.length,count); assert.equal(attempts,1);
    globalThis.fetch=streamingFetch(url=>{requests.push(url);return null;});
    await p.load({...song,duration:80},8); await settled();
    await p.apply({...state,position:8},clockOffset());
    assert.notEqual(p.stream,previous); assert.equal(p.ready,true);
    assert.equal(p.video.src,song.artifacts.mv);
    assert.ok(p.sources.length>0);
    assert.ok(requests.slice(count).every(url=>url.endsWith('/4')));
    assert.equal(p.stream.at(8),previous.at(8));
  } finally { p.clear(); globalThis.fetch=fetchAudio; }
});

test('failed audio loading releases the native video transfer too', async () => {
  const p=setup();
  globalThis.fetch=async()=>{throw new Error('Audio metadata failed');};
  try {
    await assert.rejects(p.load(song), /Audio metadata failed/);
    assert.equal(p.abort.signal.aborted,true);
    assert.equal(p.video.src,undefined);
    assert.equal(p.ready,false);
    p.clear();
    assert.equal(p.video.src,undefined);
  } finally { p.clear();globalThis.fetch=fetchAudio; }
});

test('one delayed stem freezes all audio at the last common boundary and a seek can resume', async () => {
  let release;
  globalThis.fetch=streamingFetch(url=>url.endsWith('/lead_vocals/1') ? new Promise(resolve=>{release=()=>resolve({ok:true,arrayBuffer:async()=>url});}) : null);
  const p=setup(), waiting=[];
  p.onBuffering=(...args)=>waiting.push(args);
  try {
    await p.unlock(); await p.load({...song,duration:80}); await settled();
    await p.apply({...state,position:0},clockOffset());
    assert.equal(p.sources.length,3);
    const previous=[...p.sources];
    p.context.currentTime=p.anchor+8.1; frame(); frame();
    assert.equal(p.anchor,null); assert.deepEqual(waiting,[[8,100]]);
    assert.ok(previous.every(s=>s.stopped));
    await p.load({...song,duration:80},60);
    await p.apply({...state,position:60},clockOffset());
    const resumed=p.sources.slice(0,3);
    assert.ok(resumed.every(s=>Math.abs(s.started[1]-(4+256/48000))<1e-8));
    release(); await settled();
    assert.deepEqual([...p.stream.windows.keys()],[7,8,9]);
    assert.ok(p.sources.every(s=>!s.buffer.tag.endsWith('/1')));
  } finally { release?.(); p.clear(); globalThis.fetch=fetchAudio; }
});

test('a missing window drains already rendered audio before reporting buffering', async () => {
  let release;
  globalThis.fetch=streamingFetch(url=>url.endsWith('/lead_vocals/1') ? new Promise(resolve=>{release=()=>resolve({ok:true,arrayBuffer:async()=>url});}) : null);
  const p=setup(), waiting=[];
  p.onBuffering=(...args)=>waiting.push(args);
  try {
    await p.unlock(); await p.load({...song,duration:80}); await settled();
    p.context.baseLatency=.02; p.context.outputLatency=.16;
    await p.apply({...state,position:0},clockOffset());
    const anchor=p.anchor;
    p.context.currentTime=p.audibleWhen+.18; frame();
    p.context.currentTime=anchor+8.1; frame();
    assert.deepEqual(waiting,[]);
    assert.notEqual(p.anchor,null);
    p.context.currentTime=anchor+8.19; frame(); frame();
    assert.deepEqual(waiting,[[8,100]]);
    assert.equal(p.anchor,null);
  } finally { release?.(); p.clear(); globalThis.fetch=fetchAudio; }
});

test('a failed prefetch does not retry in a loop or interrupt the buffered window', async () => {
  let failures=0;
  globalThis.fetch=streamingFetch(url=>url.endsWith('/lead_vocals/1') ? (failures++, {ok:false}) : null);
  const p=setup(), waiting=[];
  p.onBuffering=(...args)=>waiting.push(args);
  try {
    await p.unlock(); await p.load({...song,duration:80}); await settled();
    await p.apply({...state,position:0},clockOffset());
    for (let i=0;i<10;i++) { p.context.currentTime+=.1; frame(); }
    assert.equal(failures,1); assert.equal(waiting.length,0);
    p.context.currentTime=p.anchor+8.01; frame();
    assert.deepEqual(waiting,[[8,100]]);
    await assert.rejects(p.load({...song,duration:80},8),/音轨加载失败/);
    assert.equal(p.key,null);
    globalThis.fetch=streamingFetch();
    await p.load({...song,duration:80},8);
    assert.ok(p.hasAudio(8));
  } finally { p.clear(); globalThis.fetch=fetchAudio; }
});

test('transient HTTP and body failures recover inside the audio reserve', async t => {
  t.mock.timers.enable({apis:['setTimeout']});
  let attempts=0;
  globalThis.fetch=streamingFetch(url=>{
    if (!url.endsWith('/lead_vocals/1')) return null;
    attempts++;
    if (attempts===1) return {ok:false,status:503};
    return {ok:true,arrayBuffer:async()=>{
      if (attempts===2) throw new TypeError('network connection lost while reading body');
      return url;
    }};
  });
  const p=setup(), waiting=[]; p.onBuffering=(...args)=>waiting.push(args);
  try {
    await p.unlock(); await p.load({...song,duration:80}); await settled();
    await p.apply({...state,position:0},clockOffset());
    const first=[...p.sources];
    p.context.currentTime=p.audibleWhen; frame();
    assert.equal(attempts,1);
    t.mock.timers.tick(250); await settled(); assert.equal(attempts,2);
    t.mock.timers.tick(1000); await settled(); assert.equal(attempts,3);
    assert.ok(p.stream.windows.get(1).audio);
    p.context.currentTime=p.anchor+8.1; frame();
    assert.deepEqual(waiting,[]);
    assert.ok(first.every(source=>!source.stopped));
  } finally { p.clear(); globalThis.fetch=fetchAudio; }
});

test('audio retry respects Retry-After and changing songs cancels the obsolete retry', async t => {
  t.mock.timers.enable({apis:['setTimeout']});
  let attempts=0;
  globalThis.fetch=streamingFetch(url=>url==='/v1/jobs/one/playback/lead_vocals/1' ?
    (attempts++, {ok:false,status:429,headers:{get:()=> '2'}}) : null);
  const p=setup();
  try {
    await p.load({...song,duration:80}); await settled();
    t.mock.timers.tick(1999); await settled(); assert.equal(attempts,1);
    t.mock.timers.tick(1); await settled(); assert.equal(attempts,2);
    await p.load({...song,id:'two',job_id:'two',duration:80},60);
    t.mock.timers.tick(2000); await settled(); assert.equal(attempts,2);
    assert.ok(p.hasAudio(60));
  } finally { p.clear(); globalThis.fetch=fetchAudio; }
});

test('persistent server failures exhaust a bounded retry without a request storm', async t => {
  t.mock.timers.enable({apis:['setTimeout']});
  let attempts=0;
  globalThis.fetch=streamingFetch(url=>url.endsWith('/lead_vocals/1') ?
    (attempts++, {ok:false,status:503}) : null);
  const p=setup();
  try {
    await p.unlock(); await p.load({...song,duration:80}); await settled();
    await p.apply({...state,position:0},clockOffset());
    t.mock.timers.tick(250); await settled();
    t.mock.timers.tick(1000); await settled();
    for(let i=0;i<30;i++) {p.context.currentTime+=.1;frame();}
    t.mock.timers.tick(10000); await settled();
    assert.equal(attempts,3); assert.ok(p.stream.windows.get(1).error);
  } finally { p.clear(); globalThis.fetch=fetchAudio; }
});

test('an audio fetch that never finishes times out and retries without stopping buffered audio', async t => {
  t.mock.timers.enable({apis:['setTimeout']});
  let attempts=0;
  globalThis.fetch=streamingFetch((url,{signal})=>{
    if (!url.endsWith('/lead_vocals/3') || ++attempts>1) return null;
    return new Promise((resolve,reject)=>signal.addEventListener('abort',()=>reject(signal.reason),{once:true}));
  });
  const p=setup(), waiting=[]; p.onBuffering=(...args)=>waiting.push(args);
  try {
    await p.unlock(); await p.load({...song,duration:80}); await settled();
    await p.apply({...state,position:0},clockOffset());
    t.mock.timers.tick(20000); await settled();
    t.mock.timers.tick(250); await settled();
    assert.equal(attempts,2);
    assert.ok(await p.stream.encoded.get(3));
    assert.deepEqual(waiting,[]);
    assert.notEqual(p.anchor,null);
  } finally { p.clear(); globalThis.fetch=fetchAudio; }
});

for (const outcome of ['recovers', 'stays stuck', 'cancelled']) {
  test(`a stalled audio decoder ${outcome} without hanging the stream`, async t => {
    t.mock.timers.enable({apis:['setTimeout']});
    const pending=[], requests=[];
    let attempts=0;
    globalThis.fetch=streamingFetch(url=>{requests.push(url);return null;});
    const p=setup(), waiting=[];
    p.onBuffering=(...args)=>waiting.push(args);
    p.ensureContext();
    p.context.decodeAudioData=data=>{
      if (data.includes('/lead_vocals/3?') && (++attempts===1 || outcome!=='recovers'))
        return new Promise(resolve=>pending.push(()=>resolve({duration:8,tag:data})));
      return Promise.resolve({duration:8,tag:data});
    };
    try {
      await p.unlock(); await p.load({...song,duration:80}); await settled();
      await p.apply({...state,position:0},clockOffset());
      const item=p.stream.windows.get(3);
      let settledDecode=false;
      item.loading.finally(()=>{settledDecode=true;}).catch(()=>{});
      if (outcome==='cancelled') p.suspend();
      else t.mock.timers.tick(20000);
      await settled();
      if (outcome==='stays stuck') {
        t.mock.timers.tick(20000); await settled();
        assert.match(item.error?.message || '',/超时/);
      } else if (outcome==='recovers') {
        assert.ok(p.hasAudio(24));
        assert.deepEqual(waiting,[]);
        assert.notEqual(p.anchor,null);
      }
      assert.equal(attempts,outcome==='cancelled'?1:2);
      assert.equal(settledDecode,true);
      assert.equal(requests.filter(url=>url.endsWith('/lead_vocals/3')).length,1);
      for (const resolve of pending) resolve();
      await settled();
      if (outcome!=='recovers') assert.equal(item.audio,undefined);
    } finally { p.clear(); for (const resolve of pending) resolve(); globalThis.fetch=fetchAudio; }
  });
}
