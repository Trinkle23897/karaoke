import {Player, bindSeek} from './player.js';

// Preview shares the singing player's clock, buffering and independent tracks.
export function mountPreview(container, song, acquireAudio, onError) {
  const video = document.createElement('video');
  const controls = document.createElement('div');
  controls.className = 'preview-controls';
  const button = document.createElement('button');
  const listen = document.createElement('button');
  const seek = document.createElement('input');
  Object.assign(seek, {type:'range', min:0, max:song.duration, step:.1, value:0});
  seek.setAttribute('aria-label', '预览进度');
  const time = document.createElement('time');
  controls.append(button, seek, time, listen);
  container.replaceChildren(video, controls);
  const entry = {...song, id:song.job_id};
  let playing = false, silent = false, singing = false, position = 0, generation = 0;
  function renderButtons() {
    button.textContent = playing ? '暂停' : singing ? '静音预览' : '播放';
    listen.hidden = !singing && !silent;
    listen.textContent = singing ? '暂停演唱并试听' : '有声试听';
  }
  const update = value => {
    position = Math.max(0, value);
    if (!dragging()) seek.value = position;
    const seconds = Number(seek.value);
    seek.style.setProperty('--range-fill', `${100 * seconds / song.duration}%`);
    time.textContent = `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
  };
  const pause = () => {
    generation++;
    if (playing) update(silent ? video.currentTime : player.position() ?? position);
    playing = silent = false;
    player.suspend();
    renderButtons();
  };
  const player = new Player(video, update, () => { pause(); update(0); });
  const start = async (muted, automatic = false) => {
    const token = ++generation;
    playing = true;
    silent = muted;
    renderButtons();
    try {
      if (silent) {
        // Singing keeps the audio device. Fetch just the silent MV for a look.
        video.src = entry.artifacts.mv;
        video.currentTime = position;
        await video.play();
        return;
      }
      if (automatic) await player.resumeAudio();
      else await player.unlock();
      if (token !== generation) return;
      await player.load(entry, position);
      if (token !== generation) return;
      await player.apply({current_id:entry.id, position, playing:true, started_at:Date.now()/1000,
        levels:{instrumental:1, backing_vocals:1, lead_vocals:0}}, 0);
    } catch (error) {
      if (token === generation) { pause(); onError(error); }
    }
  };
  const audition = async transfer => {
    const token = generation;
    if (!await acquireAudio(transfer) || token !== generation) return;
    if (playing) pause();
    start(false);
  };
  player.onBuffering = value => { pause(); update(value); start(false, true); };
  player.onError = error => { pause(); onError(error); };
  video.ontimeupdate = () => { if (silent) update(video.currentTime); };
  video.onended = () => { if (silent && video.ended) { pause(); update(0); } };
  button.onclick = () => playing ? pause() : singing ? start(true) : audition(false);
  listen.onclick = () => audition(singing);
  const dragging = bindSeek(seek, () => entry.id, () => update(position), value => {
    const resume = playing;
    if (silent) { video.currentTime = value; update(value); return; }
    pause(); update(value);
    if (resume) audition(false);
  });
  video.poster = entry.poster || '';
  video.preload = 'none';
  update(0);
  return {
    sync(isSinging, ownsAudio) {
      singing = isSinging;
      if (playing && !silent && (singing || !ownsAudio)) pause();
      renderButtons();
    },
    close() { pause(); player.dispose(); },
  };
}
