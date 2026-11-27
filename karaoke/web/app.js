import { Player, bindSeek, bindSongButton } from './player.js';
import { mountPreview } from './preview.js';
import { isStaleRoomSnapshot, LocalPlayback, PlaybackTabs, RoomEvents, queueToTop, queueTopCandidates, preparationRetry } from './room-state.js';
import { createVideoSearch, createViewHistory, previewURL, searchLibrary, artistLibrarySongs, librarySongs, libraryArtists, songArtists, paginateSongs, filterSongs, playCount, orderResults, queueKey, queuePosition, queuePositions, historySuggestions, lyricLabel } from './search.js';

const $ = id => document.getElementById(id);
const icon = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const time = seconds => `${Math.floor(Math.max(0, seconds || 0)/60)}:${String(Math.floor(Math.max(0, seconds || 0)%60)).padStart(2,'0')}`;
let savedPlayback;
try { savedPlayback = JSON.parse(sessionStorage.getItem('karaokePlayback')); } catch { /* Start fresh if tab storage is unavailable or invalid. */ }
let tabStorage;
try { tabStorage = localStorage; } catch { /* PlaybackTabs also supports in-memory state. */ }
const tabs = new PlaybackTabs(tabStorage, `${Date.now()}-${Math.random()}`, savedPlayback);
let playback = new LocalPlayback(tabs.read().playback);
if (!tabs.record.owner) tabs.write(playback.save(), {claim:true});
let tabChange = tabs.record.change;
let finishRetry, finishSending, finishDelay = 1000, queueSyncError = false, connected = false;
let roomState = {entries:[], revision:0}, receivedRoom = false, viewRevision = 0, lastPlaybackSave = 0;
const stages = {download:'下载音频', download_video:'下载视频', verify_video:'校验视频', decode_audio:'提取音频', decode_native:'准备音频母版', separate_music:'分离伴奏', separate_lead:'分离主唱', package_two_stems:'整理音轨', package_three_stems:'整理音轨', resample_vocals:'整理人声', transcribe:'识别歌词', prepare_lyrics:'整理歌词', detect_language:'确认演唱语言', align_words:'对齐歌词', align_phonemes:'对齐起唱时间', align_repeats:'检查重复段落', align_context:'校准字幕', validate_timing:'核对起唱时间', render:'生成字幕画面', mux_video:'合成字幕 MV'};
let state = {entries:[], current_id:null, playing:false, position:0, started_at:null, levels:{instrumental:1,backing_vocals:1,lead_vocals:0}, revision:0};
let library = [], filter = 'all', view = 'library', clockOffset = 0, readySent = null, loading = false, loadError = '';
let indexedEntries, indexedLibrary, renderedLibrary, libraryById = new Map(), libraryBySource = new Map(), positionInQueue = queuePositions([]);
let toastTimer, librarySignature = '', queueSignature = '', historySignature = '', discoverySignature = '';
let queueTab = 'upcoming';
let libraryPage = 1;
let queuePage = 1, historyPage = 1, indexedHistory, recentHistory = [];
let pitchRequest = Promise.resolve();
let searchTimer, searchQuery = '', searchSongs = [], previewSong = null, skipping = false, composing = false;
let videoSource = 'songs', searchOrder = 'relevance', searchSnapshot = null, catalogueSelection = null;
const addingVideos = new Set();
const videoSearch = createVideoSearch(renderVideoResults, fetch, query => searchLibrary(library, query));
const lyricObserver = new IntersectionObserver(entries => {
  for (const entry of entries) videoSearch.checkLyrics(entry.target.dataset.lyricUrl, entry.isIntersecting);
});
const navigation = createViewHistory(history, () => ({query:$('song-input').value, title:$('title-input').value,
  filter, view, queueTab, libraryPage, queuePage, historyPage, historyQuery:$('history-query').value,
  videoSource, searchOrder, catalogueSelection, search:videoSearch.snapshot(), scroll:scrollY}), restoreView);
const current = () => playback.entry;
const playingHere = () => tabs.owns() && player.unlocked && playback.state.playing;
const playingEntryId = () => playingHere() && current()?.status === 'completed' ? current().id : '';
const playingEntries = () => [...new Set([...(roomState.playing_entry_ids || []), playingEntryId()].filter(Boolean))];
const queueOrder = id => queueToTop(state.entries, id, playingEntries());

function renderRange(input) {
  const span = Number(input.max) - Number(input.min);
  const fill = `${span > 0 ? 100 * (Number(input.value) - Number(input.min)) / span : 0}%`;
  if (input.style.getPropertyValue('--range-fill') !== fill) input.style.setProperty('--range-fill', fill);
}

function renderMixer() {
  player.setLevels(state.levels);
  const level = state.levels.lead_vocals;
  $('lead-toggle').textContent = `主唱：${level > 0 ? '开' : '关'}`;
  $('lead-toggle').setAttribute('aria-pressed', String(level > 0));
  $('lead-value').textContent = `${Math.round(level * 100)}%`;
  $('lead-down').disabled = level <= 0;
  $('lead-up').disabled = level <= 0 || level >= 1;
}

function renderStorage() {
  const storage = state.storage;
  if (!storage) return;
  const gib = bytes => (bytes/1024**3).toLocaleString('zh-CN', {maximumFractionDigits:2});
  const limited = storage.limit_bytes != null;
  $('storage-usage').textContent = limited ? `${gib(storage.used_bytes)} / ${gib(storage.limit_bytes)} GiB` : `${gib(storage.used_bytes)} GiB · 不限`;
  $('storage-meter').hidden = !limited;
  $('storage-meter').max = storage.limit_bytes || 1;
  $('storage-meter').value = storage.used_bytes;
  $('storage-note').hidden = storage.can_prepare;
  $('storage-limit').placeholder = limited ? '' : '不限';
  if (document.activeElement !== $('storage-limit')) $('storage-limit').value = limited ? storage.limit_bytes/1024**3 : '';
}

function toast(message, entry) {
  $('toast').textContent = message;
  if (entry) {
    const button = document.createElement('button');
    button.textContent = '查看待唱 →';
    button.onclick = () => showQueuedSong(entry.id);
    $('toast').append(button);
  }
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('toast').hidden = true, entry ? 6500 : 3800);
}

async function request(path, body, timeout = 0) {
  const controller = new AbortController();
  const timer = timeout ? setTimeout(() => controller.abort(), timeout) : null;
  try {
    const response = await fetch(path, {signal:controller.signal, ...(body ?
      {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {})});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请求没有成功，请检查输入后重试');
    return data;
  } finally { clearTimeout(timer); }
}

async function command(action, extra = {}, claim = !['ready','buffering'].includes(action)) {
  try {
    if (['play','pause','seek','restart','next','lead','toggle_lead','pitch','ready','buffering'].includes(action) ||
        (action === 'remove' && extra.entry_id === state.current_id)) {
      if (!claim && !tabs.owns()) return null;
      if (extra.revision !== undefined && extra.revision !== state.revision) return null;
      const claimed = claim && claimPlayback();
      const session = tabs.record.session;
      if (claimed && playback.state.playing && action !== 'pause')
        await player.unlock();
      if (!tabs.owns(session)) { followPlayback(); return null; }
      observePlayback();
      playback.command(action, extra, state.entries, clockOffset);
      renderState();
      savePlayback();
      syncFinished();
      return state;
    }
    const updated = await request('/v1/room/commands', {action, ...extra});
    acceptState(updated);
    return updated;
  } catch (error) { toast(error.message); return null; }
}

function renderConnection() {
  $('connection-dot').classList.toggle('online', connected && !queueSyncError);
  $('connection-text').textContent = queueSyncError ? '队列待同步' : connected ? '在线' : '正在重连';
}

async function syncFinished() {
  if (finishSending || finishRetry || !tabs.owns() || !playback.pendingFinishes.length) return;
  const sending = finishSending = playback, session = tabs.record.session;
  const owns = () => playback === sending && tabs.owns(session);
  renderNext();
  try {
    await sending.flushFinishes(body => request('/v1/room/commands', body, 10000), updated => {
      acceptState(updated);
      savePlayback();
      queueSyncError = false; finishDelay = 1000;
    }, owns);
  } catch {
    if (owns()) {
      queueSyncError = true;
      finishRetry = setTimeout(() => { finishRetry = null; syncFinished(); }, finishDelay);
      finishDelay = Math.min(30000, finishDelay * 2);
    }
  } finally {
    finishSending = null;
    renderNext();
    renderConnection();
    if (playback !== sending) syncFinished();
  }
}

function retryFinished() {
  clearTimeout(finishRetry); finishRetry = null;
  syncFinished();
}

function switchView(next, {scroll = true} = {}) {
  if (next !== 'player' && stage.classList.contains('is-fullscreen')) {
    if (document.fullscreenElement === stage) document.exitFullscreen();
    setFullscreen(false);
  }
  view = next;
  const mobile = matchMedia('(max-width:1100px)').matches;
  $('library-view').hidden = next !== 'library';
  $('player-view').hidden = next !== 'player';
  $('queue-panel').classList.toggle('mobile-open', next === 'queue');
  if (next === 'queue' && !mobile) { view = 'library'; $('library-view').hidden = false; }
  document.body.dataset.view = view;
  document.querySelectorAll('[data-view]').forEach(button => {
    button.classList.toggle('selected', button.dataset.view === view);
    button.setAttribute('aria-current', button.dataset.view === view ? 'page' : 'false');
  });
  if (scroll) window.scrollTo(0, 0);
  navigation.save();
}

function restoreView(saved) {
  clearTimeout(scrollSave);
  clearVideoSearch();
  $('song-input').value = saved.query || '';
  $('title-input').value = saved.title || '';
  filter = [...document.querySelectorAll('[data-filter]')].some(button => button.dataset.filter === saved.filter) ? saved.filter : 'all';
  libraryPage = saved.libraryPage || 1;
  queuePage = saved.queuePage || 1;
  historyPage = saved.historyPage || 1;
  videoSource = (saved.query && saved.videoSource) || 'songs';
  searchOrder = (saved.query && saved.searchOrder) || 'relevance';
  queueTab = saved.queueTab === 'history' ? 'history' : 'upcoming';
  $('history-query').value = saved.historyQuery || '';
  renderQueue();
  catalogueSelection = saved.catalogueSelection || null;
  searchQuery = saved.search ? saved.query.trim() : '';
  renderLibrary();
  videoSearch.restore(saved.search);
  switchView(saved.view || 'library', {scroll:false});
  requestAnimationFrame(() => window.scrollTo(0, saved.scroll || 0));
}

function renderLibrary() {
  const query = $('song-input').value.trim();
  const isURL = /^https?:\/\//i.test(query);
  $('title-row').hidden = !isURL;
  $('submit-song').innerHTML = `${isURL ? '点歌' : '搜索'} ${icon('arrow')}`;
  const artist = videoSource === 'artist' ? catalogueSelection?.title || searchSnapshot?.artist?.metadata?.title : '';
  const search = (artist || (isURL ? '' : query)).toLocaleLowerCase();
  const songs = artist ? artistLibrarySongs(library, artist, searchSnapshot?.artist?.results) : librarySongs(library, search ? 'all' : filter, search);
  const page = paginateSongs(songs, libraryPage);
  if (receivedRoom) libraryPage = page.page;
  const preparedView = filter === 'all' && !search;
  $('library-count').textContent = `${songs.length} 首`;
  $('library-heading').textContent = search ? '曲库里的歌' : '本地曲库';
  document.querySelectorAll('[data-filter]').forEach(button => {
    button.classList.toggle('selected', button.dataset.filter === filter);
    button.setAttribute('aria-pressed', String(button.dataset.filter === filter));
    const count = button.querySelector('span');
    if (count) count.textContent = librarySongs(library, button.dataset.filter).length;
  });
  $('local-library').hidden = Boolean((searchQuery && !['songs','artists','artist'].includes(videoSource)) || (search && !songs.length));
  document.querySelector('#local-library .filters').hidden = Boolean(search);
  document.querySelectorAll('.library-pagination').forEach(nav => {
    nav.hidden = page.pages < 2;
    const select = nav.querySelector('select');
    if (select.options.length !== page.pages)
      select.innerHTML = Array.from({length:page.pages}, (_, i) => `<option value="${i+1}">${i+1} / ${page.pages}</option>`).join('');
    select.value = String(page.page);
    nav.querySelector('[data-library-step="-1"]').disabled = page.page === 1;
    nav.querySelector('[data-library-step="1"]').disabled = page.page === page.pages;
  });
  const signature = JSON.stringify([page.songs, page.page, filter, query]);
  if (signature !== librarySignature) {
    librarySignature = signature;
    $('song-grid').innerHTML = page.songs.map((song, index) => songRow({...song, provider:'local', thumbnail:song.poster, uploader:song.artist}, page.offset+index, {inLibrary:true, local:true})).join('');
  }
  $('search-empty').hidden = songs.length > 0;
  $('search-empty').querySelector('h3').textContent = preparedView ? '还没有可以直接唱的歌' : '这里还没有歌曲';
  $('library-list').hidden = !songs.length;
  renderDiscovery();
  updateAddButtons();
}

function renderDiscovery() {
  const artists = libraryArtists(library);
  $('home-discovery').hidden = !artists.length || Boolean($('song-input').value.trim() || searchQuery);
  const signature = JSON.stringify(artists);
  if (signature === discoverySignature) return;
  discoverySignature = signature;
  $('home-artists').innerHTML = artists.map(artist => `<button class="home-artist" data-home-artist="${escape(artist.name)}" aria-label="查看${escape(artist.name)}的歌曲" title="${escape(artist.name)}">
    ${artist.poster ? `<img src="${escape(artist.poster)}" alt="" loading="lazy">` : `<span class="artist-initial" aria-hidden="true">${escape([...artist.name][0])}</span>`}
    <strong>${escape(artist.name)}</strong></button>`).join('');
}

function renderVideoResults(providers, mode = videoSource) {
  videoSource = mode;
  if (mode === 'artist' && providers.artist.selection) catalogueSelection = {...providers.artist.selection, returnMode:'artists'};
  else if (mode === 'artists') catalogueSelection = null;
  searchSnapshot = providers;
  $('library-view').classList.add('searching');
  $('video-results').hidden = false;
  renderLibrary();
  const info = providers[videoSource];
  if (!info) return;
  const artists = videoSource === 'artists', album = videoSource === 'album', catalogue = album || videoSource === 'artist';
  const results = info.results.map(song => {
    const cached = libraryBySource.get(queueKey(song));
    if (!cached) return song;
    const {status, lyrics_need_review, lyrics_unavailable} = cached;
    return {...song, status, lyrics_need_review, lyrics_unavailable};
  });
  const metadata = info.metadata || catalogueSelection || {};
  const catalogueHeading = metadata.catalogue_section === 'videos' ? '歌手视频' : metadata.catalogue_section === 'releases' ? '歌手作品' : '热门歌曲';
  $('clear-search').textContent = catalogue || (artists && providers.artist?.selection) ? '返回上一级' : '返回本地曲库';
  const popularityAvailable = results.some(song => playCount(song.views) !== null);
  if (searchOrder === 'popular' && !popularityAvailable) searchOrder = 'relevance';
  searchSongs = orderResults(results, album && searchOrder === 'relevance' ? 'album' : searchOrder);
  $('search-order').value = searchOrder;
  $('search-order').querySelector('[value="popular"]').disabled = !popularityAvailable;
  $('search-order').querySelector('[value="relevance"]').textContent = album ? '专辑顺序' : '默认排序';
  $('search-order-label').hidden = artists || !results.length;
  $('video-results').classList.toggle('is-catalogue', catalogue);
  $('search-heading').textContent = catalogue ? metadata.title || (album ? '专辑曲目' : catalogueHeading) : artists ? '找一位歌手' : '搜索结果';
  $('search-order-note').textContent = artists ? '选一位歌手，看看有哪些歌想唱' :
    searchOrder === 'popular' ? '已加载结果按播放量排序 · YouTube 提供' : searchOrder === 'lyrics' ? '有歌词的排在最前，其次是歌词待校对的歌曲' : searchOrder === 'ready' ? '已加载结果中，已准备歌曲排在前面' :
    searchOrder === 'title' ? '已加载结果按歌名排序' : album ? 'YouTube Music 专辑曲目顺序' :
    videoSource === 'local' ? '本地曲库 · 现场版靠后' : 'YouTube 原顺序 · 现场版靠后';
  $('search-order-note').hidden = !['popular','lyrics'].includes(searchOrder);
  $('search-order').title = $('search-order-note').textContent;
  $('search-status').innerHTML = `<span class="${info.error ? 'search-error' : ''}">${info.loading ? '正在查找…' : info.error ? escape(info.error) : results.length ? `${videoSource === 'local' ? '' : '已加载 '}${results.length} ${artists ? '位歌手' : videoSource === 'youtube' || metadata.catalogue_section === 'videos' ? '个视频' : videoSource === 'songs' || metadata.catalogue_section === 'releases' ? '项' : '首'}${catalogue && info.metadata?.complete === false ? ' · 歌手页精选' : ''}` : catalogue ? '这个页面暂未提供可点播作品' : '没有找到，试试另一个关键词或搜索类型'}${info.error ? '<button data-search-retry>重试</button>' : ''}</span>`;
  $('artist-summary').hidden = !catalogue;
  const otherArtists = !album && (providers.artists?.results.length > 1 || providers.artists?.nextCursor);
  if (catalogue) $('artist-summary').innerHTML = `${metadata.thumbnail ? `<img src="${escape(metadata.thumbnail)}" alt="" referrerpolicy="no-referrer">` : ''}<div><p>${escape(metadata.monthly_listeners || metadata.artist || (album ? '专辑曲目' : catalogueHeading))} · YouTube Music</p>${otherArtists ? '<button class="text-button" data-other-artists>其他歌手 →</button>' : ''}</div>`;
  renderSearchMore(info);
  const videos = videoSource === 'youtube', local = videoSource === 'local';
  $('search-list').className = artists ? '' : `search-list${videos ? ' video-list' : local ? ' local-list' : ''}`;
  $('search-list').classList.toggle('has-albums', !videos && !local && searchSongs.some(song => song.album));
  $('search-list').setAttribute('role', artists ? 'group' : 'table');
  $('search-list').setAttribute('aria-label', artists ? '歌手列表' : videos ? '视频列表' : '歌曲列表');
  $('music-columns').hidden = artists || !searchSongs.length;
  $('music-title-heading').textContent = videos ? '视频' : videoSource === 'songs' ? '名称' : '歌曲';
  $('music-artist-heading').textContent = videos ? '频道' : '歌手';
  $('music-alternative-heading').textContent = videos ? '来源' : '找 MV';
  $('video-grid').className = artists ? 'artist-grid' : 'music-list';
  if (artists) $('video-grid').removeAttribute('role');
  else $('video-grid').setAttribute('role', 'rowgroup');
  updateSearchRows({artists, videos, local});
  document.querySelectorAll('[data-video-source]').forEach(button => {
    const selected = button.dataset.videoSource === (album ? 'songs' : catalogue ? 'artists' : videoSource);
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  updateAddButtons();
  navigation.save();
}

function lyricNote(song) {
  if (!song?.lyrics_unavailable && !song?.lyrics_need_review) return '';
  const {title, text} = lyricLabel(song);
  return `<span class="lyric-status" title="${escape(title)}">${text}</span>`;
}

function songRow(song, index, {artists = false, videos = false, local = false, inLibrary = false} = {}) {
  const thumbnail = song.thumbnail ? `<img src="${escape(song.thumbnail)}" alt="" loading="lazy" referrerpolicy="no-referrer">` : `<span class="music-initial">${escape(song.title.slice(0,1))}</span>`;
  if (artists) return `<button class="artist-card" data-artist-result="${index}">${thumbnail}<strong>${escape(song.title)}</strong><span>热门歌曲 ${icon('arrow')}</span></button>`;
  const playable = !song.kind || song.kind === 'song';
  const catalogueAction = ['artist','album'].includes(song.kind) ? `data-catalogue="${index}"` : '';
  const open = playable ? (inLibrary ? `data-preview-song="${escape(song.id)}"` : `data-preview="${index}"`) : catalogueAction || `href="${escape(song.url)}" target="_blank" rel="noopener noreferrer"`;
  const tag = playable || catalogueAction ? 'button' : 'a';
  const action = playable ? `<button class="music-add" ${inLibrary ? `data-song="${escape(song.id)}"` : `data-add-video="${index}"`} aria-label="点歌：${escape(song.title)}">${icon('plus')}<span>点歌</span></button>` :
    `<${tag} class="music-add" ${open}>${song.kind === 'album' ? '查看专辑' : catalogueAction ? '查看歌曲' : '打开'} ${icon('arrow')}</${tag}>`;
  const performers = local ? songArtists(song) : song.artists || [];
  const credits = performers.length ? performers.map(a => local ? `<button data-search-artist="${escape(a.name)}">${escape(a.name)}</button>` : a.id ? `<button data-artist-id="${escape(a.id)}" data-artist-name="${escape(a.name)}">${escape(a.name)}</button>` : escape(a.name)).join(' · ') : escape(song.uploader || '—');
  const artistText = escape(performers.map(a => a.name).join(' · ') || song.uploader || '—');
  const plays = playCount(song.views) === null ? null : formatPlays(song.views);
  const kind = song.kind === 'album' ? '专辑' : song.kind === 'artist' ? '歌手' : song.version || '';
  const lyric = local ? lyricNote(song) : playable ? `<span class="lyric-status" data-lyric-url="${escape(song.url)}"></span>` : '';
  const meta = kind || lyric ? `<div class="music-meta">${kind ? `<span${song.kind === 'album' ? ' class="music-kind"' : ''}>${escape(kind)}</span>` : ''}${lyric}</div>` : '';
  return `<article class="music-row" ${inLibrary ? `data-card="${escape(song.id)}"` : `data-result="${index}"`} role="row"><span class="music-rank" role="cell">${String(index+1).padStart(2,'0')}</span>
    <div class="music-song" role="cell"><${tag} class="music-cover" ${open} aria-label="${playable ? '预览' : '打开'}：${escape(song.title)}">${thumbnail}<span>${icon(playable ? 'play' : 'arrow')}</span></${tag}>
      <div class="music-title"><h3><${tag} ${open} title="${escape(song.title)}"><span class="music-name">${escape(song.title)}</span><span class="music-compact-artist"><span>${artistText}</span>${kind ? `<small>${escape(kind)}</small>` : ''}${plays !== null && !local ? `<span class="music-compact-plays" aria-label="播放量 ${plays}">${plays}次</span>` : ''}${song.duration ? `<span class="music-compact-duration" aria-label="时长 ${time(song.duration)}">${time(song.duration)}</span>` : ''}</span></${tag}></h3>${meta}</div></div>
    <div class="music-credits" role="cell" title="${artistText}">${credits}</div>
    <div class="music-album" role="cell" title="${escape(song.album || '')}">${escape(song.album || '—')}</div>
    <div class="music-facts"><div class="music-plays${plays === null ? ' is-empty' : ''}" role="cell">${plays !== null ? `${plays}<small>播放</small>` : '—'}</div>
    <div class="music-duration${!song.duration ? ' is-empty' : ''}" role="cell">${song.duration ? time(song.duration) : '—'}</div></div>
    <div class="music-actions" role="cell">${action}</div>
    <div class="music-alternative" role="cell">${(local || song.provider === 'youtube_music') && playable ? `<button class="find-mv" ${inLibrary ? `data-find-mv-song="${escape(song.id)}"` : `data-find-mv="${index}"`} aria-label="找 MV：${escape(song.title)}"><span class="find-mv-prefix">找 </span>MV</button>` : videos ? `<a class="find-mv" href="${escape(song.url)}" target="_blank" rel="noopener noreferrer" aria-label="在 YouTube 打开"><span class="find-mv-prefix">YouTube </span>↗</a>` : ''}</div></article>`;
}

function updateSearchRows(options) {
  const grid = $('video-grid');
  const focus = grid.contains(document.activeElement) ? document.activeElement : null;
  const top = focus?.getBoundingClientRect().top;
  lyricObserver.disconnect();
  const existing = new Map([...grid.children].map(node => [node.dataset.searchKey, node]));
  const occurrences = new Map();
  searchSongs.forEach((song, index) => {
    const source = song.browse_id || song.url;
    const occurrence = occurrences.get(source) || 0;
    occurrences.set(source, occurrence+1);
    const key = `${source}:${occurrence}`;
    const content = JSON.stringify([song.provider, song.kind, song.title, song.thumbnail, song.uploader, song.artists,
      song.version, song.album, song.views, song.duration, options.local && song.lyrics_need_review]);
    let node = existing.get(key);
    existing.delete(key);
    if (node && node.searchContent !== content) { node.remove(); node = null; }
    if (!node) {
      const template = document.createElement('template');
      template.innerHTML = songRow(song, index, options);
      node = template.content.firstElementChild;
      node.dataset.searchKey = key;
      node.searchContent = content;
    }
    // Keep existing cards, images and focused buttons alive when appending a page.
    for (const attribute of ['data-result', 'data-artist-result', 'data-catalogue', 'data-preview', 'data-add-video', 'data-find-mv']) {
      if (node.hasAttribute(attribute)) node.setAttribute(attribute, index);
      node.querySelectorAll(`[${attribute}]`).forEach(button => button.setAttribute(attribute, index));
    }
    const rank = node.querySelector('.music-rank');
    if (rank) rank.textContent = String(index+1).padStart(2,'0');
    const label = node.querySelector('[data-lyric-url]');
    if (label) {
      const lyric = lyricLabel(song);
      label.textContent = lyric.text;
      label.title = lyric.title;
      label.classList.toggle('has-lyrics', lyric.available && !song.lyrics_need_review);
      if (song.status !== 'completed' && !song.lyrics_unavailable) lyricObserver.observe(label);
    }
    if (grid.children[index] !== node) grid.insertBefore(node, grid.children[index] || null);
  });
  existing.forEach(node => node.remove());
  // insertBefore detaches the focused control when a lyric check moves its row.
  if (focus?.isConnected) {
    if (document.activeElement !== focus) focus.focus({preventScroll:true});
    window.scrollBy(0, focus.getBoundingClientRect().top - top);
  }
}

function renderSearchMore(info) {
  const more = $('search-more'), button = $('search-more-button');
  more.hidden = videoSource === 'local' || info.loading || !!info.error || (!info.results.length && !info.nextCursor);
  $('search-more-status').textContent = info.loadingMore ? '正在加载更多…' : info.moreError ||
    (info.nextCursor ? '下拉继续看' : '已经到底了');
  $('search-more-status').classList.toggle('search-error', !!info.moreError);
  button.hidden = !info.nextCursor;
  button.disabled = !!info.loadingMore;
  button.textContent = info.expired ? '重新搜索' : info.moreError ? '重试加载' : '加载更多';
  // Reobserve after a page arrives, including when it was too short to fill the screen.
  searchObserver?.unobserve(more);
  if (!more.hidden && info.nextCursor && !info.loadingMore && !info.moreError && !info.autoBlocked) searchObserver?.observe(more);
}

function formatPlays(value) {
  return new Intl.NumberFormat('zh-CN', {notation:'compact', maximumFractionDigits:1}).format(playCount(value));
}

function openCatalogue(item) {
  navigation.push();
  if ((item.kind || 'artist') === 'artist' && searchSnapshot?.artists?.results.some(artist => artist.browse_id === item.browse_id)) {
    searchOrder = 'relevance';
    videoSearch.selectArtist(item.browse_id);
    window.scrollTo(0, 0);
    return;
  }
  const query = $('song-input').value.trim();
  clearVideoSearch();
  const returnMode = catalogueSelection?.returnMode || videoSource;
  catalogueSelection = {...item, returnMode};
  videoSource = item.kind || 'artist'; searchOrder = 'relevance';
  searchQuery = query;
  videoSearch.search(item.browse_id, videoSource);
  window.scrollTo(0, 0);
}

function changeSearchMode(mode, query) {
  navigation.push();
  if (query !== undefined) $('song-input').value = query;
  videoSource = mode; searchOrder = 'relevance'; catalogueSelection = null;
  switchView('library');
  searchVideos(true);
  window.scrollTo(0, 0);
}

function updateAddButtons() {
  document.querySelectorAll('[data-add-video], [data-song], #preview-add, #submit-song').forEach(button => {
    const form = button.id === 'submit-song';
    const song = form ? {url:$('song-input').value.trim(), title:$('title-input').value.trim()} :
      button.id === 'preview-add' ? previewSong : button.hasAttribute('data-song')
        ? libraryById.get(button.dataset.song) : searchSongs[button.dataset.addVideo];
    if (!song) return;
    if (form && !/^https?:\/\//i.test(song.url)) { button.disabled = false; button.classList.remove('is-queued'); button.removeAttribute('aria-label'); return; }
    const position = positionInQueue(song), adding = addingVideos.has(queueKey(song));
    const label = position ? '已点' : adding ? '点歌中…' : song.available === false ? '暂不可用' : '点歌';
    button.disabled = Boolean(position || adding || song.available === false);
    button.classList.toggle('is-queued', position > 0);
    button.setAttribute('aria-label', `${label}${position ? `，第${position}首` : ''}：${song.title || song.url}`);
    const markup = `${position ? '' : icon('plus')}<span>${label}${position ? `<small>第${position}首</small>` : ''}</span>`;
    if (button.innerHTML !== markup) button.innerHTML = markup;
  });
}

function clearVideoSearch() {
  clearTimeout(searchTimer);
  videoSearch.cancel();
  searchQuery = '';
  searchSongs = [];
  searchSnapshot = null;
  $('library-view').classList.remove('searching');
  $('video-results').hidden = true;
  searchObserver?.unobserve($('search-more'));
  $('search-more').hidden = true;
  $('video-grid').replaceChildren();
  $('search-status').replaceChildren();
}

function searchVideos(force = false) {
  clearTimeout(searchTimer);
  const query = $('song-input').value.trim();
  if (!query || /^https?:\/\//i.test(query) || (!force && query === searchQuery)) return;
  if (force && query === searchQuery && videoSource === 'artist' && searchSnapshot?.artist?.selection) {
    videoSearch.selectArtist(searchSnapshot.artist.selection.browse_id, true);
    return;
  }
  clearVideoSearch();
  searchQuery = query;
  if (query.length > 160) { toast('搜索关键词请控制在 160 字以内'); return; }
  videoSearch.search(['artist','album'].includes(videoSource) ? catalogueSelection.browse_id : query, videoSource);
}

function editSearch() {
  libraryPage = 1;
  if (['artist','album'].includes(videoSource)) { videoSource = catalogueSelection?.returnMode || 'songs'; catalogueSelection = null; }
  if (!composing && !$('song-input').value.trim()) {
    videoSource = 'songs'; searchOrder = 'relevance';
  }
  clearVideoSearch();
  renderLibrary();
  navigation.save();
  if (!composing) searchTimer = setTimeout(searchVideos, 700);
}

async function addVideo(song, {title, repeat = false} = {}) {
  if (!song || addingVideos.has(queueKey(song)) || song.available === false) return;
  const queued = !repeat && state.entries[positionInQueue(song)-1];
  if (queued) return;
  const key = queueKey(song);
  addingVideos.add(key); updateAddButtons();
  $('preview-error').hidden = true;
  try {
    const cached = libraryBySource.get(key);
    const previous = new Set(state.entries.map(entry => entry.id));
    const updated = await request('/v1/room/entries', {if_absent:!repeat, source_thumbnail:song.thumbnail, ...(cached?.job_id ? {job_id:cached.job_id} :
      {url:song.url, ...(title !== undefined ? {title} : {source_title:song.title})})});
    acceptState(updated);
    if ($('video-preview').open) $('video-preview').close();
    const added = updated.entries.find(entry => !previous.has(entry.id) && queueKey(entry) === key);
    const position = added ? updated.entries.findIndex(entry => entry.id === added.id)+1 : queuePosition(song, updated.entries);
    toast(`已点${position ? ` · 第${position}首` : ''}`, updated.entries[position-1]);
  } catch (error) {
    toast(error.message);
    $('preview-error').textContent = error.message;
    $('preview-error').hidden = false;
  } finally { addingVideos.delete(key); updateAddButtons(); }
}

function showQueuedSong(id) {
  const index = state.entries.findIndex(song => song.id === id);
  if (index < 0) return;
  queuePage = Math.floor(index / 50) + 1;
  $('toast').hidden = true;
  if ($('video-preview').open) $('video-preview').close();
  queueTab = 'upcoming'; renderQueue();
  if (matchMedia('(max-width:1100px)').matches) switchView('queue');
  else navigation.save();
  document.querySelector(`[data-entry="${CSS.escape(id)}"]`)?.scrollIntoView({block:'nearest'});
}

let cachedPreview = null;
async function acquirePreviewAudio(transfer) {
  followPlayback();
  if (playback.state.playing) {
    if (!transfer || !await command('pause')) return false;
  } else claimPlayback();
  return tabs.owns() && !playback.state.playing;
}

function previewVideo(song) {
  if (current() && queuePosition(song, [current()])) { switchView('player'); return; }
  clearPreview();
  previewSong = song;
  $('preview-title').textContent = song.title;
  $('preview-source').href = song.url;
  $('preview-source').textContent = '在原站观看 ↗';
  $('preview-error').hidden = true;
  const cached = libraryBySource.get(queueKey(song));
  if (cached?.status === 'completed' && cached.job_id) {
    cachedPreview = mountPreview($('preview-player'), cached, acquirePreviewAudio, error => {
      $('preview-error').textContent = error.message;
      $('preview-error').hidden = false;
    });
  }
  syncPreview();
  updateAddButtons();
  $('video-preview').showModal();
}

function syncPreview() {
  if (!previewSong) return;
  const singing = playback.state.playing, owns = tabs.owns();
  if (cachedPreview) return cachedPreview.sync(singing, owns);
  const container = $('preview-player');
  const source = previewURL(previewSong, location.origin);
  if (!source) {
    $('preview-error').textContent = '这个来源没有嵌入预览，可在原站观看';
    $('preview-error').hidden = false;
    return;
  }
  // An embedded player controls its own audio. Mount it only in the tab
  // that can sound, and remove it as soon as another tab takes over.
  if (singing || !owns) {
    let button = container.querySelector('button');
    if (!button) {
      button = document.createElement('button');
      button.className = 'primary';
      container.replaceChildren(button);
    }
    button.textContent = singing ? '暂停演唱并试听' : '打开预览';
    button.onclick = async () => { if (await acquirePreviewAudio(singing)) syncPreview(); };
    return;
  }
  if (container.querySelector('iframe')) return;
  const iframe = document.createElement('iframe');
  iframe.src = source;
  iframe.title = previewSong.title;
  iframe.allow = 'autoplay; encrypted-media; picture-in-picture; fullscreen';
  iframe.allowFullscreen = true;
  iframe.referrerPolicy = 'strict-origin-when-cross-origin';
  container.replaceChildren(iframe);
}

function clearPreview() {
  cachedPreview?.close();
  cachedPreview = previewSong = null;
  $('preview-player').replaceChildren();
}

function queuedStatus(entry) {
  return entry.next_retry_at ? '等待自动重试' : '等待准备';
}

function statusText(entry) {
  if (entry.status === 'failed') return entry.error?.message || '准备失败，请重试';
  if (entry.status === 'completed') return '';
  if (entry.status === 'queued') return queuedStatus(entry);
  return (entry.active_stages || []).map(s => stages[s] || '正在准备').join(' · ') || '开始准备';
}

function preparationElapsedHTML(entry) {
  if (entry?.status !== 'running') return '';
  const started = Date.parse(entry.preparation_started_at)/1000;
  return Number.isFinite(started) ? `<span class="preparation-elapsed" data-preparation-start="${started}">${preparationTime(started)}</span>` : '';
}

function preparationStages(entry) {
  if (entry?.status !== 'running') return [];
  return (entry.active_stages?.length ? entry.active_stages : ['starting']).map(stage => {
    const p = entry.stage_progress?.[stage] || {};
    const measured = Number.isFinite(p.completed) && Number.isFinite(p.total) && p.total > 0;
    const percent = measured ? Math.min(100, Math.floor(100*p.completed/p.total)) : null;
    const label = `${stages[stage] || '开始准备'}${p.detail ? ` · ${p.detail}` : ''}`;
    const amount = percent !== null ? `${percent}%` : p.unit === 'bytes' && Number.isFinite(p.completed) ? `${(p.completed/1024**2).toFixed(1)} MB` : '';
    return {stage, label, amount, percent};
  });
}

function preparationHTML(entry) {
  const pending = preparationStages(entry);
  if (!pending.length) return '';
  return `<div class="preparation">${pending.map(({stage, label, amount, percent}) =>
    `<div class="preparation-stage" data-stage="${escape(stage)}"><div class="preparation-label"><span title="${escape(label)}">${escape(label)}</span><span class="preparation-metrics">${amount}</span></div><div class="preparation-meter" role="progressbar" aria-valuemin="0" aria-valuemax="100" ${percent === null ? '' : `aria-valuenow="${percent}"`} aria-label="${escape(label)}"><span style="width:${percent === null ? 30 : percent}%"></span></div></div>`
  ).join('')}</div>`;
}

function updatePreparation(container, entry) {
  const pending = preparationStages(entry);
  const rows = [...container.querySelectorAll('[data-stage]')];
  if (pending.length !== rows.length || pending.some((item, i) => item.stage !== rows[i].dataset.stage)) {
    container.innerHTML = preparationHTML(entry);
    return;
  }
  pending.forEach(({label, amount, percent}, i) => {
    const row = rows[i], name = row.querySelector('.preparation-label>span');
    const metrics = row.querySelector('.preparation-metrics'), meter = row.querySelector('[role=progressbar]');
    if (name.textContent !== label) { name.textContent = label; name.title = label; meter.setAttribute('aria-label', label); }
    if (metrics.textContent !== amount) metrics.textContent = amount;
    if (meter.getAttribute('aria-valuenow') !== (percent === null ? null : String(percent))) {
      if (percent === null) meter.removeAttribute('aria-valuenow');
      else meter.setAttribute('aria-valuenow', percent);
    }
    const width = `${percent === null ? 30 : percent}%`;
    if (meter.firstElementChild.style.width !== width) meter.firstElementChild.style.width = width;
  });
}

function preparationTime(started) {
  return `已用时 ${time(Date.now()/1000 + clockOffset - started)}`;
}

setInterval(() => {
  if (document.hidden) return;
  for (const label of document.querySelectorAll('[data-preparation-start]')) {
    const text = preparationTime(Number(label.dataset.preparationStart));
    if (label.textContent !== text) label.textContent = text;
  }
}, 1000);

function updatePreparationSlots(container, songs) {
  const byId = new Map(songs.map(song => [song.id, song]));
  for (const element of container.children) {
    const slot = element.querySelector('.preparation-slot');
    if (slot) updatePreparation(slot, byId.get(element.dataset.entry));
  }
}

function renderQueuePage(kind, page) {
  const nav = $(`${kind}-pagination`);
  $(`${kind}-list`).style.setProperty('--queue-number-width', `${Math.max(2, String(page.offset + page.songs.length).length)}ch`);
  nav.hidden = page.pages < 2;
  const select = nav.querySelector('select');
  if (select.options.length !== page.pages)
    select.innerHTML = Array.from({length:page.pages}, (_, i) => `<option value="${i+1}">${i+1} / ${page.pages}</option>`).join('');
  select.value = String(page.page);
  nav.querySelector('[data-page-step="-1"]').disabled = page.page === 1;
  nav.querySelector('[data-page-step="1"]').disabled = page.page === page.pages;
}

function renderQueue() {
  document.querySelectorAll('[data-queue-tab]').forEach(button => {
    const selected = button.dataset.queueTab === queueTab;
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  $('upcoming-panel').hidden = queueTab !== 'upcoming';
  $('history-panel').hidden = queueTab !== 'history';
  renderHistory();
  $('queue-count').textContent = String(state.entries.length).padStart(2,'0');
  $('nav-count').textContent = state.entries.length;
  $('queue-empty').hidden = state.entries.length > 0;
  const preparing = state.entries.filter(e => ['running','queued'].includes(e.status)).length;
  $('queue-badge').textContent = `${preparing} 首准备中`;
  $('queue-badge').hidden = !preparing;
  if (queueTab !== 'upcoming') return;
  const page = paginateSongs(state.entries, queuePage, 50);
  if (receivedRoom) queuePage = page.page;
  renderQueuePage('queue', page);
  const playing = new Set(playingEntries()), movable = queueTopCandidates(state.entries, [...playing]);
  const signature = JSON.stringify([page.songs.map(({stage_progress, active_stages, ...entry}) => entry), page.page,
    state.current_id, state.playing, [...playing], page.songs.map(entry => movable.has(entry.id)), playback.pendingFinishes]);
  if (signature === queueSignature) {
    // Progress pushes must not replace focused buttons or interrupt a click.
    updatePreparationSlots($('queue-list'), page.songs);
    return;
  }
  queueSignature = signature;
  $('queue-list').innerHTML = page.songs.map((entry, i) => {
    const singing = playing.has(entry.id);
    const active = entry.id === state.current_id || singing;
    const finishing = playback.isFinished(entry.id);
    const retry = preparationRetry(entry);
    const status = entry.status === 'running' || !statusText(entry) ? '' : `<span class="entry-status ${entry.status === 'queued' ? 'preparing' : ''}">${escape(statusText(entry))}</span>`;
    return `<article class="queue-entry ${active ? 'current' : ''}" data-entry="${escape(entry.id)}"><span class="entry-number">${singing ? '♫' : String(page.offset+i+1).padStart(2,'0')}</span>
      <button class="entry-cover" data-action="preview" aria-label="${active ? '查看播放' : '预览'}：${escape(entry.title)}">${entry.poster ? `<img class="entry-art" src="${escape(entry.poster)}" alt="" loading="lazy">` : `<span class="entry-art">♪</span>`}</button>
      <div class="entry-info"><h3 class="entry-title"><button data-action="preview">${escape(entry.title)}</button></h3><div class="song-info-row"><p class="entry-artist">${escape(entry.artist || 'YouTube 视频')}</p>${finishing ? '<span class="entry-status">等待同步移除</span>' : entry.status === 'failed' ? '' : preparationElapsedHTML(entry) || status || lyricNote(entry)}</div>${entry.status === 'running' ? `<div class="preparation-slot">${preparationHTML(entry)}</div>` : entry.status === 'failed' ? `<p class="entry-error">${escape(statusText(entry))}</p>` : ''}</div>
      <div class="entry-tools">${retry && !finishing ? `<button data-action="${retry.action}" aria-label="${retry.label}：${escape(entry.title)}">${icon('restart')}${retry.label}</button>` : ''}<button data-action="top" aria-label="置顶：${escape(entry.title)}" ${!finishing && movable.has(entry.id) ? '' : 'disabled'}>${icon('up')}置顶</button><button data-action="remove" aria-label="移除：${escape(entry.title)}" ${finishing ? 'disabled' : ''}>${icon('trash')}移除</button></div></article>`;
  }).join('');
}

function renderHistory() {
  if (indexedHistory !== state.history) {
    indexedHistory = state.history;
    recentHistory = historySuggestions(state.history || []).recent;
  }
  const history = recentHistory;
  $('history-count').textContent = String(history.length).padStart(2,'0');
  if (queueTab !== 'history') return;
  const songs = filterSongs(history, $('history-query').value);
  const page = paginateSongs(songs, historyPage, 50);
  if (receivedRoom) historyPage = page.page;
  renderQueuePage('history', page);
  $('history-empty').hidden = songs.length > 0;
  $('history-empty').querySelector('h3').textContent = history.length ? '没有找到匹配的已唱歌曲' : '还没有唱过的歌';
  const posters = new Map(library.map(song => [song.url, song.poster]));
  const signature = JSON.stringify([page.page, page.songs.map(song => [song, posters.get(song.url)])]);
  if (signature === historySignature) return;
  historySignature = signature;
  const date = new Intl.DateTimeFormat('zh-CN', {month:'numeric', day:'numeric', hour:'2-digit', minute:'2-digit'});
  $('history-list').innerHTML = page.songs.map((song, index) => {
    const poster = posters.get(song.url);
    return `<article class="queue-entry history-entry" data-history-entry="${escape(song.id)}"><span class="entry-number">${String(page.offset+index+1).padStart(2,'0')}</span>
      ${poster ? `<img class="entry-art" src="${escape(poster)}" alt="" loading="lazy">` : `<span class="entry-art">♪</span>`}
      <div class="entry-info"><h3 class="entry-title">${escape(song.title)}</h3><p class="entry-artist">${escape(song.artist)}</p></div>
      <div class="entry-tools"><time datetime="${new Date(song.finished_at*1000).toISOString()}">${escape(date.format(song.finished_at*1000))}</time><button data-repeat-song ${song.url ? '' : 'disabled'} aria-label="再点：${escape(song.title)}">${icon('plus')}再点</button></div></article>`;
  }).join('');
}

function updateTransport() {
  const song = current();
  const idleTitle = state.playing ? '等待下一首' : '未播放';
  $('now-title').textContent = song?.title || idleTitle;
  $('now-artist').textContent = song?.artist || '';
  $('now-art').style.backgroundImage = song?.poster ? `url("${song.poster}")` : '';
  $('now-art').querySelector('span').hidden = Boolean(song?.poster);
  $('player-title').textContent = song?.title || idleTitle;
  $('player-artist').textContent = song?.artist || '';
  const poster = song?.poster || '';
  if ($('video').getAttribute('poster') !== poster) $('video').setAttribute('poster', poster);
  $('player-elapsed').innerHTML = preparationElapsedHTML(song) || lyricNote(song);
  const playing = playingHere();
  $('play-icon').setAttribute('href', playing ? '#i-pause' : '#i-play');
  $('play-toggle').setAttribute('aria-label', playing ? '暂停' : playback.allSkipped ? '唱已跳过的歌' : '播放');
  $('play-toggle').disabled = !song && !state.playing && !playback.allSkipped;
  renderNext();
  $('transport-restart').disabled = $('seek').disabled = song?.status !== 'completed';
  $('seek').max = song?.duration || 1;
  $('duration').textContent = time(song?.duration);
  $('playing-dot').hidden = !state.playing || song?.status !== 'completed';
  const pitch = state.pitch_semitones || 0;
  $('pitch-value').textContent = pitch ? `${pitch > 0 ? '+' : '−'}${Math.abs(pitch)}` : '原调';
  $('pitch-value').setAttribute('aria-label', `当前${pitch ? `${pitch > 0 ? '升' : '降'} ${Math.abs(pitch)} 个半音` : '原调'}`);
  $('pitch-down').disabled = pitch <= -6;
  $('pitch-up').disabled = pitch >= 6;
  renderMixer();
  updateCover();
}

function introSkipTarget() {
  return state.playing && tabs.owns() && player.unlocked && player.ready &&
    player.matches(current()) && !loading && !loadError
    ? player.introSkipTarget(playback.position()) : null;
}

function updateCover() {
  const song = current();
  const startingVideo = Boolean(player.pendingStart || player.videoPreparation);
  $('skip-intro').toggleAttribute('hidden', startingVideo || introSkipTarget() === null);
  if (player.anchor === null || player.entry?.id !== song?.id)
    $('opening-countdown').toggleAttribute('hidden', true);
  const frozen = !state.playing && player.frame && player.matches(song);
  $('screen-cover').toggleAttribute('hidden', Boolean(song?.status === 'completed' && !loading && !startingVideo && !loadError && player.unlocked && (player.ready || frozen)));
  const skipped = playback.allSkipped;
  const status = loadError || (skipped ? '待唱歌曲都已跳过' : !song ? state.playing ? '准备好后自动接唱' : '请先点歌' : song.status !== 'completed' ? statusText(song) : startingVideo ? '加载画面…' : loading ? '加载音轨…' : '');
  if ($('screen-status').textContent !== status) $('screen-status').textContent = status;
  $('screen-status').toggleAttribute('hidden', !status || song?.status === 'running');
  $('screen-start').toggleAttribute('hidden', !skipped && (song?.status !== 'completed' || loading || startingVideo));
  const label = skipped ? '唱已跳过的歌 ' : loadError ? '重试加载 ' : state.position > 0 ? '继续唱 ' : '开始唱 ';
  if ($('screen-start').firstChild.textContent !== label) $('screen-start').firstChild.textContent = label;
  const retry = preparationRetry(song);
  $('screen-retry').toggleAttribute('hidden', !retry || retry.action !== 'retry' || !state.entries.some(e => e.id === song.id));
}

let countdownFrame = '';
function updateTime(position) {
  const seek = $('seek');
  const value = Math.min(Number(seek.max), Math.round(Math.max(0, position) * 10) / 10);
  // The slider already displays tenths and the label displays seconds. Keep
  // their DOM unchanged between visible updates; audio keeps its precise clock.
  const seeking = draggingSeek();
  if (!seeking && Number(seek.value) !== value) seek.value = value;
  renderRange(seek);
  const label = time(seeking ? Number(seek.value) : position);
  if ($('current-time').textContent !== label) $('current-time').textContent = label;
  const opening = current()?.opening_countdown;
  const overlay = $('opening-countdown');
  const visible = position < 0 && opening && state.playing && player.anchor !== null;
  overlay.toggleAttribute('hidden', !visible);
  if (visible) {
    const remaining = Math.max(0, Math.min(opening.dots,
      Math.ceil((opening.onset-opening.settle_seconds-position)/opening.step_seconds-1e-9)));
    const signature = JSON.stringify([opening, remaining]);
    if (signature !== countdownFrame) {
      countdownFrame = signature;
      overlay.setAttribute('viewBox', `0 0 ${opening.width} ${opening.height}`);
      // Match the frozen first frame's rendered dots; the video takes over at t=0.
      overlay.replaceChildren(...Array.from({length:opening.dots}, (_, i) => {
        const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        for (const [name, value] of Object.entries({cx:(opening.x ?? opening.width/2)+(i-(opening.dots-1)/2)*opening.spacing,
          cy:opening.y, r:opening.radius, fill:i >= opening.dots-remaining ? opening.color : 'transparent',
          stroke:i >= opening.dots-remaining ? opening.color : '#f6f4eca0', 'stroke-width':2})) dot.setAttribute(name, value);
        return dot;
      }));
    }
  } else countdownFrame = '';
  updateCover();
}

function observePlayback(position = player.position()) {
  playback.observe(player.entry?.id, player.startedAt, position);
}
const player = new Player($('video'), position => {
  const played = playback.playedAt;
  observePlayback(position);
  if (played !== playback.playedAt || Date.now() - lastPlaybackSave > 1000) savePlayback();
  updateTime(position);
}, id => command('next', {entry_id:id}, false));
let bufferingRequest;
player.onBuffering = (position, startedAt, entryId = player.entry?.id) => {
  const token = `${entryId}:${startedAt}`;
  if (bufferingRequest === token) return;
  bufferingRequest = token;
  loading = true;
  updateCover();
  command('buffering', {entry_id:entryId, position, playback_started_at:startedAt})
    .finally(() => { if (bufferingRequest === token) bufferingRequest = null; });
};
player.onError = error => {
  if (!tabs.owns()) return;
  loadError = error.message || '视频播放失败，请重试';
  loading = false;
  updateCover(); toast(loadError);
};

async function syncPlayback() {
  const song = current();
  if (!tabs.owns() || !song || song.status !== 'completed' || !state.playing || !player.unlocked) {
    if (tabs.owns() && song && player.entry?.id === song.id)
      player.suspend(!state.playing && song.status === 'completed' && player.matches(song) ? $('paused-frame') : null, state.position);
    else { player.clear(); loadError = ''; }
    readySent = null; loading = false;
    updateCover(); return;
  }
  const revision = state.revision;
  try {
    if (!state.started_at) player.stop();
    observePlayback();
    const position = playback.position();
    if (state.started_at && (!player.matches(song) || !player.hasAudio(position))) {
      player.stop();
      player.onBuffering(position, state.started_at, song.id);
      return;
    }
    loading = !player.hasAudio(position) || !player.matches(song) ||
      (state.playing && player.context?.state !== 'running');
    updateCover();
    await player.load(song, position);
    if (!tabs.owns() || state.current_id !== song.id || state.revision !== revision || !player.matches(song)) return;
    if (state.playing && !state.started_at && player.unlocked) {
      await player.resumeAudio();
      if (!tabs.owns() || state.current_id !== song.id || state.revision !== revision) return;
      const preparing = player.prepareVideo(position);
      updateCover();
      await preparing;
      if (!tabs.owns() || state.current_id !== song.id || state.revision !== revision) return;
    }
    loading = false; loadError = '';
    if (state.playing && !state.started_at && player.unlocked && readySent !== song.id) {
      readySent = song.id;
      if (!await command('ready', {entry_id:song.id, revision})) readySent = null;
    }
    if (!state.playing || state.started_at) readySent = null;
    player.apply(state, 0);
    updateTransport();
  } catch (error) {
    if (!tabs.owns() || state.current_id !== song.id || state.revision !== revision) return;
    player.unlocked = false;
    loading = false; loadError = error.message;
    updateCover(); toast(loadError);
  }
}

function acceptState(updated) {
  if (isStaleRoomSnapshot(updated, roomState)) return;
  roomState = updated;
  receivedRoom = true;
  clockOffset = updated.server_time - Date.now()/1000;
  renderState();
  syncFinished();
}

function renderState() {
  if (!receivedRoom) return;
  const wasReady = current()?.status === 'completed';
  const previous = current()?.id;
  state = {...playback.reconcile(roomState, clockOffset), revision:++viewRevision};
  syncPreview();
  if (current()?.id !== previous) savePlayback();
  connectEvents();
  library = roomState.library || [];
  if (indexedEntries !== state.entries) {
    indexedEntries = state.entries;
    positionInQueue = queuePositions(state.entries);
  }
  if (indexedLibrary !== library) {
    indexedLibrary = library;
    libraryById = new Map(library.map(song => [song.id, song]));
    libraryBySource = new Map(library.map(song => [queueKey(song), song]));
  }
  if (!document.hidden) {
    if (renderedLibrary !== library) {
      renderedLibrary = library;
      renderLibrary();
      videoSearch.refreshLocal(searchSongs.some(song => {
        const cached = libraryBySource.get(queueKey(song));
        return song.status !== cached?.status || song.lyrics_need_review !== cached?.lyrics_need_review ||
          song.lyrics_unavailable !== cached?.lyrics_unavailable;
      }));
    }
    updatePreparation($('screen-preparation'), current());
    renderQueue(); updateTransport(); renderStorage();
    updateAddButtons();
  }
  if (!wasReady && state.playing && current()?.status === 'completed') switchView('player');
  if (state.started_at === null || player.anchor === null) updateTime(state.position);
  syncPlayback();
}

async function activate() {
  loadError = '';
  claimPlayback();
  const session = tabs.record.session;
  try { await player.unlock(); }
  catch { toast('浏览器还没有启用声音，请再点一次播放'); return false; }
  if (!tabs.owns(session)) { followPlayback(); return false; }
  return true;
}

async function togglePlay() {
  const wasUnlocked = tabs.owns() && player.unlocked;
  // Pause is an intent change, not a request to reactivate the audio device.
  if (state.playing && wasUnlocked) { await command('pause'); return; }
  if (!(await activate())) return;
  if (playback.allSkipped) { await command('play'); return; }
  if (state.playing && !wasUnlocked) { await syncPlayback(); return; }
  await command(state.playing ? 'pause' : 'play');
}

document.querySelectorAll('[data-view]').forEach(button => button.onclick = () => switchView(button.dataset.view));
document.querySelector('.brand').onclick = event => {
  event.preventDefault();
  navigation.push();
  $('song-input').value = ''; $('title-input').value = '';
  filter = 'all'; libraryPage = 1; videoSource = 'songs'; searchOrder = 'relevance'; catalogueSelection = null;
  clearVideoSearch(); renderLibrary(); switchView('library');
  $('song-input').focus({preventScroll:true});
};
document.querySelectorAll('[data-filter]').forEach(button => button.onclick = () => {
  filter = button.dataset.filter;
  libraryPage = 1;
  renderLibrary();
  navigation.save();
});
function changeLibraryPage(page) {
  navigation.push();
  libraryPage = page;
  renderLibrary();
  window.scrollTo(0, 0);
  navigation.save();
}
document.querySelectorAll('[data-library-step]').forEach(button => button.onclick = () =>
  changeLibraryPage(libraryPage + Number(button.dataset.libraryStep)));
document.querySelectorAll('.library-pagination select').forEach(select => select.onchange = () =>
  changeLibraryPage(Number(select.value)));
$('song-input').oninput = () => { if (view !== 'library') switchView('library'); editSearch(); };
$('more-artists').onclick = () => {
  changeSearchMode('songs', '');
  renderLibrary();
  $('song-input').focus({preventScroll:true});
};
document.querySelectorAll('[data-video-source]').forEach(button => button.onclick = () => changeSearchMode(button.dataset.videoSource));
$('search-order').onchange = () => { searchOrder = $('search-order').value; if (searchSnapshot) renderVideoResults(searchSnapshot); };
$('song-input').oncompositionstart = () => { composing = true; clearVideoSearch(); };
$('song-input').oncompositionend = () => { composing = false; editSearch(); };
$('clear-search').onclick = () => {
  if (videoSource === 'artists' && searchSnapshot?.artist?.selection) {
    if (!navigation.back()) videoSearch.selectArtist(searchSnapshot.artist.selection.browse_id);
  } else if (['artist','album'].includes(videoSource)) {
    if (!navigation.back()) changeSearchMode(videoSource === 'artist' ? 'songs' : catalogueSelection?.returnMode || 'songs');
  } else {
    navigation.push(); $('song-input').value = ''; editSearch(); $('song-input').focus();
  }
};
$('artist-summary').onclick = event => {
  if (!event.target.closest('[data-other-artists]')) return;
  navigation.push();
  videoSearch.showArtists();
  window.scrollTo(0, 0);
};
const searchObserver = typeof IntersectionObserver === 'function' ? new IntersectionObserver(entries => {
  const info = searchSnapshot?.[videoSource];
  if (entries.some(entry => entry.isIntersecting) && view === 'library' && info && !info.moreError && !info.autoBlocked) videoSearch.loadMore();
}, {rootMargin:'0px 0px 320px 0px'}) : null;
$('search-more-button').onclick = () => {
  if (searchSnapshot?.[videoSource]?.expired) searchVideos(true);
  else videoSearch.loadMore();
};
$('search-status').onclick = event => { if (event.target.closest('[data-search-retry]')) searchVideos(true); };
$('video-grid').onclick = event => {
  const catalogue = event.target.closest('[data-catalogue]');
  if (catalogue) { openCatalogue(searchSongs[catalogue.dataset.catalogue]); return; }
  const artist = event.target.closest('[data-artist-result], [data-artist-id]');
  if (artist) { openCatalogue(artist.dataset.artistId ? {browse_id:artist.dataset.artistId, title:artist.dataset.artistName} : searchSongs[artist.dataset.artistResult]); return; }
  const mv = event.target.closest('[data-find-mv]');
  if (mv) { const song = searchSongs[mv.dataset.findMv]; changeSearchMode('youtube', `${song.uploader} ${song.title}`); return; }
  const preview = event.target.closest('[data-preview]');
  const add = event.target.closest('[data-add-video]');
  if (preview) previewVideo(searchSongs[preview.dataset.preview]);
  if (add) addVideo(searchSongs[add.dataset.addVideo]);
};
$('preview-add').onclick = () => addVideo(previewSong);
$('close-preview').onclick = () => $('video-preview').close();
$('video-preview').onclose = () => { if (!$('video-preview').open) clearPreview(); };
$('video-preview').onclick = event => { if (event.target === $('video-preview')) $('video-preview').close(); };
$('add-form').onsubmit = async event => {
  event.preventDefault();
  if (composing) return;
  const url = $('song-input').value.trim();
  if (!/^https?:\/\//i.test(url)) { switchView('library'); searchVideos(); if (!url) $('song-input').focus(); return; }
  await addVideo({url}, {title:$('title-input').value.trim() || null});
};
$('song-grid').onclick = async event => {
  const button = event.target.closest('[data-song]');
  if (!button || button.disabled) return;
  const song = library.find(s => s.id === button.dataset.song);
  await addVideo(song, {title:song.title});
};
$('library-view').addEventListener('click', async event => {
  const homeArtist = event.target.closest('[data-home-artist]');
  if (homeArtist) { changeSearchMode('artists', homeArtist.dataset.homeArtist); return; }
  const artist = event.target.closest('[data-search-artist]');
  if (artist) { changeSearchMode('artists', artist.dataset.searchArtist); return; }
  const button = event.target.closest('[data-preview-song], [data-find-mv-song]');
  if (!button) return;
  const song = library.find(song => song.id === (button.dataset.previewSong || button.dataset.findMvSong));
  if (button.hasAttribute('data-preview-song')) previewVideo({...song, provider:'local'});
  else changeSearchMode('youtube', `${song.artist} ${song.title}`.trim());
});
$('queue-list').onclick = async event => {
  const button = event.target.closest('[data-action]');
  if (!button || button.disabled) return;
  const id = button.closest('[data-entry]').dataset.entry;
  const action = button.dataset.action;
  if (action === 'retry') return retryPreparation(id);
  if (action === 'preview') return previewVideo(state.entries.find(entry => entry.id === id));
  if (action === 'top') {
    const order = queueOrder(id);
    if (order && await command('reorder', {order, revision:roomState.revision})) showQueuedSong(id);
  } else {
    await command(action, {entry_id:id});
  }
};
for (const button of document.querySelectorAll('[data-queue-tab]')) {
  button.onclick = () => { queueTab = button.dataset.queueTab; renderQueue(); navigation.save(); };
  button.onkeydown = event => {
    if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 'upcoming' : event.key === 'End' ? 'history' : queueTab === 'upcoming' ? 'history' : 'upcoming';
    const tab = $(`${next}-tab`);
    tab.click(); tab.focus();
  };
}
for (const kind of ['queue', 'history']) {
  const nav = $(`${kind}-pagination`);
  const change = page => {
    if (kind === 'queue') queuePage = page; else historyPage = page;
    renderQueue(); navigation.save();
    $(`${kind}-list`).scrollTop = 0;
    $(`${kind === 'queue' ? 'upcoming' : 'history'}-tab`).scrollIntoView({block:'nearest'});
  };
  nav.querySelector('select').onchange = event => change(Number(event.target.value));
  for (const button of nav.querySelectorAll('[data-page-step]'))
    button.onclick = () => change((kind === 'queue' ? queuePage : historyPage) + Number(button.dataset.pageStep));
}
$('history-query').oninput = () => { historyPage = 1; renderHistory(); navigation.save(); };
$('history-list').onclick = async event => {
  const button = event.target.closest('[data-repeat-song]');
  if (!button || button.disabled) return;
  const song = state.history.find(song => song.id === button.closest('[data-history-entry]').dataset.historyEntry);
  button.disabled = true;
  try { await addVideo(song, {repeat:true}); }
  finally { button.disabled = false; }
};
$('play-toggle').onclick = togglePlay;
for (const [id, step] of [['pitch-down', -1], ['pitch-up', 1]]) {
  $(id).onclick = () => {
    pitchRequest = pitchRequest.then(async () => {
      if (!(await activate())) return;
      if (player.pitchError) { toast(player.pitchError.message); return; }
      const pitch = Math.max(-6, Math.min(6, (state.pitch_semitones || 0) + step));
      await command('pitch', {pitch_semitones:pitch});
    });
  };
}
$('screen-start').onclick = async () => {
  if (!(await activate())) return;
  if (!state.playing || playback.allSkipped) await command('play');
  else await syncPlayback();
};
$('screen-retry').onclick = () => retryPreparation(state.current_id);
const preparationRequests = new Set();
async function retryPreparation(id) {
  if (preparationRequests.has(id)) return;
  preparationRequests.add(id);
  document.querySelectorAll(`[data-entry="${CSS.escape(id)}"] [data-action="retry"]`).forEach(b => b.disabled = true);
  $('screen-retry').disabled = true;
  try {
    if (await command('retry', {entry_id:id})) toast('已提交重试');
  } finally {
    preparationRequests.delete(id);
    $('screen-retry').disabled = false;
    queueSignature = ''; renderQueue();
  }
}
function renderNext() {
  $('next-song').disabled = !state.current_id || skipping || (finishSending === playback && !queueSyncError);
}
bindSongButton($('next-song'), current, async song => {
  skipping = true; renderNext();
  try { await command('next', {entry_id:song.id, waiting:song.status !== 'completed'}); }
  finally { skipping = false; renderNext(); }
});
$('now-song').onclick = () => switchView('player');
bindSongButton($('transport-restart'), current, song => command('restart', {entry_id:song.id}));
bindSongButton($('skip-intro'), current, song => {
  const position = introSkipTarget();
  if (position !== null) command('seek', {entry_id:song.id, position});
});
$('lead-toggle').onclick = () => command('toggle_lead');
for (const [id, step] of [['lead-down', -5], ['lead-up', 5]]) {
  $(id).onclick = () => command('lead', {level:(Math.round(state.levels.lead_vocals * 100) + step) / 100});
}
$('storage-form').onsubmit = async event => {
  event.preventDefault();
  $('storage-save').disabled = true;
  try {
    acceptState(await request('/v1/storage', {limit_gib:Number($('storage-limit').value)}));
    toast('缓存上限已保存，旧缓存会自动清理');
  } catch (error) { toast(error.message); }
  finally { $('storage-save').disabled = false; }
};
const draggingSeek = bindSeek($('seek'), () => state.current_id,
  () => updateTime(player.position() ?? playback.position()),
  (position, entry_id) => command('seek', {entry_id, position}));
try { sessionStorage.removeItem('karaokeVolume'); } catch { /* Clearing an obsolete preference is optional. */ }
const stage = $('player-stage');
const transport = document.querySelector('.transport'), transportHome = transport.parentElement;
function setFullscreen(active) {
  if (stage.classList.contains('is-fullscreen') === active) return;
  stage.classList.toggle('is-fullscreen', active);
  for (const element of document.querySelectorAll('.topbar, .player-heading, #queue-panel')) element.inert = active;
  // Keep the video attached and move the one set of controls into the stage.
  (active ? stage : transportHome).append(transport);
  (active ? document.querySelector('.transport-playback') : $('screen')).append($('skip-intro'));
  (active ? stage : $('fullscreen')).focus({preventScroll:true});
}
async function fullscreen() {
  if (document.fullscreenElement === stage) { await document.exitFullscreen(); return; }
  if (stage.classList.contains('is-fullscreen')) { setFullscreen(false); return; }
  switchView('player');
  if (stage.requestFullscreen) {
    try { await stage.requestFullscreen(); return; } catch { /* Use the full page below. */ }
  }
  if (view === 'player') setFullscreen(true);
}
$('fullscreen').onclick = fullscreen;
$('exit-fullscreen').onclick = fullscreen;
document.addEventListener('fullscreenchange', () => setFullscreen(document.fullscreenElement === stage));
document.addEventListener('keydown', event => {
  if ($('video-preview').open) return;
  if (event.key === 'Escape' && stage.classList.contains('is-fullscreen')) { fullscreen(); return; }
  if (['INPUT','TEXTAREA','SELECT','SUMMARY'].includes(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.repeat) { if (event.code === 'Space') event.preventDefault(); return; }
  if (event.code === 'Space' && event.target.tagName !== 'BUTTON') { event.preventDefault(); togglePlay(); }
  if (event.key.toLowerCase() === 'n') $('next-song').click();
  if (event.key.toLowerCase() === 'f') fullscreen();
});
window.addEventListener('resize', () => switchView(view, {scroll:false}));
for (const [selector, variable] of [['.topbar', '--header-height'], ['.transport', '--transport-height']]) {
  new ResizeObserver(([entry]) => document.documentElement.style.setProperty(variable, `${entry.target.getBoundingClientRect().height}px`)).observe(document.querySelector(selector));
}
history.scrollRestoration = 'manual';
window.addEventListener('popstate', event => navigation.restore(event.state));
window.addEventListener('pagehide', () => navigation.save());
let scrollSave;
window.addEventListener('scroll', () => { clearTimeout(scrollSave); scrollSave = setTimeout(() => navigation.save(), 150); }, {passive:true});
const events = new RoomEvents(acceptState, online => {
  if (connected === online) return;
  connected = online; renderConnection();
  if (online) retryFinished();
});
function connectEvents() {
  // Protect an activated performance across pauses, but not an idle viewer's
  // automatic queue preselection (which has never loaded any media).
  const active = tabs.owns() && player.unlocked;
  events.connect(active ? playback.entry?.job_id || '' : '', playingEntryId());
}
function savePlayback() {
  lastPlaybackSave = Date.now();
  observePlayback();
  const saved = playback.save();
  if (!tabs.write(saved)) { followPlayback(); return; }
  tabChange = tabs.record.change;
  try { sessionStorage.setItem('karaokePlayback', JSON.stringify(saved)); } catch { /* Playback continues without tab persistence. */ }
}
function followPlayback() {
  const record = tabs.read();
  if (record.change === tabChange) return;
  tabChange = record.change;
  player.clear(); readySent = null;
  playback = new LocalPlayback(record.playback);
  queueSyncError = false;
  clearTimeout(finishRetry); finishRetry = null;
  renderState(); renderConnection();
}
function claimPlayback() {
  followPlayback();
  if (tabs.owns()) return false;
  if (receivedRoom) playback.reconcile(roomState, clockOffset);
  playback = new LocalPlayback(playback.save());
  if (!tabs.write(playback.save(), {claim:true})) { followPlayback(); return false; }
  tabChange = tabs.record.change;
  renderState();
  syncFinished();
  return true;
}
window.addEventListener('storage', event => { if (event.key === tabs.key) followPlayback(); });
window.addEventListener('online', () => { events.close(); connectEvents(); retryFinished(); });
window.addEventListener('pagehide', () => {
  if ($('video-preview').open) { $('video-preview').close(); clearPreview(); }
  observePlayback();
  if (tabs.suspend(playback.save())) player.suspend();
  events.close();
});
function resumePage() {
  const resumed = tabs.resume();
  followPlayback();
  connectEvents();
  renderState();
  if (resumed && playback.state.playing)
    player.unlock().then(() => renderState()).catch(() => updateTransport());
  retryFinished();
}
document.addEventListener('visibilitychange', () => { if (!document.hidden) resumePage(); });
window.addEventListener('pageshow', event => { if (event.persisted) resumePage(); });
connectEvents();
updateTransport();
if (!navigation.restore()) navigation.save();
// A refresh can retain playback intent, but the browser still owns permission
// to produce audio. Until unlock succeeds, no ready event starts the clock.
if (tabs.owns() && playback.state.playing) player.unlock().then(() => renderState()).catch(() => updateTransport());
