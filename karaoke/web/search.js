// Local results remain available while YouTube loads; stale responses cannot repaint.
export function createVideoSearch(onChange, fetcher = fetch, getLocal = () => []) {
  let controller, generation = 0, active;
  const albums = new Map();
  const lyricQueue = new Map();
  let lyricRequest;
  async function readPage(url, signal) {
    const response = await fetcher(url, {signal});
    const data = await response.json();
    if (!response.ok) throw Object.assign(new Error(data.detail || '搜索失败，请重试'), {expired:response.status === 410});
    if (!Array.isArray(data.results)) throw new Error('搜索结果暂不可用，请重试');
    return data;
  }
  function getAlbum(id) {
    if (!albums.has(id)) {
      const record = {};
      record.pending = readPage(`/v1/music/albums/${encodeURIComponent(id)}`).then(data => {
        record.data = data;
        return data;
      }).catch(error => { if (albums.get(id) === record) albums.delete(id); throw error; });
      albums.set(id, record);
      if (albums.size > 32) albums.delete(albums.keys().next().value);
    }
    return albums.get(id).pending;
  }
  function cancel() {
    generation++;
    controller?.abort();
    active = null;
    lyricQueue.clear();
  }
  function activate(query, mode, providers, autoSelectArtist = false) {
    cancel();
    for (const info of Object.values(providers)) info.loadingMore = false;
    controller = new AbortController();
    active = {query, mode, providers, autoSelectArtist, generation, signal:controller.signal};
    onChange(providers, mode);
    return active;
  }
  async function search(query, mode = 'youtube') {
    const providers = {local:{loading:false, results:getLocal(query), error:''}};
    if (mode !== 'local') providers[mode] = {loading:true, results:[], error:''};
    if (mode === 'artists') providers.artists.query = query;
    const request = activate(query, mode, providers, mode === 'artists');
    if (mode !== 'local') await loadPage(request, false);
  }
  async function selectArtist(id, refresh = false) {
    const providers = active?.providers;
    const artist = providers?.artists?.results.find(item => item.browse_id === id);
    if (!artist) return;
    const previous = providers.artist;
    if (refresh || previous?.selection?.browse_id !== id || previous.loading || previous.error) {
      providers.artist = {loading:true, results:[], error:'', metadata:artist, selection:artist};
    }
    const request = activate(id, 'artist', providers);
    if (providers.artist.loading) await loadPage(request, false);
  }
  function showArtists() {
    const providers = active?.providers;
    if (providers?.artists) activate(providers.artists.query, 'artists', providers);
  }
  async function loadMore() {
    const info = active?.providers[active.mode];
    if (!info?.nextCursor || info.loading || info.loadingMore) return;
    await loadPage(active, true);
  }
  async function loadPage(request, append) {
    const {query, mode, providers, signal} = request;
    const info = providers[mode];
    const cursor = append ? info.nextCursor : null;
    const notify = () => { if (request.generation === generation) onChange(providers, mode); };
    if (append) {
      info.loadingMore = true; info.moreError = ''; info.expired = false;
      notify();
    }
    try {
      const params = new URLSearchParams(mode === 'youtube' ? {q:query, provider:'youtube'} : {q:query, kind:mode});
      if (cursor) params.set('cursor', cursor);
      const url = ['artist','album'].includes(mode) ? `/v1/music/${mode}s/${encodeURIComponent(query)}${cursor ? '?'+new URLSearchParams({cursor}) : ''}`
        : `${mode === 'youtube' ? '/v1/search' : '/v1/music/search'}?${params}`;
      const data = mode === 'album' ? await getAlbum(query) : await readPage(url, signal);
      if (signal.aborted || request.generation !== generation) return;
      // Suppress only overlap with earlier pages, keeping distinct source versions.
      const seen = new Set(append ? info.results.map(song => song.browse_id || song.url).filter(Boolean) : []);
      const added = data.results.filter(song => !seen.has(song.browse_id || song.url));
      info.results = append ? [...info.results, ...added] : added;
      info.metadata = {...info.metadata, ...data};
      info.nextCursor = data.next_cursor && data.next_cursor !== cursor ? data.next_cursor : null;
      info.autoBlocked = append && added.length === 0;
      info.loading = false; info.loadingMore = false;
      info.error = ''; info.moreError = '';
      if (request.autoSelectArtist && !append && info.results[0]?.browse_id) {
        return selectArtist(info.results[0].browse_id);
      }
    } catch (error) {
      if (signal.aborted || request.generation !== generation) return;
      if (append) {
        info.loadingMore = false; info.moreError = error.message; info.expired = !!error.expired;
      } else {
        info.loading = false; info.error = error.message;
      }
    }
    notify();
  }
  function refreshLocal(sourceChanged = false) {
    if (!active) return;
    const results = getLocal(active.query);
    if (!sourceChanged && JSON.stringify(results) === JSON.stringify(active.providers.local.results)) return;
    active.providers.local = {loading:false, results, error:''};
    onChange(active.providers, active.mode);
  }
  function checkLyrics(url, visible = true) {
    const song = active?.providers[active.mode]?.results.find(song => song.url === url);
    const key = song && queueKey(song);
    if (!visible) { lyricQueue.delete(key); return; }
    if (!key?.startsWith('youtube:') || (song.kind && song.kind !== 'song') ||
        song.lyrics?.expires_at > Date.now()/1000 ||
        (lyricRequest?.request === active && lyricRequest.keys.includes(key))) return;
    // The view queues only visible rows. Never walk an entire artist catalogue.
    if (lyricQueue.size < 20) lyricQueue.set(key, active);
    queueMicrotask(drainLyrics);
  }
  async function drainLyrics() {
    if (lyricRequest || !active || !lyricQueue.size) return;
    const request = active;
    const keys = [...lyricQueue].filter(([,owner]) => owner === request).slice(0,8).map(([key]) => key);
    if (!keys.length) return;
    keys.forEach(key => lyricQueue.delete(key));
    lyricRequest = {request, keys};
    let results = {};
    try {
      const ids = keys.map(key => key.slice(8)).join(',');
      const response = await fetcher(`/v1/lyrics/availability?ids=${encodeURIComponent(ids)}`, {signal:request.signal});
      if (response.ok) results = (await response.json()).results || {};
    } catch { /* Unknown is not a claim that a song has no lyrics. */ }
    finally { lyricRequest = null; }
    if (request === active && !request.signal.aborted) {
      for (const song of request.providers[request.mode].results) {
        const key = queueKey(song), lyrics = results[key?.slice(8)];
        // A phone's wall clock need not agree with the server. Use a bounded
        // relative TTL and a short floor so expiry during transit cannot loop.
        if (keys.includes(key)) song.lyrics = ['available','unknown'].includes(lyrics?.status) && Number.isFinite(lyrics.cache_seconds) ?
          {...lyrics, expires_at:Date.now()/1000+Math.max(30,Math.min(3600,lyrics.cache_seconds))} :
          {status:'unknown', reason:'lookup_failed', expires_at:Date.now()/1000+60};
      }
      onChange(request.providers, request.mode);
    }
    drainLyrics();
  }
  function snapshot() {
    if (!active) return null;
    return structuredClone({query:active.query, mode:active.mode, providers:active.providers, autoSelectArtist:active.autoSelectArtist,
      albums:[...albums].filter(([,record]) => record.data).map(([id,record]) => [id,record.data])});
  }
  function restore(saved) {
    const info = saved?.providers?.[saved.mode];
    if (!['local','youtube','songs','artists','artist','album'].includes(saved?.mode) ||
        typeof saved?.query !== 'string' || !Array.isArray(info?.results)) return false;
    for (const [id,data] of saved.albums || []) albums.set(id, {data, pending:Promise.resolve(data)});
    while (albums.size > 32) albums.delete(albums.keys().next().value);
    const providers = structuredClone(saved.providers);
    if (saved.mode === 'artists') providers.artists.query ??= saved.query;
    const restart = info.loading && !info.results.length;
    providers[saved.mode].loading = restart;
    providers.local = {loading:false, results:getLocal(saved.query), error:''};
    const request = activate(saved.query, saved.mode, providers, saved.autoSelectArtist ?? (saved.mode === 'artists' && restart));
    if (restart) loadPage(request, false);
    return true;
  }
  return {search, selectArtist, showArtists, loadMore, cancel, refreshLocal, snapshot, restore, checkLyrics};
}

export function lyricLabel(song) {
  if (song.lyrics_unavailable) return {text:'歌词暂缺', available:false, title:'这版录音还没有可用字幕。点歌后会重新准备。'};
  if (song.lyrics_need_review) return {text:'歌词待校对', available:true, title:'已有歌词，正文或字幕时间还需校对。'};
  if (song.status === 'completed' || (song.lyrics?.status === 'available' && song.lyrics.expires_at > Date.now()/1000)) {
    return {text:'有歌词', available:true, title:'已找到这版录音的歌词。'};
  }
  return song.lyrics ? {text:'歌词未确认', available:false, title:'暂未确认这版录音的歌词，可能尚未收录或来源暂不可用。可以换一个版本；点歌后仍会尝试其他歌词来源。'} :
    {text:'歌词待查', available:false, title:'正在检查这版录音是否有已发表的歌词。'};
}

// One browser-history entry per navigation; refresh restores the same search scene.
export function createViewHistory(history, capture, apply) {
  let restoring = false;
  function save(push = false) {
    if (restoring) return;
    const depth = (history.state?.karaokeView?.depth || 0) + Number(push);
    try {
      history[push ? 'pushState' : 'replaceState']({...history.state,
        karaokeView:{version:1, depth, value:capture()}}, '');
    } catch { /* Browsers may disallow storage; navigation must remain usable. */ }
  }
  function push() { save(); save(true); }
  function restore(state = history.state) {
    const saved = state?.karaokeView;
    if (saved?.version !== 1 || !saved.value) return false;
    restoring = true;
    try { apply(saved.value); return true; }
    finally { restoring = false; }
  }
  function back() {
    if (!(history.state?.karaokeView?.depth > 0)) return false;
    history.back();
    return true;
  }
  return {save, push, restore, back};
}

export function filterSongs(songs, query, fields = ['title','artist']) {
  const normalize = text => String(text || '').normalize('NFKC').toLocaleLowerCase();
  const terms = normalize(query).trim().split(/\s+/);
  return songs.filter(song => terms.every(term => normalize(fields.map(field => song[field] || '').join(' ')).includes(term)));
}

export function historySuggestions(history) {
  const recent = new Map();
  const ordered = [...history].sort((a,b) => b.finished_at-a.finished_at);
  for (const song of ordered) {
    const key = queueKey(song);
    if (key && !recent.has(key)) recent.set(key, song);
  }
  return {recent:[...recent.values()]};
}

function artistKey(name) {
  return name.normalize('NFKC').trim().replace(/\s+/g, ' ').toLocaleLowerCase();
}

export function songArtists(song) {
  // Only the display separator we use is splittable; punctuation can be part of a name.
  const credits = song.artists?.length ? song.artists : (song.artist || '').split(/\s+·\s+/).map(name => ({name}));
  const unique = new Map();
  for (const credit of credits) {
    const name = credit.name.trim().replace(/\s+/g, ' '), key = artistKey(name);
    if (name && !unique.has(key)) unique.set(key, {...credit, name});
  }
  return [...unique.values()];
}

export function libraryArtists(library) {
  const artists = new Map();
  for (const song of librarySongs(library)) {
    const credits = songArtists(song);
    for (const {name} of credits) {
      const key = artistKey(name);
      if (!artists.has(key)) artists.set(key, {name, count:0, languages:new Set()});
      const artist = artists.get(key);
      artist.count++;
      if (credits.length === 1) artist.poster ||= song.poster;
      artist.languages.add(song.language);
    }
  }
  const ranked = [...artists.values()].sort((a,b) => b.count-a.count);
  const selected = new Set();
  for (const [language, count] of [['Chinese',8], ['English',4]]) {
    ranked.filter(artist => artist.languages.has(language) && !selected.has(artist))
      .slice(0,count).forEach(artist => selected.add(artist));
  }
  return [...selected, ...ranked.filter(artist => !selected.has(artist))].slice(0,12);
}

export function searchLibrary(library, query) {
  return librarySongs(library, 'all', query)
    .map(song => ({...song, provider:'local', thumbnail:song.poster, uploader:song.artist}));
}

export function artistLibrarySongs(library, name, catalogue = []) {
  const ready = librarySongs(library);
  const sources = new Map(catalogue.map(song => [queueKey(song), song]).filter(([key]) => key));
  const selected = artistKey(name), aliases = new Set([selected]);
  for (const song of ready) {
    const source = sources.get(queueKey(song));
    if (!source) continue;
    const local = songArtists(song), remote = songArtists(source);
    // A shared duet does not establish which local credit names the selected artist.
    if (local.length === 1 && remote.length === 1 && artistKey(remote[0].name) === selected)
      aliases.add(artistKey(local[0].name));
  }
  return ready.filter(song => sources.has(queueKey(song)) || songArtists(song).some(artist => aliases.has(artistKey(artist.name))));
}

export function librarySongs(library, filter = 'all', query = '') {
  return filterSongs(library, query).filter(song => readyToSing(song) &&
    (filter === 'all' || (filter === 'other' ?
      song.language && !['Chinese','Cantonese','English','Japanese','Korean'].includes(song.language) :
      song.language === filter)));
}

export function paginateSongs(songs, requestedPage = 1, size = 12) {
  const pages = Math.max(1, Math.ceil(songs.length / size));
  const page = Math.max(1, Math.min(pages, Number.isInteger(requestedPage) ? requestedPage : 1));
  const offset = (page - 1) * size;
  return {songs:songs.slice(offset, offset + size), page, pages, offset};
}

// Match the source, not its title: translated titles and URL aliases share a queue entry.
export function queueKey(song) {
  try {
    const url = new URL(song.url);
    const host = url.hostname.replace(/^(www|m|music)\./, '');
    const id = host === 'youtu.be' ? url.pathname.slice(1) : host === 'youtube.com'
      ? url.searchParams.get('v') || url.pathname.match(/^\/(?:shorts|live|embed)\/([^/]+)/)?.[1] : null;
    if (/^[\w-]{11}$/.test(id || '')) return `youtube:${id}`;
  } catch { /* A known job can still be queued without a source URL. */ }
  return song.url || (song.job_id ? `job:${song.job_id}` : null);
}

export function queuePosition(song, entries) {
  const key = queueKey(song);
  return entries.findIndex(entry => (key && queueKey(entry) === key) ||
    (song.job_id && entry.job_id === song.job_id)) + 1;
}

// Build once per queue snapshot, then reuse for every visible point-song button.
export function queuePositions(entries) {
  const sources = new Map(), jobs = new Map();
  entries.forEach((entry, index) => {
    const key = queueKey(entry);
    if (key && !sources.has(key)) sources.set(key, index + 1);
    if (entry.job_id && !jobs.has(entry.job_id)) jobs.set(entry.job_id, index + 1);
  });
  return song => {
    const position = Math.min(sources.get(queueKey(song)) || Infinity, jobs.get(song.job_id) || Infinity);
    return Number.isFinite(position) ? position : 0;
  };
}

export function playCount(value) {
  if (typeof value === 'number') return Number.isFinite(value) && value >= 0 ? value : null;
  const match = String(value || '').normalize('NFKC').replaceAll(',', '').trim().match(/^(?:播放次数:\s*)?(\d+(?:\.\d+)?)\s*([KMB千万萬亿億])?(?:\s*(?:plays|views|次观看|次播放))?$/i);
  return match ? Number(match[1])*({K:1e3, M:1e6, B:1e9, 千:1e3, 万:1e4, 萬:1e4, 亿:1e8, 億:1e8}[match[2]?.toUpperCase()] || 1) : null;
}

function readyToSing(song) {
  return song.status === 'completed' && !song.lyrics_unavailable;
}

export function orderResults(songs, order) {
  if (order === 'lyrics') {
    const rank = song => !lyricLabel(song).available ? 0 : song.lyrics_need_review ? 1 : 2;
    return [...songs].sort((a,b) => rank(b)-rank(a));
  }
  if (order === 'popular') return [...songs].sort((a,b) => (playCount(b.views) ?? -1)-(playCount(a.views) ?? -1));
  if (order === 'ready') return [...songs].sort((a,b) => Number(readyToSing(b))-Number(readyToSing(a)));
  if (order === 'title') return [...songs].sort((a,b) => a.title.localeCompare(b.title, 'zh-CN'));
  if (order === 'relevance') {
    // Keep provider order within each group; playable songs precede albums.
    const group = song => !song.kind || song.kind === 'song' ? 0 : song.kind === 'album' ? 1 : 2;
    const live = song => /(?:[[(（【]\s*live\b|\blive(?:\s+(?:at|in|from|on|version|performance|\d{4})\b|\s*[|/·—–-]|[\s)\]）】]*$))|[现現][场場]|演唱[会會]|演奏[会會]|ライブ/i.test(song.title || '') &&
      group(song) === 0;
    return [...songs].sort((a,b) => group(a)-group(b) || Number(live(a))-Number(live(b)));
  }
  return songs;
}

export function previewURL(song, origin) {
  try {
    const url = new URL(song.url);
    if (url.protocol === 'https:' && url.hostname === 'www.youtube.com' && !url.username && !url.password && !url.port) {
      const id = url.searchParams.get('v');
      if (/^[\w-]{11}$/.test(id || '')) return `https://www.youtube.com/embed/${id}?${new URLSearchParams({autoplay:'0', playsinline:'1', origin})}`;
    }
  } catch { /* Missing or retired source links have no embedded preview. */ }
  return '';
}
