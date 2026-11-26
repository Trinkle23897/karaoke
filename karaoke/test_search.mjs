import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {createVideoSearch, createViewHistory, previewURL, searchLibrary, artistLibrarySongs, librarySongs, paginateSongs, filterSongs, playCount, orderResults, queueKey, queuePosition, queuePositions, libraryArtists, historySuggestions, lyricLabel} from './web/search.js';

function setup() {
  const requests = [], snapshots = [];
  const search = createVideoSearch(state => snapshots.push(structuredClone(state)), (url, {signal}) =>
    new Promise(resolve => requests.push({url, signal, finish: (results, ok = true, next_cursor = null) => resolve({ok, json:async () => ok ? {results, next_cursor} : {detail:'Provider failed'}})})));
  return {search, requests, snapshots};
}
const flush = () => new Promise(resolve => setImmediate(resolve));

test('queue position index keeps the first matching source or job and refreshes after reorder', () => {
  const a={url:'https://www.youtube.com/watch?v=abcdefghijk',job_id:'a'};
  const b={url:'https://youtu.be/abcdefghijk',job_id:'b'};
  const c={url:'https://www.youtube.com/watch?v=lmnopqrstuv',job_id:'c'};
  const position=queuePositions([c,a,b]);
  assert.equal(position(b),2);
  assert.equal(position({job_id:'b'}),3);
  assert.equal(position({...a,job_id:'c'}),1);
  assert.equal(position({job_id:'unknown'}),0);
  assert.equal(queuePositions([b,c,a])(a),1);
  assert.equal(queuePositions([])(a),0);
});

test('library pages keep every song in order and clamp after the filtered library shrinks', () => {
  const songs = Array.from({length:29}, (_, id) => ({id}));
  const pages = [1,2,3].map(page => paginateSongs(songs, page));
  assert.deepEqual(pages.flatMap(page => page.songs), songs);
  assert.deepEqual(pages.map(page => [page.offset,page.songs.length]), [[0,12],[12,12],[24,5]]);
  assert.deepEqual(paginateSongs(songs,99),pages[2]);
  assert.equal(paginateSongs([...songs,{id:29}],2).page,2);
  assert.deepEqual(paginateSongs(songs.slice(0,4),3).songs,songs.slice(0,4));
  assert.equal(paginateSongs(songs.slice(0,4),3).page,1);
  assert.deepEqual(paginateSongs([],3),{songs:[],page:1,pages:1,offset:0});
  assert.equal(paginateSongs(songs,NaN).page,1);
  const queue = Array.from({length:10001}, (_, id) => ({id}));
  const last = paginateSongs(queue, 201, 50);
  assert.deepEqual(last.songs, [queue[10000]]);
  assert.equal(last.offset, 10000);
  assert.equal(paginateSongs(queue.slice(0,-1),201,50).page, 200);
  assert.deepEqual(Array.from({length:201}, (_, i) => paginateSongs(queue,i+1,50).songs).flat(), queue);
});

test('every local library view excludes missing or unfinished media, even when metadata is retained', () => {
  const songs = ['completed','video_evicted','audio_ready','source_ready','running','queued','failed','evicted','cancelled'].map((status,index) =>
    ({id:index,title:`Song ${index}`,status,language:index % 2 ? 'Japanese' : 'Chinese'}));
  assert.deepEqual(librarySongs(songs).map(song => song.id), [0]);
  assert.deepEqual(librarySongs(songs, 'all').map(song => song.id), [0]);
  assert.deepEqual(librarySongs(songs, 'Japanese'), []);
  assert.deepEqual(librarySongs(songs, 'all', 'Song 1'), []);
  assert.deepEqual(searchLibrary(songs, 'Song').map(song => song.id), [0]);
  assert.deepEqual(searchLibrary(songs, 'Song 7'), []);
  const languages = ['Chinese', 'Cantonese', 'English', 'Japanese', 'Korean', 'French', null];
  const multilingual = languages.map(language => ({title:'Song', language, status:'completed'}));
  assert.deepEqual(librarySongs(multilingual, 'Cantonese').map(s => s.language), ['Cantonese']);
  assert.deepEqual(librarySongs(multilingual, 'Chinese').map(s => s.language), ['Chinese']);
  assert.deepEqual(librarySongs(multilingual, 'other').map(s => s.language), ['French']);
  assert.equal(librarySongs(multilingual, 'all').length, languages.length);
});

test('queued feedback follows source identity and room order, including unprepared songs', () => {
  const song = {url:'https://www.youtube.com/watch?v=abcdefghijk', title:'Original title', job_id:'job'};
  const other = {url:'https://www.youtube.com/watch?v=12345678901', title:song.title, status:'queued'};
  const queue = [other, {...song, title:'Translated title', status:'running'}];
  assert.equal(queuePosition(song,queue),2);
  assert.equal(queuePosition(song,[queue[1],queue[0]]),1);
  assert.equal(queuePosition(song,[other]),0);
  assert.equal(queuePosition({job_id:'job'},queue),2);
  for (const url of ['https://youtu.be/abcdefghijk?t=20', 'https://m.youtube.com/watch?v=abcdefghijk&list=album',
    'https://music.youtube.com/watch?v=abcdefghijk', 'https://www.youtube.com/shorts/abcdefghijk',
    'https://www.youtube.com/live/abcdefghijk',
    'https://www.youtube.com/embed/abcdefghijk']) {
    assert.equal(queuePosition({url},queue),2);
  }
  assert.equal(queuePosition({url:'https://youtube.com.attacker.test/watch?v=abcdefghijk'},queue),0);
  assert.equal(queuePosition({title:song.title},queue),0);
  assert.equal(queuePosition(song,[...queue, song]),2);
  assert.equal(queueKey(song),queueKey({...song, title:'Different title'}));
  assert.equal(queue[0],other);
});

test('history search matches song and artist while preserving performance order and repeat performances', () => {
  const songs = [
    {id:'recent',title:'Ｌｅｍｏｎ',artist:'米津玄師'},
    {id:'other',title:'Unravel',artist:'TK'},
    {id:'earlier',title:'Lemon',artist:'米津玄師',status:'completed'},
  ];
  assert.deepEqual(filterSongs(songs,' lemon 米津 '), [songs[0],songs[2]]);
  assert.deepEqual(filterSongs(songs,''), songs);
  assert.deepEqual(filterSongs(songs,'tk'), [songs[1]]);
  assert.deepEqual(filterSongs(songs,'missing'), []);
  assert.equal(songs[0].id,'recent');
});

test('artist-page local songs match artist credits rather than song titles or unavailable cache', () => {
  const songs = [
    {id:'solo',title:'One',artist:'ＡＤＥＬＥ',status:'completed'},
    {id:'credit',title:'Two',artist:'Adele, Guest Vocalist',artists:[{name:'Adele'}, {name:'Guest Vocalist'}],status:'completed'},
    {id:'title',title:'Adele',artist:'Other',status:'completed'},
    {id:'missing',title:'Three',artist:'Adele',status:'video_evicted'},
    {id:'other',title:'Four',artist:'Other',status:'completed'},
    {id:'similar',title:'Five',artist:'NotAdele',status:'completed'},
    {id:'longer',title:'Six',artist:'Adele Tribute',status:'completed'},
  ];
  assert.deepEqual(artistLibrarySongs(songs,'adele').map(s => s.id), ['solo','credit']);
  assert.deepEqual(artistLibrarySongs(songs,'other').map(s => s.id), ['title','other']);
  assert.deepEqual(artistLibrarySongs(songs,'unknown'), []);
  assert.deepEqual(artistLibrarySongs(songs,''), []);
  const translated = [
    {id:'shared',artist:'Saved credit',url:'https://youtu.be/abcdefghijk',status:'completed'},
    {id:'same-credit',artist:'Saved credit',url:'https://youtu.be/12345678901',status:'completed'},
    {id:'different',artist:'Different',status:'completed'},
    {id:'evicted',artist:'Saved credit',status:'video_evicted'},
  ];
  const catalogue = [{url:'https://music.youtube.com/watch?v=abcdefghijk',artists:[{name:'Translated credit'}]}];
  assert.deepEqual(artistLibrarySongs(translated,'Translated credit',catalogue).map(s => s.id), ['shared','same-credit']);
  assert.deepEqual(artistLibrarySongs(translated,'Unrelated',[]), []);
  assert.deepEqual(artistLibrarySongs(translated,'Translated credit',[{url:catalogue[0].url}]).map(s => s.id), ['shared']);
  const duet = [{...catalogue[0],artists:[{name:'Translated credit'},{name:'Guest'}]}];
  assert.deepEqual(artistLibrarySongs(translated,'Translated credit',duet).map(s => s.id), ['shared']);
  const jointCredits = translated.map(song => ({...song,artist:'Lead · Guest'}));
  assert.deepEqual(artistLibrarySongs(jointCredits,'Translated credit',catalogue).map(s => s.id), ['shared']);
});

test('recent songs use the latest performance of each source', () => {
  const songs = [
    {id:'old',url:'https://youtu.be/abcdefghijk',title:'Old title',artist:'ＴＫ',finished_at:10},
    {id:'cover',url:'https://www.youtube.com/watch?v=12345678901',title:'Same title',artist:'Cover artist',finished_at:30},
    {id:'latest',url:'https://music.youtube.com/watch?v=abcdefghijk',title:'Same title',artist:'TK',finished_at:50},
    {id:'other',url:'https://youtu.be/00000000000',title:'Other',artist:'Another artist',finished_at:40},
    {id:'unknown',url:'https://youtu.be/11111111111',title:'Unknown',artist:'',finished_at:20},
  ];
  const original = structuredClone(songs);
  const result = historySuggestions(songs);
  assert.deepEqual(result.recent.map(s => s.id), ['latest','other','cover','unknown']);
  assert.deepEqual(songs, original);
  assert.deepEqual(historySuggestions([]), {recent:[]});
});

test('artist shelf dynamically selects twelve ready-library credits without requiring history', () => {
  const songs = [
    {title:'A', artist:'ＴＫ', status:'completed'},
    {title:'B', artist:' TK ', poster:'/cover.jpg', status:'completed'},
    {title:'C', artist:'Solo & Band', status:'completed'},
    {title:'D', artist:'Solo', status:'completed'},
    {title:'E', artist:'Unavailable', status:'video_evicted'},
    {title:'F', artist:'Unavailable', status:'running'},
    {title:'G', artist:'', status:'completed'},
    ...Array.from({length:13}, (_,i) => ({artist:`Artist ${i}`,status:'completed'})),
  ];
  const original = structuredClone(songs);
  const artists = libraryArtists(songs);
  assert.deepEqual(artists.map(a => a.name), ['ＴＫ','Solo & Band','Solo', ...Array.from({length:9}, (_,i) => `Artist ${i}`)]);
  const added = [...songs, ...Array.from({length:3}, () => ({artist:'New artist',status:'completed'}))];
  assert.equal(libraryArtists(added)[0].name, 'New artist');
  assert.equal(libraryArtists(added.filter(s => s.artist !== 'New artist'))[0].name, 'ＴＫ');
  assert.equal(artists[0].count, 2);
  assert.equal(artists[0].poster, '/cover.jpg');
  assert.equal(artists[1].count, 1);
  assert.deepEqual(libraryArtists(songs.slice(4,7)), []);
  assert.deepEqual(libraryArtists([]), []);
  assert.deepEqual(songs, original);
});

test('artist shelf starts with eight Chinese and four English credits using song language', () => {
  const songs = [
    ...Array.from({length:10}, (_,i) => ({artist:`Chinese ${i}`,language:'Chinese',status:'completed'})),
    ...Array.from({length:6}, (_,i) => ({artist:`英文 ${i}`,language:'English',status:'completed'})),
    ...Array.from({length:20}, () => ({artist:'Other',language:'Japanese',status:'completed'})),
    {artist:'Chinese 0',language:'English',status:'completed'},
    {artist:'Unavailable',language:'English',status:'running'},
  ];
  assert.deepEqual(libraryArtists(songs).map(a => a.name), [
    ...Array.from({length:8}, (_,i) => `Chinese ${i}`),
    ...Array.from({length:4}, (_,i) => `英文 ${i}`),
  ]);
  const sparse = [songs[0],songs[10],songs[16]];
  assert.deepEqual(libraryArtists(sparse).map(a => a.name), ['Chinese 0','英文 0','Other']);
  assert.deepEqual(libraryArtists([...songs, ...Array.from({length:3}, () => songs[9])])[0].name, 'Chinese 9');
});

test('collaborators have separate artist entries while punctuation within a name stays intact', () => {
  const songs = [
    {id:'duet',artist:'Lead · Guest',poster:'/duet.jpg',status:'completed'},
    {id:'solo',artist:'Lead',poster:'/solo.jpg',status:'completed'},
    {id:'duplicate',artist:'Guest',artists:[{name:'Guest'}, {name:'Ｇｕｅｓｔ'}],status:'completed'},
    {id:'band',artist:'Combined display',artists:[{name:'Orchestra · North'}],status:'completed'},
    ...['Earth, Wind & Fire','AC/DC','Simon & Garfunkel','Artist·Name'].map(artist => ({artist,status:'completed'})),
  ];
  const original = structuredClone(songs);
  const artists = libraryArtists(songs);
  assert.deepEqual(artists.map(a => a.name), ['Lead','Guest','Orchestra · North','Earth, Wind & Fire','AC/DC','Simon & Garfunkel','Artist·Name']);
  assert.equal(artists[0].count, 2);
  assert.equal(artists[0].poster, '/solo.jpg');
  assert.equal(artists[1].count, 2);
  assert.equal(artists[1].poster, undefined);
  assert.deepEqual(artistLibrarySongs(songs,'Lead').map(s => s.id), ['duet','solo']);
  assert.deepEqual(artistLibrarySongs(songs,'Guest').map(s => s.id), ['duet','duplicate']);
  assert.deepEqual(artistLibrarySongs(songs,'Orchestra · North').map(s => s.id), ['band']);
  assert.deepEqual(artistLibrarySongs(songs,'North'), []);
  assert.deepEqual(songs, original);
});

test('search calls only YouTube and preserves every returned result in order', async () => {
  const {search, requests, snapshots} = setup();
  const pending = search.search('Some song & singer');
  assert.equal(requests.length, 1);
  const query = new URL(requests[0].url, 'http://localhost').searchParams;
  assert.equal(query.get('q'), 'Some song & singer');
  assert.equal(query.get('provider'), 'youtube');
  const results = Array.from({length:35}, (_,i) => ({title:`Video ${i}`}));
  requests[0].finish(results);
  await pending;
  assert.deepEqual(snapshots.at(-1).youtube.results, results);
  assert.equal(snapshots.at(-1).youtube.loading, false);
  const failed = search.search('failure');
  requests[1].finish([], false);
  await failed;
  assert.equal(snapshots.at(-1).youtube.error, 'Provider failed');
});

test('new queries and clearing the input discard stale responses even if abort is ignored', async () => {
  const {search, requests, snapshots} = setup();
  const old = search.search('old');
  const latest = search.search('new');
  assert.ok(requests[0].signal.aborted);
  const before = snapshots.length;
  requests[0].finish([{title:'stale'}]);
  await old;
  assert.equal(snapshots.length, before);
  requests[1].finish([{title:'current'}]);
  await latest;
  assert.equal(snapshots.at(-1).youtube.results[0].title, 'current');
  const cleared = search.search('cleared');
  search.cancel();
  const count = snapshots.length;
  requests[2].finish([]);
  await cleared;
  assert.equal(snapshots.length, count);
});

test('preview uses provider embeds, never arbitrary pages or autoplay', () => {
  const yt = previewURL({provider:'youtube', url:'https://www.youtube.com/watch?v=n5sjpCQhTDk'}, 'http://localhost:8746');
  assert.equal(new URL(yt).pathname, '/embed/n5sjpCQhTDk');
  assert.equal(new URL(yt).searchParams.get('autoplay'), '0');
  assert.equal(new URL(yt).searchParams.get('origin'), 'http://localhost:8746');
  assert.equal(previewURL({url:'https://www.youtube.com/watch?v=n5sjpCQhTDk'}, 'http://localhost:8746'), yt);
  const bili = previewURL({provider:'bilibili', url:'https://www.bilibili.com/video/BV1bK411W797'}, '');
  assert.equal(bili, '');
  assert.equal(previewURL({provider:'youtube', url:'https://www.youtube.com.attacker.test/watch?v=n5sjpCQhTDk'}, ''), '');
  assert.equal(previewURL({url:'not a URL'}, ''), '');
  assert.equal(previewURL({url:'https://user:secret@www.youtube.com/watch?v=n5sjpCQhTDk'}, ''), '');
});

test('local matches appear only after completion and disappear on eviction or restore', async () => {
  let library = [{url:'local', title:'弯弯的月亮', artist:'刘欢', status:'running'}];
  const snapshots = [], requests = [];
  const search = createVideoSearch(value => snapshots.push(structuredClone(value)), () => new Promise(resolve => requests.push(resolve)), query => searchLibrary(library, query));
  const pending = search.search('刘欢 月亮', 'songs');
  assert.deepEqual(snapshots[0].local.results, []);
  assert.equal(snapshots[0].songs.loading, true);
  assert.deepEqual(snapshots[0].songs.results, []);
  library[0].status = 'completed';
  search.refreshLocal();
  assert.equal(snapshots.at(-1).local.results[0].status, 'completed');
  requests.forEach(resolve => resolve({ok:false,status:503,json:async () => ({detail:'暂时不可用'})}));
  await pending;
  assert.equal(snapshots.at(-1).local.results.length, 1);
  assert.equal(snapshots.at(-1).songs.error, '暂时不可用');
  assert.equal(snapshots.at(-1).local.results[0].status, 'completed');
  const restored = createVideoSearch(() => {}, () => assert.fail('Completed search should not refetch'), query => searchLibrary(library, query));
  library[0].poster = '/new-poster.jpg';
  assert.equal(restored.restore(search.snapshot()), true);
  assert.equal(restored.snapshot().providers.local.results[0].thumbnail, '/new-poster.jpg');
  library[0].status = 'evicted';
  search.refreshLocal();
  assert.deepEqual(snapshots.at(-1).local.results, []);
  assert.equal(restored.restore(restored.snapshot()), true);
  assert.deepEqual(restored.snapshot().providers.local.results, []);
  search.cancel();
  const count = snapshots.length;
  search.refreshLocal();
  assert.equal(snapshots.length, count);
});

test('local matches and online pages remain independent, including the same recording', async () => {
  let library = [{url:'https://www.youtube.com/watch?v=abcdefghijk', title:'First Love', artist:'宇多田光',
    status:'completed', job_id:'cached', poster:'/poster.jpg'}];
  const remote = [
    {url:'https://www.youtube.com/watch?v=zyxwvutsrqp', title:'First Love', uploader:'Another artist', provider:'youtube_music'},
    {url:library[0].url, title:'First Love', provider:'youtube_music', available:false},
    ...Array.from({length:35}, (_,i) => ({url:`online-${i}`,title:`Result ${i}`,provider:'youtube_music'})),
  ];
  const later = [{url:'later-page', title:'Another First Love'}];
  const pages = [{results:remote, next_cursor:'next'}, {results:later}];
  const search = createVideoSearch(() => {}, async () => ({ok:true, json:async () => pages.shift()}),
    query => searchLibrary(library, query));
  await search.search('first love', 'songs');
  let {providers} = search.snapshot();
  assert.deepEqual(providers.local.results, searchLibrary(library, 'first love'));
  assert.deepEqual(providers.songs.results, remote);
  assert.equal(providers.songs.nextCursor, 'next');
  await search.loadMore();
  ({providers} = search.snapshot());
  assert.deepEqual(providers.songs.results, [...remote, ...later]);
  assert.equal(providers.local.results.length, 1);
  library = [];
  search.refreshLocal();
  ({providers} = search.snapshot());
  assert.deepEqual(providers.local.results, []);
  assert.deepEqual(providers.songs.results, [...remote, ...later]);
});

test('artist searches open the first provider result directly without matching its displayed name', async () => {
  const {search, requests, snapshots} = setup();
  const song = search.search('Singer', 'songs');
  assert.equal(new URL(requests[0].url, 'http://localhost').searchParams.get('kind'), 'songs');
  const artist = search.search('Singer', 'artists');
  assert.ok(requests[0].signal.aborted);
  requests[1].finish([{title:'Translated name',browse_id:'UC'+'a'.repeat(22)}]);
  await flush();
  assert.equal(requests[2].url, '/v1/music/artists/UC'+'a'.repeat(22));
  assert.equal(snapshots.at(-1).artist.metadata.title, 'Translated name');
  assert.equal(search.snapshot().mode, 'artist');
  requests[2].finish(Array.from({length:150}, (_,i) => ({title:`Song ${i}`})));
  await artist;
  const count = snapshots.length;
  requests[0].finish([{title:'stale song'}]);
  await song;
  assert.equal(snapshots.length, count);
  assert.equal(snapshots.at(-1).artist.results.length, 150);
  const album = search.search('MPREexample', 'album');
  assert.equal(requests[3].url, '/v1/music/albums/MPREexample');
  requests[3].finish([{title:'An album track',url:'album-track'}]);
  await album;
  assert.equal(snapshots.at(-1).album.results[0].title, 'An album track');
  await search.search('Singer', 'local');
  assert.equal(requests.length, 4);
});

test('popularity uses reported counts, keeps unknowns last and never mutates provider order', () => {
  const songs = [{title:'A',views:null},{title:'B',views:'21K'},{title:'C',views:'1.2M'},
    {title:'D',views:'1,300',status:'completed'},{title:'E',views:'21K'}];
  assert.deepEqual(orderResults(songs,'relevance'), songs);
  assert.deepEqual(orderResults(songs,'popular').map(s=>s.title), ['C','B','E','D','A']);
  assert.deepEqual(orderResults(songs,'ready').map(s=>s.title), ['D','A','B','C','E']);
  assert.deepEqual(songs.map(s=>s.title), ['A','B','C','D','E']);
  for (const value of [null, undefined, '', 'unknown', -1, Infinity, '2001年', '3:21']) assert.equal(playCount(value), null);
  assert.equal(playCount('0'), 0);
  assert.equal(playCount('1.2B'), 1.2e9);
  assert.equal(playCount('播放次数：5163万'), 51630000);
  assert.equal(playCount('3443万次观看'), 34430000);
  assert.equal(playCount('52M plays'), 52000000);
  for (const [value, expected] of [['2591万',25910000], ['1.2亿',120000000], ['２.５萬',25000], ['3千',3000], ['1.2 億',120000000]]) {
    assert.equal(playCount(value), expected);
  }
  assert.deepEqual(orderResults([{views:'2.5M'}, {views:'2591万'}, {views:'1.2亿'}], 'popular').map(s=>s.views), ['1.2亿','2591万','2.5M']);
  assert.ok(previewURL({provider:'youtube_music',url:'https://www.youtube.com/watch?v=abcdefghijk'},'http://localhost').includes('/embed/abcdefghijk'));
});

test('album browsing reuses successful metadata and retries failed loads', async () => {
  const {search,requests,snapshots} = setup();
  let pending = search.search('MPREalbum','album');
  requests[0].finish([],false);
  await pending;
  pending = search.search('MPREalbum','album');
  requests[1].finish([{kind:'song',url:'one',title:'One'}]);
  await pending;
  await search.search('anything','local');
  await search.search('MPREalbum','album');
  assert.equal(requests.length,2);
  const saved = JSON.parse(JSON.stringify(search.snapshot()));
  const restored = createVideoSearch(() => {}, () => assert.fail('Album metadata fetched again'));
  assert.ok(restored.restore(saved));
  await restored.search('MPREalbum','album');
  assert.equal(snapshots.at(-1).album.results[0].url,'one');
});

test('restored search keeps loaded pages and cursor, and ignores old responses', async () => {
  const {search,requests,snapshots} = setup();
  let pending = search.search('Song','songs');
  requests[0].finish([{url:'one',title:'One'}],true,'page-two');
  await pending;
  pending = search.loadMore();
  const saved = JSON.parse(JSON.stringify(search.snapshot()));
  assert.ok(search.restore(saved));
  requests[1].finish([{url:'stale',title:'Stale'}]);
  await pending;
  assert.deepEqual(snapshots.at(-1).songs.results.map(s=>s.url),['one']);
  assert.equal(snapshots.at(-1).songs.loadingMore,false);
  pending = search.loadMore();
  assert.equal(new URL(requests[2].url,'http://localhost').searchParams.get('cursor'),'page-two');
  requests[2].finish([{url:'two',title:'Two'}]);
  await pending;
  const again = setup();
  assert.ok(again.search.restore(JSON.parse(JSON.stringify(search.snapshot()))));
  assert.equal(again.requests.length,0);
  assert.deepEqual(again.snapshots.at(-1).songs.results.map(s=>s.url),['one','two']);
  assert.equal(again.search.restore(null),false);
});

test('reloading an unfinished search restarts it instead of preserving a spinner', async () => {
  const first = setup();
  const pending = first.search.search('Pending','songs');
  const next = setup();
  assert.ok(next.search.restore(JSON.parse(JSON.stringify(first.search.snapshot()))));
  assert.equal(next.requests.length,1);
  next.requests[0].finish([{url:'fresh'}]);
  await flush();
  assert.equal(next.snapshots.at(-1).songs.loading,false);
  first.requests[0].finish([]);
  await pending;
});

test('view history restores nested catalogues, reloads and forward navigation without losing the parent', () => {
  const stack = [null]; let index=0;
  const history = {get state(){return stack[index];},
    replaceState(value){stack[index]=structuredClone(value);},
    pushState(value){stack.splice(++index,stack.length,structuredClone(value));},
    back(){index--;}};
  let view={query:'唯一',mode:'songs',order:'popular',scroll:870,pages:['one','two']};
  let navigation=createViewHistory(history,()=>view,value=>{view=structuredClone(value);navigation.save();});
  navigation.save();
  assert.equal(navigation.back(),false);
  navigation.push();
  view={query:'唯一',mode:'album',selection:'MPREalbum',scroll:0}; navigation.save();
  navigation.push();
  view={query:'唯一',mode:'artist',selection:'UCsinger',scroll:0}; navigation.save();
  // A fresh controller models a page reload with only browser history retained.
  navigation=createViewHistory(history,()=>view,value=>{view=structuredClone(value);navigation.save();});
  view={}; assert.ok(navigation.restore()); assert.equal(view.mode,'artist');
  assert.ok(navigation.back()); navigation.restore(); assert.equal(view.selection,'MPREalbum');
  assert.ok(navigation.back()); navigation.restore();
  assert.deepEqual(view,{query:'唯一',mode:'songs',order:'popular',scroll:870,pages:['one','two']});
  index++; navigation.restore(); assert.equal(view.mode,'album');
  const unavailable=createViewHistory({replaceState(){throw new Error('blocked');}},()=>view,()=>{});
  assert.doesNotThrow(()=>unavailable.save()); assert.equal(unavailable.back(),false);
});

test('each tab restores its own search and navigation after another tab changes theirs', () => {
  function tab(value) {
    const history = {state:null,replaceState(next){this.state=structuredClone(next);}};
    const local = {value,history};
    local.navigation = createViewHistory(history,()=>local.value,next=>{local.value=next;});
    return local;
  }
  const alice = tab({query:'Lemon',videoSource:'songs',searchOrder:'popular',scroll:340,queueTab:'upcoming',libraryPage:3});
  const bob = tab({query:'周杰伦',videoSource:'artists',searchOrder:'relevance',scroll:90,queueTab:'history',libraryPage:1});
  alice.navigation.save(); bob.navigation.save();
  alice.value.query='First Love'; alice.value.scroll=700; alice.navigation.save();
  for (const local of [alice,bob]) {
    local.value={};
    const reloaded=createViewHistory(local.history,()=>local.value,next=>{local.value=next;});
    assert.equal(reloaded.restore(),true);
  }
  assert.deepEqual(alice.value,{query:'First Love',videoSource:'songs',searchOrder:'popular',scroll:700,queueTab:'upcoming',libraryPage:3});
  assert.deepEqual(bob.value,{query:'周杰伦',videoSource:'artists',searchOrder:'relevance',scroll:90,queueTab:'history',libraryPage:1});
});

test('default puts songs before albums and lowers live songs without removing results or changing other sorts', () => {
  const titles = ['A (Live)', 'Studio', 'The One and Only', 'B LIVE at Wembley', 'C 现场版',
    'Alive', 'Live and Let Die', 'D【LIVE】', 'E ライブ', 'Another studio', 'F Live | Tour'];
  const songs = titles.map((title,i) => ({title,views:100-i,kind:i === 2 ? 'album' : 'song'}));
  assert.deepEqual(orderResults(songs,'relevance').map(s=>s.title), [
    'Studio','Alive','Live and Let Die','Another studio',
    'A (Live)','B LIVE at Wembley','C 现场版','D【LIVE】','E ライブ','F Live | Tour','The One and Only']);
  assert.deepEqual(orderResults(songs,'popular'), songs);
  assert.deepEqual(orderResults(songs,'album'), songs);
  assert.deepEqual(songs.map(s=>s.title), titles);
  const mixed = [
    {kind:'artist',title:'Live'}, {kind:'album',title:'Live at Wembley'},
    {kind:'song',title:'Song (Live)'}, {kind:'album',title:'Studio album'},
    {title:'Local or video result'}, {kind:'playlist',title:'Playlist'}, {kind:'song',title:'Studio song'}];
  assert.deepEqual(orderResults(mixed,'relevance'), [mixed[4],mixed[6],mixed[2],mixed[1],mixed[3],mixed[0],mixed[5]]);
  assert.deepEqual(orderResults(mixed,'ready'), mixed);
  assert.deepEqual(orderResults(mixed,'title'), [mixed[0],mixed[1],mixed[4],mixed[5],mixed[2],mixed[3],mixed[6]]);
});


test('pages append once, retain distinct versions, preserve rows on failure and stop at the end', async () => {
  const {search, requests, snapshots} = setup();
  const first = search.search('Singer', 'songs');
  const rows = [{url:'studio',title:'Song'}, {url:'live',title:'Song (Live)'}];
  requests[0].finish(rows, true, 'page-2');
  await first;
  const more = search.loadMore();
  await search.loadMore();
  assert.equal(requests.length, 2);
  assert.equal(new URL(requests[1].url, 'http://localhost').searchParams.get('cursor'), 'page-2');
  assert.deepEqual(snapshots.at(-1).songs.results, rows);
  requests[1].finish([], false);
  await more;
  assert.deepEqual(snapshots.at(-1).songs.results, rows);
  assert.equal(snapshots.at(-1).songs.moreError, 'Provider failed');
  const retry = search.loadMore();
  assert.equal(requests[2].url, requests[1].url);
  requests[2].finish([rows[1], {url:'cover',title:'Song'}], true, 'page-3');
  await retry;
  assert.deepEqual(snapshots.at(-1).songs.results.map(row=>row.url), ['studio','live','cover']);
  assert.equal(snapshots.at(-1).songs.moreError, '');
  const last = search.loadMore();
  requests[3].finish([{url:'last',title:'Last song'}]);
  await last;
  await search.loadMore();
  assert.equal(requests.length, 4);
  assert.equal(snapshots.at(-1).songs.results.length, 4);
  assert.equal(snapshots.at(-1).songs.nextCursor, null);
});

test('changing query or mode cancels pending pagination and an empty page never loops automatically', async () => {
  const {search, requests, snapshots} = setup();
  const first = search.search('Singer', 'songs');
  requests[0].finish([{url:'first'}], true, 'page-2');
  await first;
  const more = search.loadMore();
  const artists = search.search('Singer', 'artists');
  assert.ok(requests[1].signal.aborted);
  const count = snapshots.length;
  requests[1].finish([{url:'stale song'}], true, 'page-3');
  await more;
  assert.equal(snapshots.length, count);
  requests[2].finish([{browse_id:'artist-one'}], true, 'artist-next');
  await flush();
  requests[3].finish([{url:'first-song'}]);
  await artists;
  search.showArtists();
  const empty = search.loadMore();
  requests[4].finish([], true, 'artist-next-again');
  await empty;
  assert.equal(snapshots.at(-1).artists.autoBlocked, true);
  assert.equal(snapshots.at(-1).artists.nextCursor, 'artist-next-again');
  assert.equal(requests.length, 5);
  const manual = search.loadMore();
  requests[5].finish([{browse_id:'artist-two'}]);
  await manual;
  assert.equal(snapshots.at(-1).artists.results.length, 2);
  await search.search('local', 'local');
  await search.loadMore();
  assert.equal(requests.length, 6);
});

test('artist choices retain provider order and pagination without automatically reopening the first artist', async () => {
  const {search, requests, snapshots} = setup();
  const pending = search.search('Common name', 'artists');
  const first = {title:'First result',browse_id:'artist-one'};
  const second = {title:'Common name',browse_id:'artist-two'};
  requests[0].finish([first,second], true, 'more-artists');
  await flush();
  requests[1].finish([{url:'one'}], true, 'more-songs');
  await pending;
  const more = search.loadMore();
  assert.equal(requests[2].url, '/v1/music/artists/artist-one?cursor=more-songs');
  requests[2].finish([{url:'two'}]);
  await more;
  search.showArtists();
  assert.equal(search.snapshot().mode, 'artists');
  assert.deepEqual(snapshots.at(-1).artists.results, [first,second]);
  const saved = JSON.parse(JSON.stringify(search.snapshot()));
  assert.ok(search.restore(saved));
  assert.equal(requests.length, 3);
  const cancelled = search.loadMore();
  const params = new URL(requests[3].url, 'http://localhost').searchParams;
  assert.equal(params.get('q'), 'Common name');
  assert.equal(params.get('cursor'), 'more-artists');
  await search.selectArtist(first.browse_id);
  assert.ok(requests[3].signal.aborted);
  requests[3].finish([{browse_id:'stale-artist'}]);
  await cancelled;
  search.showArtists();
  assert.equal(snapshots.at(-1).artists.loadingMore, false);
  const choices = search.loadMore();
  assert.equal(requests[4].url, requests[3].url);
  requests[4].finish([{browse_id:'artist-three',title:'Another result'}]);
  await choices;
  assert.equal(search.snapshot().mode, 'artists');
  assert.equal(snapshots.at(-1).artists.results.length, 3);
  await search.selectArtist(first.browse_id);
  assert.equal(requests.length, 5);
  assert.deepEqual(snapshots.at(-1).artist.results, [{url:'one'},{url:'two'}]);
  const restored = setup();
  assert.ok(restored.search.restore(JSON.parse(JSON.stringify(search.snapshot()))));
  assert.equal(restored.requests.length, 0);
  restored.search.showArtists();
  assert.equal(restored.snapshots.at(-1).artists.results.length, 3);
});

test('switching artists cancels the old catalogue, and retries keep the artist search context', async () => {
  const {search, requests, snapshots} = setup();
  const pending = search.search('Singer', 'artists');
  requests[0].finish([{title:'First',browse_id:'one'},{title:'Second',browse_id:'two'}]);
  await flush();
  search.showArtists();
  assert.ok(requests[1].signal.aborted);
  const next = search.selectArtist('two');
  requests[2].finish([], false);
  await next;
  assert.equal(snapshots.at(-1).artist.error, 'Provider failed');
  const count = snapshots.length;
  requests[1].finish([{url:'stale'}]);
  await pending;
  assert.equal(snapshots.length, count);
  const retry = search.selectArtist('two', true);
  assert.equal(requests[3].url, '/v1/music/artists/two');
  requests[3].finish([{url:'current'}]);
  await retry;
  assert.equal(snapshots.at(-1).artist.metadata.title, 'Second');
  assert.equal(snapshots.at(-1).artists.results.length, 2);
  assert.deepEqual(snapshots.at(-1).artist.results, [{url:'current'}]);
});

test('reloading an automatically selected artist resumes only its pending catalogue', async () => {
  const first = setup();
  const pending = first.search.search('Singer', 'artists');
  first.requests[0].finish([{title:'Singer',browse_id:'one'},{title:'Other',browse_id:'two'}]);
  await flush();
  const next = setup();
  assert.ok(next.search.restore(JSON.parse(JSON.stringify(first.search.snapshot()))));
  assert.equal(next.requests[0].url, '/v1/music/artists/one');
  assert.equal(next.snapshots.at(-1).artist.metadata.title, 'Singer');
  next.requests[0].finish([{url:'fresh'}]);
  await flush();
  assert.deepEqual(next.snapshots.at(-1).artist.results, [{url:'fresh'}]);
  next.search.showArtists();
  assert.equal(next.search.snapshot().query, 'Singer');
  assert.equal(next.snapshots.at(-1).artists.results.length, 2);
  first.requests[1].finish([]);
  await pending;
});

test('empty or failed artist discovery does not request a catalogue or silently switch search type', async () => {
  const {search, requests, snapshots} = setup();
  let pending = search.search('Nobody', 'artists');
  requests[0].finish([]);
  await pending;
  assert.equal(search.snapshot().mode, 'artists');
  assert.deepEqual(snapshots.at(-1).artists.results, []);
  pending = search.search('Failure', 'artists');
  requests[1].finish([], false);
  await pending;
  assert.equal(requests.length, 2);
  assert.equal(snapshots.at(-1).artists.error, 'Provider failed');
});

test('artist pages keep their metadata and expired cursors leave existing songs available', async () => {
  const requests = [], snapshots = [];
  const search = createVideoSearch(value => snapshots.push(structuredClone(value)), (url) => new Promise(resolve => requests.push({url,resolve})));
  const initial = search.search('UC'+'a'.repeat(22), 'artist');
  requests[0].resolve({ok:true,json:async () => ({title:'Singer',results:[{url:'song'}],next_cursor:'next'})});
  await initial;
  const more = search.loadMore();
  assert.ok(requests[1].url.endsWith('?cursor=next'));
  requests[1].resolve({ok:false,status:410,json:async () => ({detail:'搜索已过期，请重新搜索'})});
  await more;
  assert.equal(snapshots.at(-1).artist.expired, true);
  assert.equal(snapshots.at(-1).artist.metadata.title, 'Singer');
  assert.equal(snapshots.at(-1).artist.results[0].url, 'song');
});

test('theme follows system changes until chosen, and resumes after the saved choice is cleared', () => {
  const script = readFileSync(new URL('./web/theme.js', import.meta.url), 'utf8');
  for (const startsDark of [false,true]) {
    const events = {}, storage = new Map(), root = {dataset:{}}, meta = {};
    const system = {matches:startsDark, addEventListener:(_, fn) => { events.system = fn; }};
    const button = {addEventListener:(_, fn) => { events.click = fn; }, setAttribute:() => {}};
    const document = {documentElement:root, querySelector:() => meta, getElementById:() => button,
      addEventListener:(_, fn) => { events.ready = fn; }};
    const window = {matchMedia:() => system, addEventListener:(_, fn) => { events.storage = fn; }};
    const localStorage = {getItem:key => storage.get(key), setItem:(key,value) => storage.set(key,value)};
    runInNewContext(script, {document, window, localStorage});
    assert.equal(root.dataset.theme, startsDark ? 'dark' : 'light');
    assert.equal(storage.size, 0);
    events.ready();
    system.matches = !startsDark; events.system();
    assert.equal(root.dataset.theme, startsDark ? 'light' : 'dark');
    events.click();
    const choice = root.dataset.theme;
    assert.equal(storage.get('karaokeTheme'), choice);
    system.matches = !system.matches; events.system();
    system.matches = !system.matches; events.system();
    assert.equal(root.dataset.theme, choice);
    runInNewContext(script, {document, window, localStorage});
    assert.equal(root.dataset.theme, choice);
    storage.clear(); events.storage({key:'karaokeTheme'});
    assert.equal(root.dataset.theme, system.matches ? 'dark' : 'light');
    assert.equal(meta.content, system.matches ? '#202022' : '#ffffff');
  }
});

test('visible lyric checks are batched, cached and do not fetch the whole catalogue', async () => {
  const requests = [], states = [];
  const search = createVideoSearch(s => states.push(structuredClone(s)), (url, {signal}) =>
    new Promise(resolve => requests.push({url, signal, finish:data => resolve({ok:true,json:async()=>data})})));
  const songs = Array.from({length:100}, (_,i) => ({url:`https://www.youtube.com/watch?v=${String(i).padStart(11,'0')}`,kind:'song',title:`Song ${i}`}));
  const initial = search.search('songs');
  requests[0].finish({results:songs}); await initial;
  assert.equal(requests.length,1);
  for (const song of songs.slice(0,4)) search.checkLyrics(song.url);
  search.checkLyrics(songs[0].url);
  search.checkLyrics(songs[3].url,false);
  await flush();
  assert.equal(requests.length,2);
  const ids = new URL(requests[1].url,'http://test').searchParams.get('ids').split(',');
  assert.deepEqual(ids,['00000000000','00000000001','00000000002']);
  // A server clock far behind the phone must not cause a refetch loop.
  const packet = {status:'available',expires_at:1,cache_seconds:3600};
  requests[1].finish({results:{[ids[0]]:packet,[ids[1]]:{...packet,status:'unknown'}}}); await flush();
  const checked = states.at(-1).youtube.results;
  assert.equal(checked[0].lyrics.status,'available');
  assert.equal(checked[1].lyrics.status,'unknown');
  assert.equal(checked[2].lyrics.reason,'lookup_failed');
  assert.equal(checked[4].lyrics,undefined);
  for (const song of songs.slice(0,3)) search.checkLyrics(song.url);
  await flush(); assert.equal(requests.length,2);
  const saved=search.snapshot();
  search.restore(saved); search.checkLyrics(songs[0].url); await flush();
  assert.equal(requests.length,2);
});

test('a cancelled lyric batch cannot repaint a new query, including the same recording', async () => {
  const requests=[], states=[];
  const search=createVideoSearch(s=>states.push(structuredClone(s)), (url,{signal})=>
    new Promise(resolve=>requests.push({url,signal,finish:data=>resolve({ok:true,json:async()=>data})})));
  const song={url:'https://www.youtube.com/watch?v=abcdefghijk',kind:'song',title:'Song'};
  let initial=search.search('old'); requests[0].finish({results:[song]}); await initial;
  search.checkLyrics(song.url); await flush();
  initial=search.search('new'); requests[2].finish({results:[song]}); await initial;
  search.checkLyrics(song.url); await flush();
  assert.equal(requests[1].signal.aborted,true);
  requests[1].finish({results:{abcdefghijk:{status:'available',expires_at:Date.now()/1000+3600}}}); await flush();
  assert.equal(states.at(-1).youtube.results[0].lyrics,undefined);
  assert.equal(requests.length,4);
  requests[3].finish({results:{abcdefghijk:{status:'unknown',expires_at:1,cache_seconds:60}}}); await flush();
  assert.equal(states.at(-1).youtube.results[0].lyrics.status,'unknown');
});

test('lyric priority keeps unconfirmed versions and separates availability from readiness', () => {
  const expiry=Date.now()/1000+300;
  const missing={title:'Missing',status:'completed',lyrics_unavailable:true,lyrics_need_review:true,
    lyrics:{status:'available',expires_at:expiry}};
  const review={title:'Review',status:'completed',lyrics_need_review:true};
  const songs=[{title:'Unchecked'}, {title:'Expired',lyrics:{status:'available',expires_at:0}},
    missing,review,
    {title:'Found',lyrics:{status:'available',expires_at:expiry}}, {title:'Failed',lyrics:{status:'unknown',expires_at:expiry}},
    {title:'Ready',status:'completed'}];
  assert.equal(lyricLabel(missing).text,'歌词暂缺');
  assert.equal(lyricLabel(review).text,'歌词待校对');
  assert.deepEqual(librarySongs(songs).map(s=>s.title),['Review','Ready']);
  assert.deepEqual(orderResults(songs,'lyrics').map(s=>s.title),['Found','Ready','Review','Unchecked','Expired','Missing','Failed']);
  assert.deepEqual(orderResults(songs,'ready').map(s=>s.title),['Review','Ready','Unchecked','Expired','Missing','Found','Failed']);
  assert.equal(songs[0].title,'Unchecked');
});
