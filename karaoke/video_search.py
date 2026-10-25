"""YouTube video search and YouTube Music catalogues in provider order."""
import asyncio
import json
import math
import re
import secrets
import sys
import time
import unicodedata
from collections import OrderedDict
from html import unescape
from typing import Any
from urllib.parse import urlsplit

from opencc import OpenCC

from .job_schema import thumbnail_url, video_url

_RELEASE = re.compile(r'\bofficial\b|官方|オフィシャル', re.I)
_UNOFFICIAL = re.compile(r'\b(?:fan[ -]?made|unofficial)\b|饭制|飯製|搬运|搬運|非官方', re.I)
_PUBLISHER = re.compile(r'music|records?|recordings?|entertainment|vevo|音[乐樂]|唱片|娱[乐樂]|娛樂|演[艺藝]|[卫衛][视視]|[广廣]播|[电電][视視]', re.I)
_FAN_CHANNEL = re.compile(r'\bfan\b|lyrics?|covers?|搬运|搬運|歌[词詞]|粉[丝絲]|后援|後援', re.I)
_simplified = OpenCC('t2s')


def matches_query(query, item):
    def normalize(value):
        return _simplified.convert(unicodedata.normalize('NFKC', value)).casefold()
    title = unescape(re.sub(r'<[^>]*>', '', str(item.get('title') or '')))
    text = normalize(f"{title} {item.get('channel') or item.get('author') or item.get('uploader') or ''}")
    text = ''.join(char for char in text if char.isalnum())
    terms = re.findall(r'[\u3400-\u9fff]+|[^\W_\u3400-\u9fff]+', normalize(query))
    for term in terms:
        if term in ('official', 'mv', '官方') or term in text:
            continue
        # Also accept an unspaced Chinese artist + title in either display order.
        if re.fullmatch(r'[\u3400-\u9fff]{4,}', term) and any(
                term[:i] in text and term[i:] in text for i in range(2, len(term)-1)):
            continue
        return False
    return True


def official_source(provider, item):
    """Conservative provenance annotation; a release title alone is never evidence."""
    title = unescape(re.sub(r'<[^>]*>', '', str(item.get('title') or '')))
    if _UNOFFICIAL.search(title):
        return None
    if provider == 'youtube':
        if item.get('channel_is_official_artist') is True:
            return 'youtube_official_artist_channel'
        channel = item.get('channel') or item.get('uploader') or ''
        if (item.get('channel_is_verified') is True and _RELEASE.search(title)
                and _PUBLISHER.search(channel) and not _FAN_CHANNEL.search(channel)):
            return 'youtube_verified_music_channel_and_release_title'
    return None


def youtube_entries(query, continuation=None):
    from yt_dlp import YoutubeDL
    from yt_dlp.extractor.youtube import YoutubeSearchIE

    class ArtistSearch(YoutubeSearchIE):
        next_page = None
        visitor_data = None

        def _extract_response(self, *args, **kwargs):
            if continuation:
                kwargs['query'].update(continuation['query'])
                kwargs['headers'] = self.generate_api_headers(ytcfg=kwargs['ytcfg'],
                    visitor_data=continuation.get('visitor_data'), default_client=kwargs['default_client'])
            response = super()._extract_response(*args, **kwargs)
            self.visitor_data = self._extract_visitor_data(response)
            return response

        def _extract_entries(self, parent_renderer, continuation_list):
            yield from super()._extract_entries(parent_renderer, continuation_list)
            self.next_page = continuation_list[0]
            continuation_list[:] = [None]  # One native page per request.

        def _extract_video(self, renderer):
            result = super()._extract_video(renderer)
            # yt-dlp folds artist badges into the generic verified flag; retain the distinction.
            result['channel_is_official_artist'] = any(
                badge.get('metadataBadgeRenderer', {}).get('style') == 'BADGE_STYLE_TYPE_VERIFIED_ARTIST'
                for badge in (renderer.get('ownerBadges') or []))
            return result

    with YoutubeDL({'quiet': True, 'no_warnings': True, 'socket_timeout': 12, 'extractor_retries': 0}) as ydl:
        extractor = ArtistSearch(ydl)
        entries = list(extractor._search_results(query))
        return dict(results=search_results('youtube', entries), continuation=(
            dict(query=extractor.next_page, visitor_data=extractor.visitor_data) if extractor.next_page else None))


def search_results(provider, entries, query=''):
    results = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        try:
            url = video_url(item.get('url') or item['webpage_url'])
            if urlsplit(url).hostname != 'www.youtube.com':
                continue
            title = str(item.get('title') or '').strip()
            if not title:
                continue
            duration = float(item.get('duration') or 0)
            if not math.isfinite(duration) or duration < 0:
                duration = 0
            thumbnail = item.get('thumbnail') or next((t.get('url') for t in
                reversed(item.get('thumbnails') or []) if t.get('url')), '')
            results.append(dict(url=url, title=title, provider=provider, duration=duration,
                                uploader=str(item.get('uploader') or item.get('channel') or ''),
                                thumbnail=thumbnail_url(thumbnail), views=item.get('view_count'),
                                source_evidence=official_source(provider, item), live_status=item.get('live_status')))
        except (KeyError, TypeError, ValueError):
            continue
    return results


def music_thumbnail(item):
    for thumb in reversed(item.get('thumbnails') or []):
        url = thumb.get('url', '')
        if safe := thumbnail_url(url):
            return safe
    return None


def music_results(entries, kind):
    results = []
    for rank, item in enumerate(entries, 1):
        if not isinstance(item, dict):
            continue
        result_type = item.get('resultType') or ('artist' if kind == 'artists' else 'song')
        primary_artist = (item.get('artists') or [{}])[0]
        title = (item.get('artist') or item.get('title') or primary_artist.get('name')) if result_type == 'artist' else item.get('title')
        if not isinstance(title, str) or not title.strip():
            continue
        common = dict(title=title, provider='youtube_music', thumbnail=music_thumbnail(item), rank=rank)
        if result_type == 'artist':
            browse_id = item.get('browseId') or primary_artist.get('id') or ''
            if re.fullmatch(r'UC[\w-]{22}', browse_id):
                results.append(dict(common, kind='artist', version='歌手', browse_id=browse_id,
                    url='https://music.youtube.com/channel/'+browse_id))
            continue
        artists: list[dict[str, Any]] = [dict(name=a['name'], id=a.get('id') if re.fullmatch(r'UC[\w-]{22}', a.get('id') or '') else None)
                   for a in (item.get('artists') or []) if isinstance(a, dict) and isinstance(a.get('name'), str)]
        common.update(artists=artists, uploader=' · '.join(a['name'] for a in artists) or item.get('artist') or '')
        if result_type not in ('song', 'video', 'episode'):
            browse_id = item.get('browseId') or ''
            playlist_id = item.get('playlistId') or ''
            if re.fullmatch(r'[\w-]+', browse_id):
                url = 'https://music.youtube.com/browse/'+browse_id
            elif re.fullmatch(r'[\w-]+', playlist_id):
                url = 'https://music.youtube.com/playlist?list='+playlist_id
            else:
                continue
            results.append(dict(common, kind=result_type, browse_id=browse_id or playlist_id, url=url,
                version={'album': '专辑', 'playlist': '歌单', 'podcast': '播客', 'profile': '用户'}.get(result_type, result_type)))
            continue
        video_id = item.get('videoId') or ''
        if not re.fullmatch(r'[\w-]{11}', video_id):
            continue
        duration = item.get('duration_seconds')
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            duration = 0
        results.append(dict(common, kind='episode' if result_type == 'episode' else 'song', url='https://www.youtube.com/watch?v='+video_id,
            duration=duration,
            album=(item.get('album') or {}).get('name'), views=item.get('views'),
            version={'MUSIC_VIDEO_TYPE_ATV': '专辑音频', 'MUSIC_VIDEO_TYPE_OMV': 'MV'}.get(item.get('videoType'),
                {'video': '视频', 'episode': '播客单集'}.get(result_type, '歌曲')),
            available=item.get('isAvailable') is not False))
    return results


def _renderer(value, name):
    """Find the provider shelf while leaving parsing to the pinned provider library."""
    if isinstance(value, dict):
        if name in value:
            return value[name]
        value = value.values()
    elif not isinstance(value, list):
        return None
    for child in value:
        found = _renderer(child, name)
        if found is not None:
            return found
    return None


def music_entries(kind, query, continuation=None):
    from ytmusicapi import YTMusic
    from ytmusicapi.continuations import get_continuation_params, get_continuation_token
    from ytmusicapi.parsers.playlists import parse_playlist_items
    from ytmusicapi.parsers.search import get_search_params, parse_search_results, parse_top_result
    from ytmusicapi.parsers.songs import parse_song_run

    def playlist_entries(contents):
        entries = []
        for row in contents:
            parsed = parse_playlist_items([row])
            if not parsed:
                continue
            # The playlist parser omits the play-count column on artist pages.
            for column in row['musicResponsiveListItemRenderer'].get('flexColumns', [])[1:]:
                for run in column.get('musicResponsiveListItemFlexColumnRenderer', {}).get('text', {}).get('runs', []):
                    value = parse_song_run(run)
                    if value['type'] == 'views':
                        parsed[0]['views'] = value['data']
            entries.extend(parsed)
        return entries

    class PagedMusic(YTMusic):
        response = {}

        def _send_request(self, *args, **kwargs):
            self.response = super()._send_request(*args, **kwargs)
            return self.response

    music = PagedMusic(language='zh_CN')
    metadata = {}
    if kind in ('songs', 'artists'):
        body = dict(query=query)
        if kind == 'artists':  # Only the explicitly selected artist view is filtered.
            body['params'] = get_search_params(kind, None, False)
        music._send_request('search', body,
                            continuation['params'] if continuation else '')
        entries, next_page = [], None

        def visit(value):
            nonlocal next_page
            if isinstance(value, list):
                for child in value:
                    visit(child)
            elif isinstance(value, dict):
                if 'musicCardShelfRenderer' in value:
                    card = value['musicCardShelfRenderer']
                    top = parse_top_result(card, music.parser.get_search_result_types())
                    # Localized artist labels can be unknown to the parser, which
                    # defaults them to albums. A channel link identifies the artist.
                    if top.get('resultType') == 'album' and re.fullmatch(r'UC[\w-]{22}', top.get('browseId') or ''):
                        top['resultType'] = 'artist'
                    entries.append(top)
                    visit(card.get('contents', []))
                elif 'musicResponsiveListItemRenderer' in value:
                    entries.extend(parse_search_results([value], 'artist' if kind == 'artists' else None))
                else:
                    if value.get('continuations'):
                        next_page = dict(params=get_continuation_params(value))
                    token = get_continuation_token([value])
                    if token:
                        next_page = dict(params='&continuation='+token)
                    for child in value.values():
                        visit(child)

        # Parse all native sections in order, without checking titles or localized headings.
        visit(music.response)
    elif kind == 'artist' and re.fullmatch(r'UC[\w-]{22}', query):
        known = (continuation or {}).get('metadata_by_video', {})
        if continuation:
            music._send_request('browse', dict(continuation=continuation['token']))
            contents = music.response['onResponseReceivedActions'][0]['appendContinuationItemsAction']['continuationItems']
            entries = playlist_entries(contents)
        else:
            artist = music.get_artist(query)
            section = next((name for name in ('songs', 'videos')
                            if (artist.get(name) or {}).get('browseId') or
                            (artist.get(name) or {}).get('results')), 'releases')
            shelf = artist.get(section) or {}
            known = {row['videoId']: {field: row[field] for field in
                     ('views', 'artists', 'thumbnails', 'duration_seconds', 'album', 'videoType') if row.get(field) is not None}
                     for row in shelf.get('results', []) if row.get('videoId')}
            playlist_id = shelf.get('browseId')
            metadata = dict(title=artist['name'], browse_id=query, thumbnail=music_thumbnail(artist),
                url='https://music.youtube.com/channel/'+query, monthly_listeners=artist.get('monthlyListeners'),
                catalogue_section=section, order='youtube_music_top_songs' if section == 'songs' else
                'youtube_music_'+section, complete=bool(playlist_id))
            entries = (music.get_playlist(playlist_id, limit=0, related=False, suggestions_limit=0)['tracks']
                       if playlist_id else shelf.get('results', []))
            contents = (_renderer(music.response, 'musicPlaylistShelfRenderer') or {}).get('contents', []) if playlist_id else []
            if any('musicResponsiveListItemRenderer' in row for row in contents):
                entries = playlist_entries(contents)
            if section == 'releases':
                entries = [dict(row, resultType='album', artists=[dict(name=artist['name'], id=query)])
                           for name in ('albums', 'singles') for row in (artist.get(name) or {}).get('results', [])]
        token = get_continuation_token(contents) if contents else None
        # Browsing the full playlist must not erase richer artist-shelf metadata.
        # Join by recording identity; similarly titled versions stay independent.
        for row in entries:
            for field, value in known.get(row.get('videoId'), {}).items():
                if row.get(field) in (None, '', []):
                    row[field] = value
        next_page = dict(token=token, metadata_by_video=known) if token else None
    elif kind == 'album' and re.fullmatch(r'MPRE[\w-]+', query):
        album = music.get_album(query)
        metadata = dict(title=album['title'], browse_id=query, thumbnail=music_thumbnail(album),
            url='https://music.youtube.com/browse/'+query,
            artist=' · '.join(a['name'] for a in album.get('artists') or []))
        entries = [dict(row, album=dict(name=album['title']), artists=row.get('artists') or album.get('artists'),
                        thumbnails=row.get('thumbnails') or album.get('thumbnails'))
                   for row in album['tracks']]
        next_page = None
    else:
        raise ValueError('Invalid music catalogue request')
    return dict(metadata, results=music_results(entries, 'songs' if kind == 'artist' else kind), continuation=next_page)


async def fetch_results(provider, query, continuation=None):
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'karaoke.video_search', query, provider,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        output, _ = await process.communicate(json.dumps(continuation).encode())
        if process.returncode:
            raise RuntimeError('YouTube search unavailable')
        return json.loads(output)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


class SearchCursorExpired(ValueError):
    pass


class VideoSearch:
    def __init__(self):
        self.cache, self.pending, self.cursors = OrderedDict(), {}, OrderedDict()
        self.slots = asyncio.Semaphore(2)

    async def search(self, provider, query, cursor=None):
        continuation, offset = None, 0
        if cursor:
            saved = self.cursors.get(cursor)
            if not saved or saved[1:3] != (provider, query) or time.monotonic()-saved[0] >= 1800:
                raise SearchCursorExpired('Search cursor expired or does not match this query')
            _, _, _, continuation, offset = saved
            self.cursors.move_to_end(cursor)
        key = provider, query, cursor
        cached = self.cache.get(key)
        if cached and time.monotonic()-cached[0] < 120:
            # The page may outlive its next-page handle in the independent cursor LRU.
            next_cursor = cached[1].get('next_cursor')
            if next_cursor is None or next_cursor in self.cursors:
                self.cache.move_to_end(key)
                return cached[1]
        if key not in self.pending:
            if len(self.pending) >= 8:
                raise OverflowError('Too many pending searches')
            self.pending[key] = asyncio.create_task(self._load(key, continuation, offset))
            self.pending[key].add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        # A browser disconnect must not cancel another client's identical query.
        return await asyncio.shield(self.pending[key])

    async def _load(self, key, continuation, offset):
        try:
            async with asyncio.timeout(25):
                async with self.slots:
                    result = await fetch_results(*key[:2], continuation)
            next_page = result.pop('continuation', None)
            result['next_cursor'] = None
            for index, row in enumerate(result['results'], offset+1):
                row['rank'] = index
            if next_page and next_page != continuation:
                cursor = secrets.token_urlsafe(24)
                self.cursors[cursor] = (time.monotonic(), *key[:2], next_page, offset+len(result['results']))
                while len(self.cursors) > 256:
                    self.cursors.popitem(last=False)
                result['next_cursor'] = cursor
            self.cache[key] = time.monotonic(), result
            self.cache.move_to_end(key)
            while len(self.cache) > 32:
                self.cache.popitem(last=False)
            return result
        finally:
            self.pending.pop(key, None)

    async def close(self):
        tasks = list(self.pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == '__main__':
    provider = sys.argv[2] if len(sys.argv) > 2 else 'youtube'
    continuation = json.loads(sys.stdin.read() or 'null')
    print(json.dumps(youtube_entries(sys.argv[1], continuation) if provider == 'youtube' else
                     music_entries(provider, sys.argv[1], continuation), ensure_ascii=False))
