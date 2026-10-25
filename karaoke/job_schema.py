"""Public job inputs, safe provider errors and single-video platform URLs."""
import fcntl
import hashlib
import json
import math
import os
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


def playback_stems(artifacts):
    """Published playback layouts; old three-track songs remain playable."""
    return (('accompaniment', 'lead_vocals') if 'accompaniment' in artifacts else
            ('instrumental', 'backing_vocals', 'lead_vocals'))


def playback_version(result):
    artifacts = result.get('artifacts', {})
    stems = [(name, artifacts.get(name, {}).get('sha256')) for name in playback_stems(artifacts)]
    if result.get('shared_preview_gain') is None or not all(digest for _, digest in stems):
        return None
    return hashlib.sha256(json.dumps([stems, result['shared_preview_gain']]).encode()).hexdigest()


def preparation_rank(job_ids):
    return {job_id: rank for rank, job_id in enumerate(dict.fromkeys(job_ids))}


def download_backoff(url, directory=None, *, until=0):
    """Share explicit provider limits with already-started pipeline children."""
    directory = directory or os.getenv('KARAOKE_STAGE_LOCK_DIR')
    if not directory:
        return 0
    folder = Path(directory)
    try:
        video_url(url)
    except ValueError:
        return 0  # Retired sources in stored jobs must not hold the active queue.
    path = folder/'youtube.retry.json'

    def read():
        try:
            deadline = json.loads(path.read_text())
        except FileNotFoundError:
            return 0
        if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline < 0:
            raise ValueError('Invalid youtube download backoff')
        return deadline

    if not until:
        return read()
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'youtube.retry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        deadline = max(read(), until)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(deadline))
        temporary.replace(path)
    return deadline


def retry_fields(error, diagnostic, number, *, failed_at):
    """Scope retries to the failure and respect explicit server delays."""
    exponent = 2**min(10, max(0, number-1))
    fields = dict(status='failed', provider_blocked=False, next_retry_at=None)
    if error.get('provider') == 'openai':
        scope = 'openai'+(':'+error['model'] if error.get('model') else '')
        fields.update(status='external' if error['retryable'] else 'blocked', retry_provider=scope,
                      provider_blocked=error['code'] == 'openai_rate_limit')
        if error['retryable']:
            fields['next_retry_at'] = failed_at+max(error.get('retry_after_seconds', 0), min(900, 30*exponent))
        return fields
    if error['code'] == 'interrupted' or re.search(r'^ImportError: dlopen\(.*system is shutting down', diagnostic, re.M):
        return dict(fields, status='external', retry_provider='local', next_retry_at=failed_at+min(900, 60*exponent))
    if error['code'] not in ('download_failed', 'download_video_failed'):
        return fields
    # A suggested remedy ("check login settings") is not a login challenge.
    text = (diagnostic or error.get('message', '')).lower()
    blocked = any(s in text for s in ('sign in', 'login required', 'login_required', '登录验证',
        'risk control', '-352', 'captcha', 'video unavailable', 'video is unavailable', 'private video', '无法访问',
        'unsupported url', 'requested format is not available', 'not a live video',
        'no such file or directory', 'unable to connect to proxy', 'proxyerror', 'tunnel connection failed'))
    http_status = r'(?:http error|error code:|http/\d(?:\.\d)?|[（(])\s*'
    blocked |= text.strip() == 'unavailable'
    rate_limited = bool(re.search(http_status+r'429\b', text))
    blocked |= bool(re.search(http_status+r'(?:400|401|404|407|410|451)\b', text))
    if blocked and not rate_limited:
        return dict(fields, status='blocked')
    forbidden = bool(re.search(http_status+r'403\b', text))
    delay = max(error.get('retry_after_seconds') or 0, retry_after_seconds(diagnostic) or 0,
                min(1800, (300 if forbidden and not rate_limited else 60)*exponent))
    return dict(fields, status='blocked' if blocked else 'external',
                provider_blocked=rate_limited, next_retry_at=failed_at+delay)


def retry_after_seconds(value):
    """Retain a server's retry hint without exposing response headers or body."""
    hints = []
    headers = getattr(getattr(value, 'response', None), 'headers', {})
    for name, scale in (('retry-after-ms', .001), ('retry-after', 1.)):
        if name not in headers:
            continue
        try:
            hints.append(float(headers[name])*scale)
        except ValueError:
            try:
                hints.append((parsedate_to_datetime(headers[name])-datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError):
                pass
    for amount, unit in re.findall(r'try again in\s+(\d+(?:\.\d+)?)\s*(ms|seconds?|s|minutes?|m)\b', str(value), re.I):
        hints.append(float(amount)*(.001 if unit.lower() == 'ms' else 60 if unit.lower().startswith('m') else 1))
    for name, raw in re.findall(r'^\s*(retry-after(?:-ms)?):\s*([^\r\n]+)', str(value), re.I | re.M):
        try:
            hints.append(float(raw)*(.001 if name.lower().endswith('-ms') else 1))
        except ValueError:
            try:
                hints.append((parsedate_to_datetime(raw)-datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError):
                pass
    return max((v for v in hints if math.isfinite(v) and v >= 0), default=None)


def provider_error(value):
    """Classify SDK exceptions (or their log lines) without exposing response data."""
    if isinstance(value, BaseException):
        names = [type(value).__name__] if type(value).__module__.startswith('openai') else []
    else:
        names = re.findall(r'^openai\.(\w+Error):', str(value), re.M)
    known = {
        'APITimeoutError': ('timeout', '语音或歌词服务响应超时，请稍后重试', True),
        'APIConnectionError': ('connection', '无法连接语音或歌词服务，请稍后重试', True),
        'InternalServerError': ('server_error', '语音或歌词服务暂时异常，请稍后重试', True),
        'RateLimitError': ('rate_limit', '语音或歌词服务额度或频率受限，请稍后重试', True),
        'ContentFilterFinishReasonError': ('content_rejected', '歌词服务拒绝处理本次内容，需要人工检查', False),
        'AuthenticationError': ('authentication', '语音或歌词服务认证失败，请检查服务配置', False),
        'PermissionDeniedError': ('permission', '语音或歌词服务拒绝访问，请检查服务权限', False),
    }
    for name in reversed(names):
        if name in known:
            code, message, retryable = known[name]
            text = str(value)
            if name == 'RateLimitError' and 'request too large' in text.lower():
                code, message, retryable = 'request_too_large', '请求超过模型或项目限制，需要调整请求或配置', False
            elif name == 'RateLimitError' and 'insufficient_quota' in text.lower():
                code, message, retryable = 'quota', '语音或歌词服务额度不足，需要检查项目额度', False
            result = dict(code='openai_'+code, message=message, provider='openai', retryable=retryable)
            delay = retry_after_seconds(value)
            if delay is not None:
                result['retry_after_seconds'] = delay
            model = re.search(r'for ([\w.-]+) in (?:project|organization)', text)
            if model:
                result['model'] = model.group(1)
            return result
    return None


@lru_cache(maxsize=256)
def _language_code(value):
    """Resolve language names and legacy/region tags with the shared registry."""
    from langcodes import Language, LanguageTagError, find, standardize_tag
    try:
        tag = Language.get(value) if re.fullmatch(r'[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]+)*', value) else find(value)
        code = standardize_tag(tag.language or 'und', macro=True)
        return code if code != 'und' and Language.get(code).is_valid() else None
    except (LookupError, LanguageTagError):
        return None


def language_code(value):
    return _language_code(value.strip()) if isinstance(value, str) and value.strip() else None


def song_language(value):
    from langcodes import Language
    code = language_code(value)
    return Language.get(code).display_name('en') if code else None


def thumbnail_url(value):
    """Only display artwork served by the supported source platforms."""
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        if (parsed.scheme == 'https' and not parsed.username and not parsed.password
                and parsed.port in (None, 443) and any(
                    parsed.hostname == host or (parsed.hostname or '').endswith('.'+host)
                    for host in ('ytimg.com', 'googleusercontent.com'))):
            return value
    except ValueError:
        pass
    return None


def video_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in ('http', 'https') or parsed.username or parsed.password:
        raise ValueError('请输入 YouTube 的 HTTP(S) 视频链接')
    if parsed.port not in (None, 80, 443):
        raise ValueError('视频链接不支持自定义端口')
    host = (parsed.hostname or '').lower()
    parts = parsed.path.strip('/').split('/')
    query = parse_qs(parsed.query)
    if host in ('youtube.com', 'www.youtube.com', 'm.youtube.com', 'music.youtube.com', 'youtu.be'):
        identifier = (parts[0] if host == 'youtu.be' else
                      query.get('v', [''])[0] if parsed.path == '/watch' else
                      parts[1] if len(parts) == 2 and parts[0] in ('shorts', 'live', 'embed') else '')
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}', identifier):
            raise ValueError('请提供单个 YouTube 视频链接，而非频道或播放列表')
        return 'https://www.youtube.com/watch?v=' + identifier
    raise ValueError('当前只接受 YouTube 单个视频链接')


class JobRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    url: str = Field(max_length=2048)
    lyrics: list[str] | None = Field(default=None, max_length=1000)
    title: str | None = Field(default=None, max_length=200)
    artist: str | None = Field(default=None, max_length=200)
    language: str | None = Field(default=None, max_length=80)
    mv_audio: Literal['instrumental_backing', 'instrumental'] = 'instrumental_backing'
    force: bool = False

    @field_validator('url')
    @classmethod
    def canonical_url(cls, value):
        return video_url(value)

    @field_validator('lyrics', mode='before')
    @classmethod
    def lyric_lines(cls, value):
        if isinstance(value, str):
            value = value.splitlines()
        if value is None:
            return None
        if not isinstance(value, list) or any(not isinstance(line, str) for line in value):
            raise ValueError('lyrics 应为分行文本或字符串数组')
        lines = [line.strip() for line in value if line.strip()]
        if not lines or any(len(line) > 200 for line in lines) or sum(map(len, lines)) > 50000:
            raise ValueError('歌词不能为空，每行最多 200 字符，总计最多 50000 字符')
        return lines

    @field_validator('language', mode='before')
    @classmethod
    def language_alias(cls, value):
        if value is None:
            return None
        language = song_language(value)
        if language is None:
            raise ValueError('请使用有效的语言名称或 ISO 语言代码')
        return language
