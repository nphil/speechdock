"""Stream untrusted model URLs without allowing access to private networks."""

import asyncio
import ipaddress
import socket
from contextlib import asynccontextmanager
from urllib.parse import unquote, urljoin, urlsplit

import h11
import httpx


MAX_MODEL_BYTES = 8 * 1024**3
_MAX_REDIRECTS = 5
_TIMEOUT = 60
_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tbz2", ".tar.xz", ".txz", ".7z", ".rar", ".gz", ".bz2", ".xz")


class ModelTransferError(Exception):
    """A download failure whose message does not disclose the source URL."""


def _url_parts(url: str):
    try:
        if not isinstance(url, str) or any(ord(c) <= 32 or ord(c) == 127 for c in url):
            raise ValueError
        parts = urlsplit(url)
        if (parts.scheme.lower() != "https" or not parts.hostname or
                parts.username is not None or parts.password is not None or
                parts.port not in (None, 443) or "\\" in parts.netloc):
            raise ValueError
        parsed = httpx.URL(url)
        if parsed.fragment:
            raise ValueError
        host = parsed.raw_host.decode("ascii")
        if not host or "%" in host:
            raise ValueError
        if unquote(parts.path).lower().endswith(_ARCHIVE_SUFFIXES):
            raise ModelTransferError("Archive downloads are not supported")
        return host, parsed.raw_path
    except (ValueError, UnicodeError, httpx.InvalidURL):
        raise ModelTransferError("Invalid HTTPS model URL") from None


async def _resolve(host: str) -> str:
    try:
        records = await asyncio.get_running_loop().getaddrinfo(
            host, 443, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
        addresses = []
        for family, _, _, _, sockaddr in records:
            if family not in (socket.AF_INET, socket.AF_INET6):
                raise ValueError
            address = ipaddress.ip_address(sockaddr[0])
            if (not address.is_global or address.is_multicast or
                    (isinstance(address, ipaddress.IPv6Address) and
                     address.ipv4_mapped is not None and
                     (not address.ipv4_mapped.is_global or address.ipv4_mapped.is_multicast))):
                raise ValueError
            addresses.append(str(address))
        if not addresses:
            raise ValueError
        return addresses[0]
    except (OSError, ValueError):
        raise ModelTransferError("Model URL does not resolve exclusively to public addresses") from None


class _Transfer:
    def __init__(self, reader, connection, total_bytes):
        self._reader = reader
        self._connection = connection
        self.total_bytes = total_bytes
        self._received = 0
        self._started = False

    async def aiter_bytes(self):
        if self._started:
            raise ModelTransferError("Model response has already been consumed")
        self._started = True
        try:
            while True:
                event = await _next_event(self._reader, self._connection)
                if isinstance(event, h11.Data):
                    self._received += len(event.data)
                    if self._received > MAX_MODEL_BYTES:
                        raise ModelTransferError("Model exceeds the 8 GiB size limit")
                    yield event.data
                elif isinstance(event, h11.EndOfMessage):
                    return
                else:
                    raise ModelTransferError("Invalid model download response")
        except asyncio.TimeoutError:
            raise ModelTransferError("Model download timed out") from None
        except h11.ProtocolError:
            raise ModelTransferError("Model download response was incomplete or invalid") from None
        except OSError:
            raise ModelTransferError("Model download connection failed") from None


async def _next_event(reader, connection):
    while True:
        event = connection.next_event()
        if event is not h11.NEED_DATA:
            return event
        data = await asyncio.wait_for(reader.read(65536), timeout=_TIMEOUT)
        connection.receive_data(data)


@asynccontextmanager
async def stream_model(url: str):
    """Yield a bounded streaming response; the caller owns storage and cleanup.

    ``total_bytes`` is the declared length or None. ``aiter_bytes()`` yields
    raw model bytes and rejects streams exceeding MAX_MODEL_BYTES.
    """
    current = url
    for redirects in range(_MAX_REDIRECTS + 1):
        host, target = _url_parts(current)
        address = await _resolve(host)
        writer = None
        transfer = None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(address, 443, ssl=True, server_hostname=host),
                timeout=15,
            )
            connection = h11.Connection(h11.CLIENT, max_incomplete_event_size=65536)
            host_header = f"[{host}]" if ":" in host else host
            request = h11.Request(method=b"GET", target=target, headers=[
                (b"host", host_header.encode("ascii")),
                (b"accept", b"application/octet-stream"),
                (b"accept-encoding", b"identity"),
                (b"connection", b"close"),
            ])
            writer.write(connection.send(request) + connection.send(h11.EndOfMessage()))
            await asyncio.wait_for(writer.drain(), timeout=_TIMEOUT)
            while True:
                event = await _next_event(reader, connection)
                if isinstance(event, h11.InformationalResponse):
                    continue
                if not isinstance(event, h11.Response):
                    raise ModelTransferError("Invalid model download response")
                break
            headers = httpx.Headers(event.headers)
            if event.status_code in (301, 302, 303, 307, 308):
                location = headers.get("location")
                if (not location or redirects == _MAX_REDIRECTS or
                        any(ord(c) <= 32 or ord(c) == 127 for c in location)):
                    raise ModelTransferError("Model download redirect limit or invalid redirect")
                current = urljoin(current, location)
                continue
            if event.status_code != 200:
                raise ModelTransferError(f"Model download returned HTTP {event.status_code}")
            if headers.get("content-encoding", "identity").lower().strip() != "identity":
                raise ModelTransferError("Encoded model downloads are not supported")
            content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type in {"application/zip", "application/x-tar", "application/gzip", "application/x-gzip", "application/x-7z-compressed", "application/vnd.rar", "application/x-rar-compressed"}:
                raise ModelTransferError("Archive downloads are not supported")
            length = headers.get("content-length")
            if length is not None:
                if not length.isascii() or not length.isdecimal():
                    raise ModelTransferError("Invalid model download length")
                total = int(length)
                if total > MAX_MODEL_BYTES:
                    raise ModelTransferError("Model exceeds the 8 GiB size limit")
            else:
                total = None
            transfer = _Transfer(reader, connection, total)
        except (OSError, asyncio.TimeoutError, h11.ProtocolError, ValueError, httpx.InvalidURL):
            raise ModelTransferError("Model download failed") from None
        finally:
            # A successful response remains open while the caller consumes it.
            if writer is not None and transfer is None:
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass
        try:
            yield transfer
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
        return
