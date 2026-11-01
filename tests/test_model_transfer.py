"""Security boundaries for model URL transfers."""

import asyncio
import logging
import socket

import pytest

from speechdock import model_transfer as transfer


class Writer:
    def __init__(self):
        self.request = b""
        self.closed = False

    def write(self, data):
        self.request += data

    async def drain(self):
        pass

    def close(self):
        self.closed = True

    async def wait_closed(self):
        pass


def records(*addresses):
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 443))
            for address in addresses]


def setup_network(monkeypatch, answers, responses):
    loop = asyncio.get_running_loop()
    calls = []
    writers = []

    async def resolve(host, port, **kwargs):
        assert port == 443
        return records(*answers[host])

    async def connect(host, port, **kwargs):
        calls.append((host, port, kwargs))
        reader = asyncio.StreamReader()
        reader.feed_data(responses[len(calls) - 1])
        reader.feed_eof()
        writer = Writer()
        writers.append(writer)
        return reader, writer

    monkeypatch.setattr(loop, "getaddrinfo", resolve)
    monkeypatch.setattr(asyncio, "open_connection", connect)
    return calls, writers


async def test_connects_to_validated_ip_with_original_tls_name(monkeypatch):
    calls, writers = setup_network(monkeypatch, {"models.example": ["8.8.8.8"]}, [
        b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nmodel"
    ])
    async with transfer.stream_model("https://models.example/file?token=secret") as response:
        assert response.total_bytes == 5
        assert b"".join([chunk async for chunk in response.aiter_bytes()]) == b"model"
    assert calls == [("8.8.8.8", 443, {"ssl": True, "server_hostname": "models.example"})]
    assert b"Host: models.example\r\n" in writers[0].request or b"host: models.example\r\n" in writers[0].request
    assert writers[0].closed


async def test_mixed_public_private_dns_is_rejected_before_connect(monkeypatch):
    calls, _ = setup_network(monkeypatch, {"models.example": ["8.8.8.8", "127.0.0.1"]}, [])
    with pytest.raises(transfer.ModelTransferError):
        async with transfer.stream_model("https://models.example/model"):
            pytest.fail("private DNS answer accepted")
    assert not calls


async def test_redirect_to_private_dns_is_rejected(monkeypatch):
    calls, writers = setup_network(monkeypatch, {
        "models.example": ["8.8.8.8"], "internal.example": ["192.168.1.1"]
    }, [b"HTTP/1.1 302 Found\r\nLocation: https://internal.example/model\r\nContent-Length: 0\r\n\r\n"])
    with pytest.raises(transfer.ModelTransferError):
        async with transfer.stream_model("https://models.example/model"):
            pytest.fail("private redirect accepted")
    assert len(calls) == 1
    assert writers[0].closed


@pytest.mark.parametrize("url", [
    "http://models.example/model",
    "https://models.example:444/model",
    "https://user:password@models.example/model",
    "https://models.example/model.zip?download=1",
    "https://models.example/model%2Etar%2Egz",
])
async def test_invalid_url_is_rejected_without_connection(monkeypatch, url):
    calls, _ = setup_network(monkeypatch, {}, [])
    with pytest.raises(transfer.ModelTransferError):
        async with transfer.stream_model(url):
            pytest.fail("invalid URL accepted")
    assert not calls


async def test_error_and_debug_logs_do_not_disclose_url(monkeypatch, caplog):
    secret_url = "https://models.example/private-path?token=secret-value"
    setup_network(monkeypatch, {"models.example": ["8.8.8.8"]}, [
        b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n"
    ])
    with caplog.at_level(logging.DEBUG), pytest.raises(transfer.ModelTransferError) as exc:
        async with transfer.stream_model(secret_url):
            pytest.fail("error response accepted")
    assert secret_url not in str(exc.value)
    assert "private-path" not in str(exc.value)
    assert "secret-value" not in str(exc.value)
    assert "private-path" not in caplog.text
    assert "secret-value" not in caplog.text


async def test_stream_enforces_size_limit_without_content_length(monkeypatch):
    monkeypatch.setattr(transfer, "MAX_MODEL_BYTES", 4)
    setup_network(monkeypatch, {"models.example": ["8.8.8.8"]}, [
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nmodel\r\n0\r\n\r\n"
    ])
    with pytest.raises(transfer.ModelTransferError, match="size limit"):
        async with transfer.stream_model("https://models.example/model") as response:
            assert response.total_bytes is None
            async for _ in response.aiter_bytes():
                pass
