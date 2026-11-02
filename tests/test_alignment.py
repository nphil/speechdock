"""Boundary tests for identity, source integrity, bounded admission and CTC topology."""
import asyncio
import hashlib
import threading

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from speechdock.alignment import AlignmentService, attach_alignment_routes, response
from speechdock.alignment.core import AcousticAligner, Rejected, viterbi, words
from speechdock.alignment.protocol import Window, manifest_hash
from speechdock.alignment.source import Configuration, PlexSource, open_media, parse_subtitles, stream_index


def request_body():
    return dict(protocol=1, request_id="request", generation=3, playback_id="playback",
                identity=dict(server_machine_id="fixture", rating_key="1", media_id="2", part_id="3", part_key="/library/parts/3/version/file.mkv", audio_stream_id="4", subtitle_stream_id="5", subtitle_codec="srt"),
                subtitle_source_kind="sidecar", subtitle_hash_kind="source_bytes_sha256", subtitle_source_sha256="0" * 64,
                language="en", timeline="container_pts", media_origin_pts_ms=0, subtitle_offset_ms=0,
                window_start_ms=0, window_end_ms=2000,
                cues=[dict(cue_id="cue", ordinal=0, authored_start_ms=0, authored_end_ms=2000, start_ms=0, end_ms=2000, text="Hello, world!")])


def test_utf8_punctuation_is_reversible_and_unsupported_not_erased():
    text = "“Hello,” world!"
    parsed = words(text)
    assert [text.encode()[start:end].decode() for _, start, end in parsed] == ["“Hello,”", "world!"]
    for text in ("café", "123", "[laughs]", "hello-world", "你好", "♪Hello♪"):
        with pytest.raises(Rejected):
            words(text)


def test_ctc_repeated_letters_require_separating_blank():
    logp = np.full((5, 3), -20, dtype=np.float32)
    logp[np.arange(5), [0, 1, 0, 1, 0]] = 0
    assert viterbi(logp, [1, 1], 0) == [[1], [3]]
    with pytest.raises(Rejected):
        viterbi(logp[:1], [1, 1], 0)


@pytest.mark.parametrize("decoded", ["HI", "|HI", "HI|", "|HI||THERE||", "|HELLO|"])
def test_alignment_preserves_independent_delimiters_and_native_word_frames(decoded):
    aligner = AcousticAligner.__new__(AcousticAligner)
    aligner.vocab = {char: index + 1 for index, char in enumerate(sorted(set(decoded)))}
    aligner.inverse = {index: char for char, index in aligner.vocab.items()}
    aligner.blank = 0
    sequence = [0]
    positions = []
    for char in decoded:
        positions.append(len(sequence))
        sequence.extend([aligner.vocab[char]] * (6 if char == "|" else 1))
        sequence.append(0)
    logp = np.full((len(sequence), len(aligner.vocab) + 1), -20, dtype=np.float32)
    logp[np.arange(len(sequence)), sequence] = 0
    aligner.emissions = lambda _: logp
    text = " ".join(decoded.replace("|", " ").split()).title() + "."
    origin = 1000
    bound = origin + len(sequence) * 20
    result = aligner.align(np.empty(0), text, origin, origin, bound)
    expected = []
    cursor = 0
    for word in text.rstrip(".").upper().split():
        cursor = decoded.index(word, cursor)
        expected.append((origin + positions[cursor] * 20,
                         origin + (positions[cursor + len(word) - 1] + 1) * 20))
        cursor += len(word)
    assert [(word["start_ms"], word["end_ms"]) for word in result["words"]] == expected
    assert [text.encode()[word["start_byte"]:word["end_byte"]].decode() for word in result["words"]] == text.split()
    assert result["minimum_character_log_score"] == 0
    for target, start, end, reason in [
        ("Different words.", origin, bound, "independent_lexical_mismatch"),
        ("Buenos dias.", origin, bound, "independent_lexical_mismatch"),
        ("Hi hi.", origin, bound, "ambiguous_repeated_words"),
        (text, expected[0][0] + 20, bound, "outside_visible_bounds"),
        (text, origin, expected[-1][1] - 20, "outside_visible_bounds"),
    ]:
        with pytest.raises(Rejected, match=reason):
            aligner.align(np.empty(0), target, origin, start, end)


def test_window_limits_and_existing_clipping():
    body = request_body()
    body["cues"][0]["start_ms"] = 100
    assert Window.model_validate(body).cues[0].start_ms == 100
    for change in ({"window_end_ms": 30001}, {"generation": True}, {"subtitle_hash_kind": "window_manifest_sha256"}):
        with pytest.raises(ValidationError):
            Window.model_validate(body | change)
    body["cues"][0]["end_ms"] = 2001
    with pytest.raises(ValidationError):
        Window.model_validate(body)


def test_source_parser_preserves_duplicates_overlaps_and_rejects_markup():
    source = b"1\n00:00:00,000 --> 00:00:02,000\nHello.\n\n2\n00:00:00,000 --> 00:00:02,000\nHello.\n\n3\n00:00:01,000 --> 00:00:03,000\nWorld.\n"
    parsed = parse_subtitles(source)
    assert len(parsed) == 3 and parsed[0] == parsed[1] and parsed[2][0] < parsed[1][1]
    with pytest.raises(Rejected):
        parse_subtitles(source.replace(b"World.", b"<i>World.</i>"))


def test_roots_deny_symlinks_traversal_and_nonregular(tmp_path):
    (tmp_path / "media").write_bytes(b"data")
    (tmp_path / "link").symlink_to(tmp_path / "media")
    roots = (("/plex", str(tmp_path)),)
    with open_media("/plex/media", roots) as fd:
        assert fd >= 0
    for path in ("/plex/link", "/plex/../media", "/other/media", "/plex"):
        with pytest.raises(Rejected):
            with open_media(path, roots):
                pass


def test_stream_selection_is_container_index_not_audio_ordinal():
    streams = {"streams": [{"index": 7, "codec_type": "audio", "codec_name": "aac", "channels": 2, "tags": {"language": "eng"}}]}
    selected = {"index": "7", "codec": "aac", "channels": "2", "languageCode": "eng"}
    assert stream_index(selected, streams, "audio") == 7
    with pytest.raises(Rejected):
        stream_index(selected | {"index": "0"}, streams, "audio")
    with pytest.raises(Rejected):
        stream_index(selected | {"channels": "6"}, streams, "audio")


def test_manifest_hash_utf8_and_source_multiplicity(tmp_path):
    body = request_body()
    body.update(subtitle_source_kind="embedded", subtitle_hash_kind="window_manifest_sha256")
    body["cues"][0]["text"] = "“Hello,” world!"
    request = Window.model_validate(body)
    expected = hashlib.sha256('[[0,0,2000,"“Hello,” world!"]]'.encode()).hexdigest()
    assert manifest_hash(request.cues) == expected
    request.subtitle_source_sha256 = expected
    source = PlexSource(Configuration("http://localhost", tmp_path / "token", (), tmp_path))
    srt = '1\n00:00:00,000 --> 00:00:02,000\n“Hello,” world!\n'.encode()
    source.verify_cues(request, srt)
    assert source.verify_cues(request, srt + b"\n" + srt) == {"cue": "ambiguous_duplicate_cue"}
    request.cues.append(request.cues[0].model_copy(update={"cue_id": "duplicate"}))
    request.subtitle_source_sha256 = manifest_hash(request.cues)
    with pytest.raises(Rejected, match="subtitle_manifest_mismatch"):
        source.verify_cues(request, srt)
    assert source.verify_cues(request, srt + b"\n" + srt) == {"cue": "ambiguous_duplicate_cue", "duplicate": "ambiguous_duplicate_cue"}


def test_api_rejects_oversize_and_never_echoes_secret_validation_input():
    app = FastAPI()
    attach_alignment_routes(app)
    with TestClient(app) as client:
        assert client.post("/api/alignment/windows", content=b"x" * 32769).status_code == 413
        invalid = client.post("/api/alignment/windows", json={"token": "private-token"})
        assert invalid.status_code == 422 and "private-token" not in invalid.text
        body = request_body()
        result = client.post("/api/alignment/windows", json=body).json()
        assert result["request_cues"] == body["cues"]
        assert result["cues"][0]["words"] == []
        assert result["cues"][0]["reason"] == "alignment_not_started"


def test_cancellation_does_not_open_extra_cpu_slots():
    async def run():
        service = AlignmentService()
        entered, release = threading.Event(), threading.Event()
        def work(request):
            entered.set()
            release.wait(3)
            return response(request, "test")
        service.process = work
        request = Window.model_validate(request_body())
        first = asyncio.create_task(service.submit(request))
        await asyncio.to_thread(entered.wait, 3)
        second = asyncio.create_task(service.submit(request))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert (await service.submit(request))["cues"][0]["reason"] == "alignment_busy"
        release.set()
        await second
        await service.close()
    asyncio.run(run())
