"""Opt-in REAL acoustic/local-PMS API proof. No production connection or fabricated audio.
Run with SPDOCK_ALIGNMENT_PROOF_DIR pointing at checksummed model assets and
SPDOCK_ALIGNMENT_MOVIE_PROOF_DIR at the saved independent Parakeet JSON/WAV.
This is an isolated Plex-compatible HTTP fixture, NOT proof of a real PMS deployment.
"""
import hashlib
import json
import os
import resource
import subprocess
import threading
import time
import wave
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from speechdock.alignment import alignment_lifespan, attach_alignment_routes
from speechdock.alignment.core import Rejected
from speechdock.alignment.protocol import Window, manifest_hash
from speechdock.alignment.source import Configuration, decode_audio, open_media, probe


@pytest.mark.skipif(not os.environ.get("SPDOCK_ALIGNMENT_PROOF_DIR"), reason="Explicit real acoustic fixture opt-in required")
def test_real_movie_cpu_and_complete_local_api(tmp_path):
    models = Path(os.environ["SPDOCK_ALIGNMENT_PROOF_DIR"])
    proof = Path(os.environ["SPDOCK_ALIGNMENT_MOVIE_PROOF_DIR"])
    original = proof / "dialogue-75.wav"
    independent = json.loads((proof / "parakeet-dialogue.json").read_text())
    text = "Worry about yourself."
    assert text in independent["text"]  # Not copied from the aligner's greedy transcript.
    srt = b"1\n00:00:14,000 --> 00:00:16,000\nWorry about yourself.\n"
    (tmp_path / "source.srt").write_bytes(srt)
    media = tmp_path / "movie.mkv"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-i", str(original), "-i", str(tmp_path / "source.srt"), "-map", "0:a", "-map", "1:a", "-map", "2:s", "-t", "75", "-c:a", "pcm_s16le", "-c:s", "srt", "-metadata:s:a:0", "language=spa", "-metadata:s:a:1", "language=eng", "-metadata:s:s:0", "language=eng", str(media)], check=True, timeout=30)
    token = tmp_path / "token"
    token.write_text("isolated-fixture-token")
    selected = {"embedded": False, "changed": False}
    class PMS(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("X-Plex-Token") != "isolated-fixture-token":
                self.send_error(401)
                return
            if self.path == "/identity":
                content = b'<MediaContainer machineIdentifier="local-fixture" />'
            elif self.path == "/library/metadata/1":
                key = '' if selected["embedded"] else ' key="/library/streams/5"'
                version = "changed" if selected["changed"] else "version"
                content = f'<MediaContainer><Video ratingKey="1"><Media id="2"><Part id="3" key="/library/parts/3/{version}/movie.mkv" file="/fixture/movie.mkv"><Stream id="4" streamType="2" index="1" codec="pcm_s16le" channels="1" languageCode="eng"/><Stream id="5" streamType="3" index="2" codec="srt" languageCode="eng"{key}/></Part></Media></Video></MediaContainer>'.encode()
            elif self.path.startswith("/library/streams/5"):
                content = srt
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), PMS)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = Configuration(f"http://127.0.0.1:{server.server_port}", token, (("/fixture", str(tmp_path)),), models)
    @asynccontextmanager
    async def lifespan(app):
        async with alignment_lifespan(app, config):
            yield
    app = FastAPI(lifespan=lifespan)
    attach_alignment_routes(app)
    body = dict(protocol=1, request_id="proof", generation=42, playback_id="independent-movie",
                identity=dict(server_machine_id="local-fixture", rating_key="1", media_id="2", part_id="3", part_key="/library/parts/3/version/movie.mkv", audio_stream_id="4", subtitle_stream_id="5", subtitle_codec="srt"),
                subtitle_source_kind="sidecar", subtitle_hash_kind="source_bytes_sha256", subtitle_source_sha256=hashlib.sha256(srt).hexdigest(), language="en", timeline="container_pts", media_origin_pts_ms=0, subtitle_offset_ms=0, window_start_ms=14000, window_end_ms=16000,
                cues=[dict(cue_id="independent-parakeet", ordinal=0, authored_start_ms=14000, authored_end_ms=16000, start_ms=14000, end_ms=16000, text=text)])
    evidence = {"proof_kind": "local Plex-compatible fixture, real saved movie PCM, independent saved Parakeet text", "cases": {}}
    try:
        with TestClient(app) as client:
            started = time.perf_counter()
            answer = client.post("/api/alignment/windows", json=body)
            assert answer.status_code == 200, answer.text
            answer = answer.json()
            evidence["cold_api_seconds"] = time.perf_counter() - started
            evidence["cases"]["sidecar"] = answer
            assert answer["cues"][0]["status"] == "aligned", answer
            assert answer["request_cues"] == body["cues"]
            result_words = answer["cues"][0]["words"]
            assert len(result_words) == 3
            assert all(14000 <= item["start_ms"] < item["end_ms"] <= 16000 for item in result_words)
            assert all(left["end_ms"] <= right["start_ms"] for left, right in zip(result_words, result_words[1:]))
            assert [text.encode()[item["start_byte"]:item["end_byte"]].decode() for item in result_words] == text.split()
            real_align = app.state.alignment.aligner.align
            def changing_source(*args):
                measured = real_align(*args)
                selected["changed"] = True
                return measured
            app.state.alignment.aligner.align = changing_source
            fenced = client.post("/api/alignment/windows", json=body).json()
            assert fenced["cues"][0]["status"] == "estimated" and fenced["cues"][0]["words"] == []
            evidence["cases"]["changed_during_alignment"] = fenced["cues"][0]["reason"]
            app.state.alignment.aligner.align = real_align
            selected["changed"] = False
            selected["embedded"] = True
            embedded = body | {"subtitle_source_kind": "embedded", "subtitle_hash_kind": "window_manifest_sha256"}
            embedded["subtitle_source_sha256"] = manifest_hash(Window.model_validate(embedded).cues)
            started = time.perf_counter()
            result = client.post("/api/alignment/windows", json=embedded).json()
            evidence["warm_embedded_api_seconds"] = time.perf_counter() - started
            evidence["cases"]["embedded"] = result
            assert result["cues"][0]["status"] == "aligned", result
            bad = embedded | {"subtitle_source_sha256": "f" * 64}
            assert client.post("/api/alignment/windows", json=bad).json()["cues"][0]["reason"] == "subtitle_hash_mismatch"
            unsupported = client.post("/api/alignment/windows", json=body | {"language": "es"}).json()
            assert unsupported["cues"][0]["reason"] == "unsupported_language"
            selected["changed"] = True
            assert client.post("/api/alignment/windows", json=body).json()["cues"][0]["reason"] == "media_identity_changed"
            selected["changed"] = False
            with wave.open(str(original)) as audio:
                pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
            aligner = app.state.alignment.aligner
            sample = pcm[14000 * 16:16000 * 16]
            evidence["matched_scores"] = aligner.align(sample, text, 14000, 14000, 16000)
            for label, audio, target, bounds in [
                ("silence", np.zeros(32000, dtype=np.float32), text, (14000, 16000)),
                ("mismatch", sample, "Bring me the newspaper.", (14000, 16000)),
                ("translation", sample, "Preocupate por ti mismo.", (14000, 16000)),
                ("unsupported", sample, "Worry about 123.", (14000, 16000)),
                ("repeated", sample, "Worry worry about yourself.", (14000, 16000)),
                ("clipped", sample, text, (15300, 16000)),
            ]:
                with pytest.raises(Rejected) as error:
                    aligner.align(audio, target, 14000, *bounds)
                evidence["cases"][label] = str(error.value)
            spanish = Path(os.environ["SPDOCK_ALIGNMENT_SPANISH_PROOF"])
            with wave.open(str(spanish)) as audio:
                assert audio.getframerate() == 16000 and audio.getnchannels() == 1
                native = np.frombuffer(audio.readframes(min(audio.getnframes(), 15 * 16000)), dtype="<i2").astype(np.float32) / 32768
            with pytest.raises(Rejected) as error:
                aligner.align(native, "Worry about yourself.", 0, 0, 15000)
            evidence["cases"]["native_spanish_english_text"] = str(error.value)
            # Test nonzero and negative container origins, without assuming format.start_time=0.
            for shift in (5, -5):
                shifted = tmp_path / f"shift{shift}.mkv"
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-copyts", "-itsoffset", str(shift), "-i", str(media), "-map", "0", "-c", "copy", "-avoid_negative_ts", "disabled", str(shifted)], check=True, timeout=30)
                with open_media("/fixture/" + shifted.name, config.roots) as fd:
                    inventory = probe(fd)
                    evidence[f"container_start_shift_{shift}"] = inventory["format"]["start_time"]
                    assert float(inventory["format"]["start_time"]) == shift
                    audio, origin = decode_audio(fd, 1, 14000 + shift * 1000, 16000 + shift * 1000, 0, inventory)
                    aligned = aligner.align(audio, text, origin, 14000 + shift * 1000, 16000 + shift * 1000)
                    evidence["cases"][f"pts_shift_{shift}"] = aligned
                    audio, origin = decode_audio(fd, 1, 14000, 16000, shift * 1000, inventory)
                    shifted_origin = aligner.align(audio, text, origin, 14000, 16000)
                    assert shifted_origin["words"] == evidence["matched_scores"]["words"]
            started = time.perf_counter()
            aligner.emissions(pcm[:15 * 16000])
            evidence["fifteen_second_cpu_inference_seconds"] = time.perf_counter() - started
            evidence["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    output = models / "local-api-proof.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({key: value for key, value in evidence.items() if key != "cases"}, indent=2))
