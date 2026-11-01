"""Owned model downloads and conservative llama-swap configuration changes."""

import asyncio
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import stat
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from ruamel.yaml import YAML

from .catalog import CATALOG
from .runtimes import model_command, provision_runtimes


class ModelError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


WARNING = ("Applying changes reloads llama-swap and interrupts loaded models. "
           "Using this model can evict the daily and big groups; external GPU use is not controlled.")


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _safe_directory(path):
    path = Path(os.path.abspath(path))
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ModelError(409, f"Symlink is not permitted: {component}")
    if not path.is_dir():
        raise ModelError(503, f"Directory is unavailable: {path}")
    return path


def _regular(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ModelError(409, f"Unsafe or shared file: {path.name}")
    return True


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic(path, data, mode=0o600, owner=None):
    fd, name = tempfile.mkstemp(prefix=".speechdock-", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        if owner and (os.fstat(fd).st_uid, os.fstat(fd).st_gid) != owner:
            os.fchown(fd, *owner)
        with os.fdopen(fd, "wb") as f:
            fd = -1
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        _sync_dir(path.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


class ModelManager:
    def __init__(self, data_dir: Path, settings_provider):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        _safe_directory(self.data_dir)
        self.state_path = self.data_dir / "model-state.json"
        self.models_dir = Path(os.environ.get("SPDOCK_MODELS_DIR", "/models-owned"))
        self.engine_dir = os.environ.get("SPDOCK_ENGINE_MODELS_DIR", "/models/whisper/speechdock")
        self.config_path = Path(os.environ.get("SPDOCK_LLAMA_CONFIG", "/llama-config/config.yaml"))
        self.validator = Path(os.environ.get("SPDOCK_LLAMA_VALIDATOR", "/validator/llama-swap"))
        self.settings_provider = settings_provider
        self._lock = asyncio.Lock()
        self._serial = asyncio.Lock()
        self._tasks = {}
        self._verification_cache = {}
        self._state = {"files": {}, "registrations": {}, "group": None, "downloads": {}, "pending": {}, "imports": {}}
        if _regular(self.state_path):
            try:
                data = json.loads(self.state_path.read_text())
                if not isinstance(data, dict) or any(not isinstance(data.get(k), dict) for k in ("files", "registrations", "downloads")):
                    raise ValueError("invalid state")
                self._state.update(data)
            except (ValueError, OSError) as exc:
                raise ModelError(503, "Model ownership state is unreadable; refusing unsafe changes") from exc
        if not isinstance(self._state.get("imports"), dict):
            raise ModelError(503, "Model ownership state is unreadable; refusing unsafe changes")
        self.runtimes = provision_runtimes(self.models_dir)
        self._recover_downloads()
        for item in self._state["downloads"].values():
            if item.get("status") in ("queued", "downloading", "uploading", "verifying"):
                item.update(status="failed", error="Download interrupted by restart")
        self._save()
    def _recover_downloads(self):
        """Recover publications whose inode belongs to our private staging directory."""
        for ident, pending in list(self._state.get("pending", {}).items()):
            entry = CATALOG.get(ident) or self._state["imports"].get(ident)
            if not entry or not isinstance(pending, dict):
                continue
            name = pending.get("directory", "")
            if not re.fullmatch(r"\.speechdock-download-[a-z0-9_]{8}", name):
                continue
            try:
                root = _safe_directory(self.models_dir)
                stage = root / name
                if stage.exists() or stage.is_symlink():
                    info = stage.lstat()
                    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                            or stat.S_IMODE(info.st_mode) != 0o700
                            or [info.st_dev, info.st_ino] != pending.get("stage_identity")):
                        continue
                    part = stage / "model.part"
                    if part.exists() or part.is_symlink():
                        info = part.lstat()
                        if not stat.S_ISREG(info.st_mode):
                            continue
                        if ident in self._state["imports"]:
                            identity = [info.st_dev, info.st_ino]
                            if identity != pending.get("part_identity") or info.st_nlink not in (1, 2):
                                continue
                            if info.st_nlink == 2:
                                target = root / entry["filename"]
                                target_info = target.lstat()
                                if (not stat.S_ISREG(target_info.st_mode) or
                                        [target_info.st_dev, target_info.st_ino] != identity or
                                        pending.get("file_identity") != identity):
                                    continue
                        part.unlink()
                        _sync_dir(stage)
                    stage.rmdir()
                    _sync_dir(root)
                target = root / entry["filename"]
                if pending.get("file_identity") and _regular(target):
                    info = target.stat(follow_symlinks=False)
                    if [info.st_dev, info.st_ino] == pending["file_identity"]:
                        self._state["files"][ident] = {k: entry[k] for k in ("filename", "sha256", "size_bytes")}
                        if self._verified(entry):
                            self._state["downloads"][ident].update(status="completed" if ident in self._state["imports"] else "complete", progress=1.0, error=None)
                        else:
                            self._state["files"].pop(ident, None)
                del self._state["pending"][ident]
            except (OSError, ModelError, KeyError):
                continue


    def _save(self):
        _safe_directory(self.data_dir)
        if self.state_path.exists() or self.state_path.is_symlink():
            _regular(self.state_path)
        _atomic(self.state_path, json.dumps(self._state, sort_keys=True).encode(), 0o600)

    def _entry(self, ident):
        entry = CATALOG.get(ident) or self._state["imports"].get(ident)
        if not entry:
            raise ModelError(404, "Unknown or protected model")
        return entry

    def _path(self, entry):
        directory = _safe_directory(self.models_dir)
        name = entry["filename"]
        if Path(name).name != name or name in (".", ".."):
            raise ModelError(409, "Unsafe model filename")
        return directory / name

    def _verified_for_status(self, entry):
        if entry["id"] not in self._state["files"]:
            return False
        path = self._path(entry)
        if not _regular(path):
            return False
        st = path.stat(follow_symlinks=False)
        identity = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, entry["sha256"])
        cached = self._verification_cache.get(entry["id"])
        if cached and cached[0] == identity:
            return cached[1]
        result = self._verified(entry)
        self._verification_cache[entry["id"]] = (identity, result)
        return result

    def _verified(self, entry):
        record = self._state["files"].get(entry["id"])
        if not record or record != {"filename": entry["filename"], "sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}:
            return False
        path = self._path(entry)
        if not _regular(path):
            return False
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != entry["size_bytes"]:
                return False
            digest = hashlib.sha256()
            with os.fdopen(fd, "rb") as f:
                fd = -1
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest() == entry["sha256"]
        finally:
            if fd >= 0:
                os.close(fd)

    async def _engine_ids(self):
        settings = self.settings_provider()
        url = settings.get("engine_url", "https://llama.nateshome.net").rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            return None
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
                response = await client.get(url + "/v1/models")
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict) or not isinstance(body.get("data"), list):
                    return None
                ids = set()
                for item in body["data"]:
                    if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                        continue
                    ids.add(item["id"])
                    meta = item.get("meta")
                    swap = meta.get("llamaswap") if isinstance(meta, dict) else None
                    aliases = swap.get("aliases") if isinstance(swap, dict) else None
                    if isinstance(aliases, list):
                        ids.update(alias for alias in aliases if isinstance(alias, str))
                return ids
        except (httpx.HTTPError, ValueError, TypeError):
            return None

    async def list_models(self):
        ids = await self._engine_ids()
        result = [{"id": "shared-turbo", "name": "Whisper Large v3 Turbo", "description": "Existing shared model; managed outside SpeechDock.",
                   "model_id": "whisper-1", "filename": None, "size_bytes": None, "sha256": None,
                   "multilingual": True, "translation_capable": False, "quality": "High", "speed": "Fast",
                   "runtime": "whisper.cpp", "architecture": "whisper", "format": "ggml", "source": "shared",
                   "source_host": None, "source_filename": None, "language_detection_capable": True,
                   "requires_language": False, "checksum_status": "pending", "runtime_ready": ids is not None and "whisper-1" in ids,
                   "runtime_error": None,
                   "installed": False, "verified": False, "registered": ids is not None and "whisper-1" in ids,
                   "read_only": True, "download": None, "can_register": False,
                   "warning": "Shared model; file ownership and integrity are not managed by SpeechDock."}]
        for entry in (*CATALOG.values(), *self._state["imports"].values()):
            try:
                verified = await asyncio.to_thread(self._verified_for_status, entry)
            except (OSError, ModelError):
                verified = False
            download = self._state["downloads"].get(entry["id"])
            runtime = entry.get("runtime", "whisper.cpp")
            runtime_info = self.runtimes.get(runtime, {})
            result.append({k: entry.get(k) for k in ("id", "name", "description", "model_id", "filename", "size_bytes", "sha256", "multilingual", "translation_capable", "quality", "speed", "source_host", "source_filename")} | {
                "runtime": runtime, "architecture": entry.get("architecture", "whisper"),
                "format": entry.get("format", "ggml"), "source": entry.get("source", "catalog"),
                "language_detection_capable": entry.get("language_detection_capable", True),
                "requires_language": entry.get("requires_language", False),
                "checksum_status": entry.get("checksum_status", "pinned"),
                "runtime_ready": bool(runtime_info.get("available")), "runtime_error": runtime_info.get("error"),
                "installed": entry["id"] in self._state["files"], "verified": verified,
                "registered": ids is not None and entry["model_id"] in ids,
                "read_only": False, "download": dict(download) if download else None,
                "can_register": verified and bool(runtime_info.get("available")) and entry["id"] not in self._state["registrations"], "warning": WARNING})
        return result

    async def downloads(self):
        return [dict(item) for item in self._state["downloads"].values()]

    async def install(self, catalog_id):
        entry = self._entry(catalog_id)
        if catalog_id in self._state["imports"]:
            raise ModelError(409, "Manual imports cannot be installed from the catalog")
        async with self._lock:
            task = self._tasks.get(catalog_id)
            if task and not task.done():
                return dict(self._state["downloads"][catalog_id])
            path = self._path(entry)
            if _regular(path):
                raise ModelError(409, "Model file already exists; refusing to overwrite it")
            if shutil.disk_usage(path.parent).free < entry["size_bytes"]:
                raise ModelError(507, "Not enough free disk space for this model")
            item = {"id": catalog_id, "status": "queued", "received_bytes": 0,
                    "total_bytes": entry["size_bytes"], "progress": 0.0, "error": None}
            self._state["downloads"][catalog_id] = item
            self._save()
            self._tasks[catalog_id] = asyncio.create_task(self._download(entry, item))
            return dict(item)

    async def _download(self, entry, item):
        temp = None
        try:
            async with self._serial:
                path = self._path(entry)
                if _regular(path):
                    raise ModelError(409, "Model file appeared during download")
                if shutil.disk_usage(path.parent).free < entry["size_bytes"]:
                    raise ModelError(507, "Not enough free disk space")
                stage = Path(tempfile.mkdtemp(prefix=".speechdock-download-", dir=path.parent))
                stage_info = stage.stat(follow_symlinks=False)
                _sync_dir(path.parent)
                self._state["pending"][entry["id"]] = {"directory": stage.name, "stage_identity": [stage_info.st_dev, stage_info.st_ino]}
                self._save()
                temp = stage / "model.part"
                fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                digest = hashlib.sha256()
                item["status"] = "downloading"
                self._save()
                last_save = time.monotonic()
                with os.fdopen(fd, "wb") as output:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=15), follow_redirects=True, max_redirects=5) as client:
                        async with client.stream("GET", entry["url"]) as response:
                            response.raise_for_status()
                            if urlsplit(entry["url"]).scheme == "https" and any(urlsplit(str(r.url)).scheme != "https" for r in [*response.history, response]):
                                raise ModelError(502, "Insecure download redirect refused")
                            async for chunk in response.aiter_bytes(1024 * 1024):
                                item["received_bytes"] += len(chunk)
                                if item["received_bytes"] > entry["size_bytes"]:
                                    raise ModelError(502, "Download exceeded expected size")
                                digest.update(chunk)
                                output.write(chunk)
                                item["progress"] = item["received_bytes"] / item["total_bytes"]
                                if time.monotonic() - last_save >= 1:
                                    self._save()
                                    last_save = time.monotonic()
                    output.flush()
                    os.fsync(output.fileno())
                item["status"] = "verifying"
                self._save()
                if item["received_bytes"] != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
                    raise ModelError(502, "Downloaded file failed size or checksum verification")
                # Hard-linking cannot overwrite an unexpected destination, unlike rename.
                info = temp.stat(follow_symlinks=False)
                self._state["pending"][entry["id"]]["file_identity"] = [info.st_dev, info.st_ino]
                self._save()
                os.link(temp, path, follow_symlinks=False)
                _sync_dir(path.parent)
                self._state["files"][entry["id"]] = {k: entry[k] for k in ("filename", "sha256", "size_bytes")}
                item.update(status="complete", progress=1.0, error=None)
                self._save()
        except asyncio.CancelledError:
            item.update(status="cancelled", error=None)
            self._save()
        except Exception as exc:
            item.update(status="failed", error=str(exc)[:300] or "Download failed")
            self._save()
        finally:
            if entry["id"] in self._state["pending"]:
                self._recover_downloads()
                self._save()

    async def cancel_download(self, catalog_id):
        self._entry(catalog_id)
        task = self._tasks.get(catalog_id)
        item = self._state["downloads"].get(catalog_id)
        if item and item.get("status") == "awaiting_upload":
            item.update(status="cancelled", error=None)
            self._save()
            return dict(item)
        if not task or task.done():
            raise ModelError(409, "No active download to cancel")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        item = self._state["downloads"][catalog_id]
        if item["status"] in ("queued", "downloading", "uploading", "verifying"):
            item.update(status="cancelled", error=None)
            self._save()
        return dict(self._state["downloads"][catalog_id])

    def _definition(self, entry):
        root = self.engine_dir
        if not root.startswith("/") or not re.fullmatch(r"/[A-Za-z0-9_./-]+", root) or ".." in Path(root).parts:
            raise ModelError(503, "Unsafe engine model directory")
        argv = model_command(entry, Path(root))
        command = " ".join('"${PORT}"' if arg == '${PORT}' else shlex.quote(arg) for arg in argv)
        # Threshold already includes at least 1536 MiB of conservative spare headroom.
        cmd = ("sh -c 'F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d \" \" ); "
               "case \"$F\" in \"\"|*[!0-9]*) echo \"GPU memory unavailable\" >&2; exit 1;; esac; "
               f"if [ \"$F\" -lt {entry['free_mib']} ]; then echo \"Insufficient GPU memory\" >&2; exit 1; fi; "
               f"exec {command}'" )
        health = "/health" if entry.get("runtime") == "crispasr" else "/v1/health"
        return {"name": entry["name"], "cmd": cmd, "proxy": "http://127.0.0.1:${PORT}", "checkEndpoint": health, "ttl": 180}

    def _load_config(self):
        _safe_directory(self.config_path.parent)
        if not _regular(self.config_path):
            raise ModelError(503, "llama-swap configuration is unavailable")
        data = self.config_path.read_bytes()
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.allow_duplicate_keys = False
        try:
            config = yaml.load(data)
        except Exception as exc:
            raise ModelError(409, "Cannot safely parse llama-swap configuration") from exc
        if not isinstance(config, dict) or not isinstance(config.get("models"), dict):
            raise ModelError(409, "Unsupported llama-swap configuration")
        if "routing" in config or "matrix" in config:
            raise ModelError(409, "Custom routing cannot be safely modified")
        if "groups" in config and not isinstance(config["groups"], dict):
            raise ModelError(409, "Unsupported llama-swap groups")
        return data, yaml, config

    def _check_owned(self, config):
        models = config["models"]
        groups = config.get("groups", {})
        owned = self._state["registrations"]
        for ident, definition in owned.items():
            entry = self._entry(ident)
            if entry["model_id"] not in models or _canonical(models[entry["model_id"]]) != definition:
                raise ModelError(409, "An owned model definition was changed externally")
        expected_group = self._state.get("group")
        if expected_group is None:
            if "speechdock" in groups:
                raise ModelError(409, "The speechdock group already exists and is not owned")
        elif "speechdock" not in groups or _canonical(groups["speechdock"]) != expected_group:
            raise ModelError(409, "The SpeechDock group was changed externally")
        for group_name, group in groups.items():
            if not isinstance(group, dict) or not isinstance(group.get("members", []), list):
                raise ModelError(409, "Unsupported group membership")
            if group_name != "speechdock" and any(member in {self._entry(k)["model_id"] for k in owned} for member in group.get("members", [])):
                raise ModelError(409, "An owned model belongs to another group")

    async def _change_registration(self, ident, add, confirm_reload):
        entry = self._entry(ident)
        if not confirm_reload:
            raise ModelError(409, "Confirm the llama-swap reload before applying this change")
        async with self._lock:
            if add and not self.runtimes.get(entry.get("runtime", "whisper.cpp"), {}).get("available"):
                raise ModelError(503, "The required model runtime is unavailable")
            if add and not await asyncio.to_thread(self._verified, entry):
                raise ModelError(409, "Install and verify the owned model before registering")
            if add and ident in self._state["registrations"]:
                raise ModelError(409, "Model is already registered")
            if not add and ident not in self._state["registrations"]:
                raise ModelError(409, "Model registration is not owned by SpeechDock")
            before, yaml, config = self._load_config()
            self._check_owned(config)
            previous_registrations = dict(self._state["registrations"])
            previous_group = self._state["group"]
            models = config["models"]
            model_id = entry["model_id"]
            if add:
                if model_id in models or any(model_id in (m.get("aliases") or []) for m in models.values() if isinstance(m, dict)):
                    raise ModelError(409, "Model ID or alias already exists")
                if any(model_id in g.get("members", []) for g in config.get("groups", {}).values()):
                    raise ModelError(409, "Model ID is already in a group")
                models[model_id] = self._definition(entry)
            else:
                del models[model_id]
            groups = config.setdefault("groups", {})
            members = [self._entry(k)["model_id"] for k in self._state["registrations"] if k != ident]
            if add:
                members.append(model_id)
            if members:
                groups["speechdock"] = {"swap": True, "exclusive": True, "persistent": False, "members": members}
            else:
                del groups["speechdock"]
            stream = io.StringIO()
            yaml.dump(config, stream)
            candidate = stream.getvalue().encode()
            if not self.validator.is_file() or not os.access(self.validator, os.X_OK):
                raise ModelError(503, "The deployed llama-swap validator is unavailable")
            fd, tmp = tempfile.mkstemp(prefix=".speechdock-validate-", suffix=".yaml", dir=self.config_path.parent)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(candidate)
                    f.flush()
                    os.fsync(f.fileno())
                try:
                    proc = await asyncio.create_subprocess_exec(str(self.validator), "-config", tmp, "-validate", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                except (OSError, asyncio.TimeoutError) as exc:
                    if 'proc' in locals() and proc.returncode is None:
                        proc.kill()
                        await proc.wait()
                    raise ModelError(503, "Could not run llama-swap configuration validator") from exc
                if proc.returncode:
                    raise ModelError(409, "llama-swap rejected the configuration: " + (stderr or stdout).decode(errors="replace")[:300])
            finally:
                os.unlink(tmp)
            if not _regular(self.config_path) or self.config_path.read_bytes() != before:
                raise ModelError(409, "Configuration changed during validation; retry")
            st = self.config_path.stat()
            backup = self.data_dir / "llama-swap-config.backup.yaml"
            if backup.exists() or backup.is_symlink():
                _regular(backup)
            _atomic(backup, before, 0o600)
            if not _regular(self.config_path) or self.config_path.read_bytes() != before:
                raise ModelError(409, "Configuration changed during backup; retry")
            _atomic(self.config_path, candidate, stat.S_IMODE(st.st_mode), (st.st_uid, st.st_gid))
            # Persist the ownership fingerprint immediately; external edits are never adopted.
            if add:
                self._state["registrations"][ident] = _canonical(models[model_id])
            else:
                del self._state["registrations"][ident]
            self._state["group"] = _canonical(groups["speechdock"]) if members else None
            try:
                self._save()
            except Exception:
                if _regular(self.config_path) and self.config_path.read_bytes() == candidate:
                    _atomic(self.config_path, before, stat.S_IMODE(st.st_mode), (st.st_uid, st.st_gid))
                self._state["registrations"] = previous_registrations
                self._state["group"] = previous_group
                raise
            # Watcher is poll-based; require the actual engine to report the new state.
            verified = False
            for _ in range(10):
                await asyncio.sleep(1)
                ids = await self._engine_ids()
                if ids is not None and ((model_id in ids) == add):
                    verified = True
                    break
            if not verified:
                # Never overwrite a concurrent external edit, even for rollback.
                if _regular(self.config_path) and self.config_path.read_bytes() == candidate:
                    _atomic(self.config_path, before, stat.S_IMODE(st.st_mode), (st.st_uid, st.st_gid))
                    self._state["registrations"] = previous_registrations
                    self._state["group"] = previous_group
                    self._save()
                    raise ModelError(503, "Engine did not confirm the reload; configuration restored")
                raise ModelError(409, "Engine did not confirm reload and configuration changed externally; no rollback was attempted")
            return {"id": ident, "registered": add, "warning": WARNING}

    async def register(self, catalog_id, confirm_reload=False):
        return await self._change_registration(catalog_id, True, confirm_reload)

    async def unregister(self, catalog_id, confirm_reload=False):
        return await self._change_registration(catalog_id, False, confirm_reload)

    async def delete(self, catalog_id, confirm=False):
        entry = self._entry(catalog_id)
        if catalog_id in self._state["imports"]:
            raise ModelError(409, "Use the import removal action for manual models")
        if not confirm:
            raise ModelError(409, "Confirm deletion of this model file")
        async with self._lock:
            if catalog_id in self._state["registrations"]:
                raise ModelError(409, "Remove the model registration before deleting files")
            task = self._tasks.get(catalog_id)
            if task and not task.done():
                raise ModelError(409, "Cancel the download first")
            if not await asyncio.to_thread(self._verified, entry):
                raise ModelError(409, "File is not an intact SpeechDock-owned model")
            ids = await self._engine_ids()
            if ids is None or entry["model_id"] in ids:
                raise ModelError(409, "Cannot confirm that the engine has removed this model")
            path = self._path(entry)
            if not _regular(path):
                raise ModelError(409, "Model file disappeared")
            path.unlink()
            _sync_dir(path.parent)
            del self._state["files"][catalog_id]
            self._save()
            return {"id": catalog_id, "deleted": True}

    async def create_import(self, *, name, source, url=None, expected_sha256=None,
                            expected_size_bytes=None, source_filename=None):
        if not isinstance(name, str) or not name.strip() or len(name) > 150 or "://" in name:
            raise ModelError(400, "Provide a model name of at most 150 characters")
        if source not in ("upload", "url"):
            raise ModelError(400, "Unsupported import source")
        if expected_sha256 is not None and (not isinstance(expected_sha256, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256)):
            raise ModelError(400, "Expected checksum must be a SHA-256 hex digest")
        if expected_size_bytes is not None and (type(expected_size_bytes) is not int or not 0 < expected_size_bytes <= 8 * 1024**3):
            raise ModelError(400, "Expected model size must be between 1 byte and 8 GiB")
        if source_filename is not None and (not isinstance(source_filename, str) or len(source_filename) > 255 or Path(source_filename).name != source_filename or any(c in source_filename for c in "\\/?#")):
            raise ModelError(400, "Invalid source filename")
        host = None
        if source == "url":
            try:
                parsed = urlsplit(url or "")
                if parsed.scheme != "https" or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.port not in (None, 443):
                    raise ValueError
                host = parsed.hostname
            except ValueError:
                raise ModelError(400, "A public HTTPS URL on port 443 is required") from None
        elif url is not None:
            raise ModelError(400, "Upload imports do not accept a URL")
        ident = "import-" + uuid.uuid4().hex
        entry = {"id": ident, "name": name.strip(), "model_id": "speechdock-" + ident,
                 "filename": ident + ".model", "size_bytes": None, "sha256": None,
                 "expected_sha256": expected_sha256.lower() if expected_sha256 else None,
                 "expected_size_bytes": expected_size_bytes, "source": source,
                 "source_host": host, "source_filename": source_filename,
                 "runtime": None, "architecture": None, "format": None,
                 "multilingual": False, "translation_capable": False,
                 "language_detection_capable": False, "requires_language": False,
                 "checksum_status": "pending", "quality": None, "speed": None,
                 "description": "Manually imported model.", "free_mib": 8192}
        item = {"id": ident, "status": "queued" if source == "url" else "awaiting_upload",
                "received_bytes": 0, "total_bytes": expected_size_bytes, "progress": None, "error": None}
        async with self._lock:
            self._state["imports"][ident] = entry
            self._state["downloads"][ident] = item
            self._save()
            if source == "url":
                self._tasks[ident] = asyncio.create_task(self._import_url(entry, item, url))
        model = next(m for m in await self.list_models() if m["id"] == ident)
        return {"model": model, "download": dict(item)}

    async def _import_url(self, entry, item, url):
        from .model_transfer import stream_model
        try:
            async with stream_model(url) as transfer:
                await self._receive_import(entry, item, transfer.aiter_bytes(), transfer.total_bytes)
        except asyncio.CancelledError:
            item.update(status="cancelled", error=None)
            self._save()
        except Exception:
            if item.get("status") != "failed":
                item.update(status="failed", error="Could not download this model from the supplied URL")
                self._save()

    async def receive_upload(self, catalog_id, stream, content_length=None):
        entry = self._entry(catalog_id)
        async with self._lock:
            item = self._state["downloads"].get(catalog_id)
            if entry.get("source") != "upload" or not item or item["status"] != "awaiting_upload":
                raise ModelError(409, "This import is not awaiting an upload")
            item["status"] = "uploading"
            self._tasks[catalog_id] = asyncio.current_task()
            self._save()
        try:
            await self._receive_import(entry, item, stream, content_length)
        finally:
            if self._tasks.get(catalog_id) is asyncio.current_task():
                self._tasks.pop(catalog_id, None)
        return dict(item)

    async def _receive_import(self, entry, item, stream, content_length):
        from .model_formats import inspect_model
        from .model_transfer import ModelTransferError
        limit = 8 * 1024**3
        ident = entry["id"]
        try:
            if content_length is not None and (type(content_length) is not int or content_length < 0 or content_length > limit):
                raise ModelError(413, "Model exceeds the 8 GiB limit")
            expected = entry["expected_size_bytes"]
            if expected is not None and content_length is not None and expected != content_length:
                raise ModelError(400, "Model size does not match the expected size")
            item["total_bytes"] = content_length if content_length is not None else expected
            async with self._serial:
                path = self._path(entry)
                if path.exists() or path.is_symlink():
                    raise ModelError(409, "Model file already exists")
                if item["total_bytes"] and shutil.disk_usage(path.parent).free < item["total_bytes"]:
                    raise ModelError(507, "Not enough free disk space")
                stage = Path(tempfile.mkdtemp(prefix=".speechdock-download-", dir=path.parent))
                info = stage.stat(follow_symlinks=False)
                _sync_dir(path.parent)
                self._state["pending"][ident] = {"directory": stage.name, "stage_identity": [info.st_dev, info.st_ino]}
                self._save()
                temp = stage / "model.part"
                fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                try:
                    info = os.fstat(fd)
                    self._state["pending"][ident]["part_identity"] = [info.st_dev, info.st_ino]
                    self._save()
                except BaseException:
                    os.close(fd)
                    raise
                digest = hashlib.sha256()
                item["status"] = "uploading" if entry["source"] == "upload" else "downloading"
                last_save = time.monotonic()
                with os.fdopen(fd, "wb") as output:
                    async for chunk in stream:
                        if not isinstance(chunk, (bytes, bytearray)):
                            raise ModelError(400, "Invalid model stream")
                        total = item["received_bytes"] + len(chunk)
                        if total > limit or (item["total_bytes"] is not None and total > item["total_bytes"]):
                            raise ModelError(413, "Model exceeds its declared size or the 8 GiB limit")
                        if len(chunk) > shutil.disk_usage(path.parent).free:
                            raise ModelError(507, "Not enough free disk space")
                        output.write(chunk)
                        digest.update(chunk)
                        item["received_bytes"] = total
                        item["progress"] = total / item["total_bytes"] if item["total_bytes"] else None
                        if time.monotonic() - last_save >= 1:
                            self._save()
                            last_save = time.monotonic()
                    output.flush()
                    os.fsync(output.fileno())
                item["status"] = "verifying"
                self._save()
                if not item["received_bytes"] or (item["total_bytes"] is not None and item["received_bytes"] != item["total_bytes"]):
                    raise ModelError(400, "Model transfer was truncated")
                value = digest.hexdigest()
                if entry["expected_sha256"] and value != entry["expected_sha256"]:
                    raise ModelError(400, "Model checksum does not match the expected SHA-256")
                try:
                    profile = await asyncio.to_thread(inspect_model, temp)
                except (ValueError, OSError):
                    raise ModelError(400, "Unsupported, malformed or truncated model file") from None
                entry.update(profile, size_bytes=item["received_bytes"], sha256=value,
                             free_mib=max(8192, (item["received_bytes"] + 1024**2 - 1) // 1024**2 + 2048),
                             checksum_status="matched" if entry["expected_sha256"] else "computed")
                info = temp.stat(follow_symlinks=False)
                self._state["pending"][ident]["file_identity"] = [info.st_dev, info.st_ino]
                self._save()
                os.link(temp, path, follow_symlinks=False)
                _sync_dir(path.parent)
                self._state["files"][ident] = {k: entry[k] for k in ("filename", "sha256", "size_bytes")}
                item.update(status="completed", progress=1.0, error=None)
                self._save()
        except asyncio.CancelledError:
            item.update(status="cancelled", error=None)
            self._save()
            raise
        except Exception as exc:
            message = str(exc) if isinstance(exc, (ModelError, ModelTransferError)) else "Model transfer failed"
            item.update(status="failed", error=message)
            self._save()
            if isinstance(exc, ModelError):
                raise
            raise ModelError(400, message) from None
        finally:
            if ident in self._state["pending"]:
                self._recover_downloads()
                self._save()

    async def remove_import(self, catalog_id, confirm=False):
        entry = self._entry(catalog_id)
        if catalog_id not in self._state["imports"]:
            raise ModelError(404, "Unknown manual import")
        if not confirm:
            raise ModelError(409, "Confirm deletion of this import")
        async with self._lock:
            item = self._state["downloads"].get(catalog_id, {})
            task = self._tasks.get(catalog_id)
            if catalog_id in self._state["registrations"]:
                raise ModelError(409, "Unregister this model first")
            if (task and not task.done()) or item.get("status") in ("awaiting_upload", "queued", "downloading", "uploading", "verifying"):
                raise ModelError(409, "Cancel the import first")
            if catalog_id in self._state["pending"]:
                raise ModelError(409, "Import recovery is incomplete")
            if catalog_id in self._state["files"]:
                if not await asyncio.to_thread(self._verified, entry):
                    raise ModelError(409, "File is not an intact SpeechDock-owned model")
                ids = await self._engine_ids()
                if ids is None or entry["model_id"] in ids:
                    raise ModelError(409, "Cannot confirm the engine has removed this model")
                path = self._path(entry)
                if not _regular(path):
                    raise ModelError(409, "Model file disappeared")
                path.unlink()
                _sync_dir(path.parent)
                del self._state["files"][catalog_id]
            else:
                path = self._path(entry)
                if path.exists() or path.is_symlink():
                    raise ModelError(409, "A foreign file occupies this import's path")
            del self._state["imports"][catalog_id]
            self._state["downloads"].pop(catalog_id, None)
            self._save()
            return {"id": catalog_id, "deleted": True}

    async def close(self):
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
