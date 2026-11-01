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
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from ruamel.yaml import YAML

from .catalog import CATALOG


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
        self._state = {"files": {}, "registrations": {}, "group": None, "downloads": {}, "pending": {}}
        if _regular(self.state_path):
            try:
                data = json.loads(self.state_path.read_text())
                if not isinstance(data, dict) or any(not isinstance(data.get(k), dict) for k in ("files", "registrations", "downloads")):
                    raise ValueError("invalid state")
                self._state.update(data)
            except (ValueError, OSError) as exc:
                raise ModelError(503, "Model ownership state is unreadable; refusing unsafe changes") from exc
        self._recover_downloads()
        for item in self._state["downloads"].values():
            if item.get("status") in ("queued", "downloading", "verifying"):
                item.update(status="failed", error="Download interrupted by restart")
        self._save()
    def _recover_downloads(self):
        """Recover publications whose inode belongs to our private staging directory."""
        for ident, pending in list(self._state.get("pending", {}).items()):
            entry = CATALOG.get(ident)
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
                            self._state["downloads"][ident].update(status="complete", progress=1.0, error=None)
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
        if ident not in CATALOG:
            raise ModelError(404, "Unknown or protected model")
        return CATALOG[ident]

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
                   "installed": False, "verified": False, "registered": ids is not None and "whisper-1" in ids,
                   "read_only": True, "download": None, "can_register": False,
                   "warning": "Shared model; file ownership and integrity are not managed by SpeechDock."}]
        for entry in CATALOG.values():
            try:
                verified = await asyncio.to_thread(self._verified_for_status, entry)
            except (OSError, ModelError):
                verified = False
            download = self._state["downloads"].get(entry["id"])
            result.append({k: entry[k] for k in ("id", "name", "description", "model_id", "filename", "size_bytes", "sha256", "multilingual", "translation_capable", "quality", "speed")} | {
                "installed": entry["id"] in self._state["files"], "verified": verified,
                "registered": ids is not None and entry["model_id"] in ids,
                "read_only": False, "download": dict(download) if download else None,
                "can_register": verified and entry["id"] not in self._state["registrations"], "warning": WARNING})
        return result

    async def downloads(self):
        return [dict(item) for item in self._state["downloads"].values()]

    async def install(self, catalog_id):
        entry = self._entry(catalog_id)
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
        if not task or task.done():
            raise ModelError(409, "No active download to cancel")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        item = self._state["downloads"][catalog_id]
        if item["status"] in ("queued", "downloading", "verifying"):
            item.update(status="cancelled", error=None)
            self._save()
        return dict(self._state["downloads"][catalog_id])

    def _definition(self, entry):
        root = self.engine_dir
        if not root.startswith("/") or not re.fullmatch(r"/[A-Za-z0-9_./-]+", root) or ".." in Path(root).parts:
            raise ModelError(503, "Unsafe engine model directory")
        model_path = shlex.quote(root.rstrip("/") + "/" + entry["filename"])
        # Threshold already includes at least 1536 MiB of conservative spare headroom.
        cmd = ("sh -c 'F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d \" \" ); "
               "case \"$F\" in \"\"|*[!0-9]*) echo \"GPU memory unavailable\" >&2; exit 1;; esac; "
               f"if [ \"$F\" -lt {entry['free_mib']} ]; then echo \"Insufficient GPU memory\" >&2; exit 1; fi; "
               f"exec whisper-server --host 127.0.0.1 --port ${{PORT}} -m {model_path} "
               "--request-path /v1 --inference-path /audio/transcriptions --language auto'" )
        return {"name": entry["name"], "cmd": cmd, "proxy": "http://127.0.0.1:${PORT}", "checkEndpoint": "/v1/health", "ttl": 180}

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
            if group_name != "speechdock" and any(member in {CATALOG[k]["model_id"] for k in owned} for member in group.get("members", [])):
                raise ModelError(409, "An owned model belongs to another group")

    async def _change_registration(self, ident, add, confirm_reload):
        entry = self._entry(ident)
        if not confirm_reload:
            raise ModelError(409, "Confirm the llama-swap reload before applying this change")
        async with self._lock:
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
            members = [CATALOG[k]["model_id"] for k in self._state["registrations"] if k != ident]
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

    async def close(self):
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
