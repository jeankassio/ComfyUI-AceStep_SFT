"""Read declared adapter triggers without guessing from tensor keys or tag counts."""

import hashlib
from http.client import HTTPException
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from safetensors import safe_open


_MAX_METADATA_BYTES = 2 * 1024 * 1024
_TRIGGER_KEYS = (
    "modelspec.trigger_phrase", "modelspec.trigger_word", "trigger_word",
    "trigger_words", "trigger", "trainedWords", "ss_trained_words",
    "activation text", "activation_text",
)


def _words(value):
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            try:
                value = json.loads(value)
            except (ValueError, RecursionError):
                return ""
        else:
            return value
    if isinstance(value, list) and all(isinstance(word, str) for word in value):
        return ", ".join(dict.fromkeys(word.strip() for word in value if word.strip()))
    return ""


def _sidestep_trigger(data):
    if not isinstance(data, dict):
        return ""
    metadata = data.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (ValueError, RecursionError):
            metadata = {}
    default = _words(data.get("custom_tag")) or (
        _words(metadata.get("custom_tag")) if isinstance(metadata, dict) else ""
    )
    samples = data.get("samples")
    if isinstance(samples, list) and samples:
        effective = {
            _words(sample.get("custom_tag")) or default
            for sample in samples if isinstance(sample, dict)
        }
        # Per-song tags can describe different identities, not one combined trigger.
        return effective.pop() if len(effective) == 1 else ""
    return default


def _declared_trigger(data, depth=0):
    if not isinstance(data, dict):
        return ""
    for key in _TRIGGER_KEYS:
        trigger = _words(data.get(key))
        if trigger:
            return trigger
    metadata = data.get("metadata")
    if isinstance(metadata, dict) and depth < 2:
        return _declared_trigger(metadata, depth + 1)
    return ""


def trigger_metadata_files(lora_path):
    path = Path(lora_path)
    files = [path.with_suffix(suffix) for suffix in (
        ".sidestep.json", ".trigger.txt", ".json", ".civitai.info", ".info",
    )]
    files.extend([Path(str(path) + ".json"), Path(str(path) + ".civitai.info")])
    # Generic dataset filenames are meaningful only inside an adapter directory.
    # Never follow dataset paths supplied by a training config or remote metadata.
    if (path.parent / "adapter_config.json").is_file():
        files.extend(path.parent / name for name in (
            "sidestep_metadata.json", "preprocess_meta.json", "dataset.json",
        ))
    return files


def _read_sidecar(path):
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_METADATA_BYTES + 1)
        if len(raw) > _MAX_METADATA_BYTES:
            return {}
        text = raw.decode("utf-8-sig")
        return {"custom_tag": text.strip()} if path.suffix == ".txt" else json.loads(text)
    except (OSError, ValueError, RecursionError):
        return {}


def _civitai_trigger(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    request = Request(
        "https://civitai.com/api/v1/model-versions/by-hash/" + digest.hexdigest(),
        headers={"Accept": "application/json", "User-Agent": "ComfyUI-AceStep-SFT"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            raw = response.read(_MAX_METADATA_BYTES + 1)
        if len(raw) > _MAX_METADATA_BYTES:
            return ""
        data = json.loads(raw)
        return _words(data.get("trainedWords")) if isinstance(data, dict) else ""
    except (URLError, OSError, HTTPException, ValueError, RecursionError):
        print("[AceStep SFT][LoRA] Civitai trigger lookup unavailable; use local metadata or the override.")
        return ""


def local_trigger_word(lora_path):
    path = Path(lora_path)
    metadata = {}
    if path.suffix.lower() == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as adapter:
            metadata = adapter.metadata() or {}
    trigger = _sidestep_trigger(metadata)
    if trigger:
        return trigger
    sidecars = [_read_sidecar(file) for file in trigger_metadata_files(path)]
    for data in sidecars:
        trigger = _sidestep_trigger(data)
        if trigger:
            return trigger
    trigger = _declared_trigger(metadata)
    if trigger:
        return trigger
    for data in sidecars:
        trigger = _declared_trigger(data)
        if trigger:
            return trigger
    return ""


def resolve_trigger_word(lora_path, trigger_word_override="", lookup_civitai=True):
    override = trigger_word_override.strip()
    if override:
        return override
    path = Path(lora_path)
    trigger = local_trigger_word(path)
    if trigger:
        return trigger
    if lookup_civitai:
        trigger = _civitai_trigger(path)
        if trigger:
            return trigger
    print("[AceStep SFT][LoRA] No declared trigger found; output is empty. Use trigger_word_override if needed.")
    return ""


def trigger_file_signature(lora_path):
    path = Path(lora_path)
    files = [path, *trigger_metadata_files(path)]
    stats = []
    for file in files:
        try:
            stat = file.stat()
            stats.append((str(file), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        except FileNotFoundError:
            stats.append((str(file), None))
    return hashlib.sha256(repr(stats).encode("utf-8")).hexdigest()
