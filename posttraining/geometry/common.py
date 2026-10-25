from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path

from safetensors.torch import save_file

from ..runtime import atomic_json


FORMAT = 'dust-activation-geometry-v1'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_record(path, root):
    path, root = Path(path), Path(root)
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha256(path))


def check_file(root, record):
    path = Path(root) / record['path']
    if not path.is_file() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
        raise ValueError(f'Artifact size/checksum mismatch: {path}')
    return path


def atomic_tensors(path, tensors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, str(temporary))
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def site_seed(seed, site, stream, rank=0):
    key = f'{seed}:{site}:{stream}:{rank}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2**63-1)
