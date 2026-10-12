"""Environment-driven settings. Every value can be set via a JENNIT_* env var."""
import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}")


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    if raw.strip().lower() in ("1", "true", "yes", "on"):
        return True
    if raw.strip().lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _path(name: str, default: str) -> Path:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return Path(default)
    return Path(raw).expanduser()


@dataclass(frozen=True)
class Settings:
    db_dsn: str
    spool_root: Path
    data_root: Path
    threads: int
    api_token: str | None
    max_upload_bytes: int
    pixels_threshold: int
    keep_intermediates: bool
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls(
            db_dsn=os.environ.get(
                "JENNIT_DB_DSN",
                "postgresql://jennit:jennit@127.0.0.1:5432/jennit"),
            spool_root=_path("JENNIT_SPOOL_ROOT", "./spool"),
            data_root=_path("JENNIT_DATA_ROOT", "./data"),
            threads=_int("JENNIT_THREADS", 16),
            api_token=os.environ.get("JENNIT_API_TOKEN") or None,
            max_upload_bytes=_int("JENNIT_MAX_UPLOAD_BYTES", 2 * 1024 * 1024 * 1024),
            pixels_threshold=_int("JENNIT_PIXELS_THRESHOLD", 5),
            keep_intermediates=_bool("JENNIT_KEEP_INTERMEDIATES", False),
            host=os.environ.get("JENNIT_HOST", "0.0.0.0"),
            port=_int("JENNIT_PORT", 8000),
        )
        if s.threads < 1:
            raise ValueError("JENNIT_THREADS must be >= 1")
        if s.max_upload_bytes < 1:
            raise ValueError("JENNIT_MAX_UPLOAD_BYTES must be >= 1")
        return s
