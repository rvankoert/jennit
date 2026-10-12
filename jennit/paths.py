"""Safety of relative folder paths in batch uploads."""

MAX_REL_PATH_LEN = 512
MAX_SEGMENT_LEN = 200


def sanitize_rel_path(path: str | None) -> str:
    """Return a safe, normalized relative path (``''`` means root).

    Rules: forward slashes only; no absolute paths, backslashes, control
    characters or ``..`` segments; empty and ``.`` segments are dropped.
    Raises ``ValueError`` on anything else.
    """
    if not path or not path.strip():
        return ""
    p = path.replace("\\", "/")
    if p.startswith("/"):
        raise ValueError("absolute paths are not allowed")
    parts: list[str] = []
    for part in p.split("/"):
        part = part.strip()
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError("path traversal ('..') is not allowed")
        if any(ord(c) < 32 for c in part):
            raise ValueError(f"control characters in path segment: {part!r}")
        if len(part) > MAX_SEGMENT_LEN:
            raise ValueError(f"path segment too long: {part!r}")
        parts.append(part)
    rel = "/".join(parts)
    if len(rel) > MAX_REL_PATH_LEN:
        raise ValueError("path too long")
    return rel
