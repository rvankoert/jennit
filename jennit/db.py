"""PostgreSQL access for jennit batches (separate database, see JENNIT_DB_DSN)."""
import re
import uuid
from pathlib import Path
from urllib.parse import urlparse

import asyncpg

SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"


async def create_pool(dsn: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, min_size=2, max_size=20, command_timeout=300)


def database_name(dsn: str) -> str:
    name = urlparse(dsn).path.lstrip("/")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"unsafe database name in DSN: {name!r}")
    return name


async def ensure_database(dsn: str) -> None:
    """Create the target database if it does not exist (connect via the 'postgres' db)."""
    name = database_name(dsn)
    admin_dsn = dsn.rsplit("/", 1)[0] + "/postgres"
    conn = await asyncpg.connect(admin_dsn)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()


async def init_schema(pool: asyncpg.Pool) -> None:
    # asyncpg uses the extended protocol: one statement per execute(),
    # so schema.sql must stay plain single statements separated by ';'.
    statements = [s.strip() for s in SCHEMA_FILE.read_text().split(";")]
    async with pool.acquire() as conn:
        for statement in statements:
            if statement:
                await conn.execute(statement)


async def create_batch(pool, name: str, client_ref: str | None = None) -> dict:
    row = await pool.fetchrow(
        "INSERT INTO batches (name, client_ref) VALUES ($1, $2) RETURNING *",
        name, client_ref,
    )
    return dict(row)


async def get_batch(pool, batch_id: uuid.UUID) -> dict | None:
    row = await pool.fetchrow("SELECT * FROM batches WHERE id = $1", batch_id)
    return dict(row) if row else None


async def list_batches(pool) -> list[dict]:
    rows = await pool.fetch(
        "SELECT id, name, status, uploaded_count, processed_count, failed_count, "
        "created_at, completed_at FROM batches ORDER BY created_at DESC"
    )
    return [dict(r) for r in rows]


async def signal_complete(pool, batch_id: uuid.UUID) -> bool:
    """Flip complete_signaled, and — when every scan is already done —
    complete the batch here, because the in-scan completion check inside
    finish_scan can only fire from a scan that finishes *after* the signal.

    Returns True if this call switched the batch to 'completed'.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE batches SET complete_signaled = TRUE, status = 'processing' "
                "WHERE id = $1 AND status IN ('uploading', 'processing')",
                batch_id,
            )
            completed = await conn.fetchval(
                """
                WITH b AS (
                    SELECT 1 FROM batches
                    WHERE id = $1
                      AND status = 'processing' AND complete_signaled
                      AND (processed_count + failed_count) >= uploaded_count
                      FOR UPDATE
                )
                UPDATE batches SET status = 'completed', completed_at = now()
                WHERE id = $1 AND status = 'processing' AND complete_signaled
                  AND (processed_count + failed_count) >= uploaded_count
                RETURNING id
                """,
                batch_id,
            )
    return completed is not None


async def delete_batch(pool, batch_id: uuid.UUID) -> None:
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM batches WHERE id = $1", batch_id)


async def enqueue_scan(pool, batch_id: uuid.UUID, rel_path: str,
                       file_name: str, file_size: int) -> tuple[int, bool]:
    """Insert the scan (or re-queue an existing path) and bump batch counters.

    Returns (scan_id, inserted). `inserted` is False when the same rel_path
    was uploaded before (re-upload): the row is reset to 'pending' and
    uploaded_count is NOT incremented again.
    """
    folder = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
    async with pool.acquire() as conn:
        async with conn.transaction():
            scan = await conn.fetchrow(
                """
                INSERT INTO scans (batch_id, rel_path, folder_path, file_name,
                                   file_size_bytes, status)
                VALUES ($1, $2, $3, $4, $5, 'pending')
                ON CONFLICT (batch_id, rel_path) DO UPDATE
                    SET status = 'pending',
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        uploaded_at = now()
                RETURNING id, (xmax = 0) AS inserted
                """,
                batch_id, rel_path, folder, file_name, file_size,
            )
            # xmax = 0 on INSERT, > 0 on UPDATE — tells us which branch ran.
            await conn.execute(
                """
                UPDATE batches
                SET uploaded_count = uploaded_count + $2,
                    status = CASE WHEN status = 'completed' THEN 'processing'
                                  ELSE status END,
                    complete_signaled = CASE WHEN status = 'completed' THEN FALSE
                                             ELSE complete_signaled END,
                    completed_at = CASE WHEN status = 'completed' THEN NULL
                                        ELSE completed_at END
                WHERE id = $1
                """,
                batch_id, 1 if scan["inserted"] else 0,
            )
    return scan["id"], scan["inserted"]


async def mark_processing(pool, scan_id: int) -> None:
    await pool.execute(
        "UPDATE scans SET status = 'processing' WHERE id = $1 AND status = 'pending'",
        scan_id,
    )


async def finish_scan(pool, scan_id: int, batch_id: uuid.UUID,
                      ok: bool, values: dict) -> bool:
    """Persist the result row, bump the batch counter, and mark the batch
    completed when the last scan finishes (FOR UPDATE serializes racers).

    `values` must contain exactly the keys: is_valid, is_overexposed,
    compression_ratio, jpeg_quality, black_pixels, white_pixels, angle,
    md5, sha512, dpi, exif, color_profile, color_histograms, errors
    (the JSONB columns already as JSON *strings* or None).
    Returns True if this call completed the batch.
    """
    status = "done" if ok else "failed"
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                UPDATE scans SET
                    status = $2, finished_at = now(),
                    is_valid = $3, is_overexposed = $4,
                    compression_ratio = $5, jpeg_quality = $6,
                    black_pixels = $7, white_pixels = $8, angle = $9,
                    md5 = $10, sha512 = $11, dpi = $12, exif = $13,
                    color_profile = $14, color_histograms = $15, errors = $16
                WHERE id = $1
                """,
                scan_id, status, values["is_valid"], values["is_overexposed"],
                values["compression_ratio"], values["jpeg_quality"],
                values["black_pixels"], values["white_pixels"], values["angle"],
                values["md5"], values["sha512"], values["dpi"], values["exif"],
                values["color_profile"], values["color_histograms"], values["errors"],
            )
            counter = "processed_count" if ok else "failed_count"
            await conn.execute(
                f"UPDATE batches SET {counter} = {counter} + 1 WHERE id = $1",
                batch_id,
            )
            completed = await conn.fetchval(
                """
                WITH b AS (
                    SELECT 1 FROM batches
                    WHERE id = $1
                      AND status = 'processing' AND complete_signaled
                      AND (processed_count + failed_count) >= uploaded_count
                      FOR UPDATE
                )
                UPDATE batches SET status = 'completed', completed_at = now()
                WHERE id = $1 AND status = 'processing' AND complete_signaled
                  AND (processed_count + failed_count) >= uploaded_count
                RETURNING id
                """,
                batch_id,
            )
    return completed is not None


async def folder_stats(pool, batch_id: uuid.UUID) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT COALESCE(NULLIF(folder_path, ''), '(root)') AS folder,
               count(*) AS total,
               count(*) FILTER (WHERE status = 'done') AS done,
               count(*) FILTER (WHERE status = 'failed') AS failed,
               count(*) FILTER (WHERE status IN ('pending', 'processing')) AS in_flight
        FROM scans WHERE batch_id = $1
        GROUP BY folder ORDER BY folder
        """,
        batch_id,
    )
    return [dict(r) for r in rows]


async def page_scans(pool, batch_id: uuid.UUID, *, limit: int = 100,
                     offset: int = 0, status: str | None = None,
                     folder: str | None = None) -> list[dict]:
    sql = "SELECT * FROM scans WHERE batch_id = $1"
    args: list = [batch_id]
    if status:
        args.append(status)
        sql += f" AND status = ${len(args)}"
    if folder is not None:
        args.append(folder)
        sql += f" AND folder_path = ${len(args)}"
    args.extend([limit, offset])
    sql += f" ORDER BY id LIMIT ${len(args) - 1} OFFSET ${len(args)}"
    return [dict(r) for r in await pool.fetch(sql, *args)]


EXPORT_SELECT = """
    SELECT rel_path, folder_path, status, finished_at, is_valid, is_overexposed,
           file_size_bytes, md5, sha512, dpi, compression_ratio, jpeg_quality,
           black_pixels, white_pixels, angle, color_profile, color_histograms,
           exif, errors
    FROM scans
"""


async def iter_export_rows(pool, batch_id: uuid.UUID, folder: str | None = None):
    """Async generator yielding per-scan rows (dicts) in rel_path order, 5000 at a time."""
    sql = EXPORT_SELECT + " WHERE batch_id = $1"
    args: list = [batch_id]
    if folder is not None:
        args.append(folder)
        sql += f" AND folder_path = ${len(args)}"
    sql += " ORDER BY rel_path"
    async with pool.acquire() as conn:
        # a server-side cursor requires an explicit transaction in asyncpg;
        # the cursor is released implicitly when the transaction ends
        # (no Cursor.close() method exists)
        async with conn.transaction():
            cursor = await conn.cursor(sql, *args)
            while True:
                rows = await cursor.fetch(5000)
                if not rows:
                    break
                for row in rows:
                    yield dict(row)


if __name__ == "__main__":
    import asyncio

    from .config import Settings

    async def _main() -> None:
        s = Settings.from_env()
        await ensure_database(s.db_dsn)
        pool = await create_pool(s.db_dsn)
        try:
            await init_schema(pool)
            print("database ready:", database_name(s.db_dsn))
        finally:
            await pool.close()

    asyncio.run(_main())
