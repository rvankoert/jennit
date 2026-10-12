"""Asyncio work queue + thread pool that processes uploaded scans.

One scan at a time per worker: the spooled file is analyzed with the
existing process_image() (in a worker thread), the result row is committed
to Postgres, then the spool file and intermediate artifacts are deleted —
only the report-relevant columns remain, per the retention requirement.
"""
import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import analysis, db, report
from .config import Settings

log = logging.getLogger("jennit.pipeline")


def map_row_to_scan(row: dict | None, pipeline_error: str | None = None) -> dict:
    """Convert process_image()'s result dict into scans-table column values."""
    if row is None:
        return {
            "is_valid": False, "is_overexposed": None, "compression_ratio": None,
            "jpeg_quality": None, "black_pixels": 0, "white_pixels": 0, "angle": None,
            "md5": None, "sha512": None, "dpi": "[null, null]", "exif": None,
            "color_profile": None, "color_histograms": None,
            "errors": pipeline_error or "processing crashed before any result",
        }

    def jdump(value):
        if value is None:
            return None
        return json.dumps(analysis.jsonable(value), default=str)

    return {
        # analysis.jsonable() converts numpy scalars (np.bool_, np.float64, ...)
        # to native Python types, which asyncpg requires for parameter binding.
        "is_valid": analysis.jsonable(row["Is Valid"]),
        "is_overexposed": analysis.jsonable(row["Is Overexposed"]),
        "compression_ratio": analysis.jsonable(row["Compression Ratio"]),
        "jpeg_quality": analysis.jsonable(row["JPEG Quality"]),
        "black_pixels": analysis.jsonable(row["Black Pixels"]),
        "white_pixels": analysis.jsonable(row["White Pixels"]),
        "angle": analysis.jsonable(row["Angle"]),
        "md5": row["MD5 Checksum"],
        "sha512": row["SHA-512 Hash"],
        "dpi": json.dumps(analysis.jsonable(row["DPI"])),
        "exif": jdump(row["EXIF Metadata"]),
        "color_profile": jdump(row["Color Profile"]),
        "color_histograms": jdump(row["Color Histograms"]),
        "errors": row["Errors"] or None,
    }


class Pipeline:
    def __init__(self, pool, settings: Settings):
        self.pool = pool
        self.settings = settings
        self.queue: asyncio.Queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=settings.threads,
                                           thread_name_prefix="jennit-scan")
        self._workers: list[asyncio.Task] = []

    async def start(self) -> None:
        for _ in range(self.settings.threads):
            self._workers.append(asyncio.create_task(self._worker()))

    async def requeue_stale(self) -> None:
        """Re-enqueue scans that were pending/processing at last shutdown.

        Their spool files survive the crash (they are only deleted after
        the row is committed); if a file is missing, the worker marks the
        scan failed with "spool file missing".
        """
        rows = await self.pool.fetch(
            "SELECT batch_id, id, rel_path FROM scans "
            "WHERE status IN ('pending', 'processing')")
        for r in rows:
            self.queue.put_nowait((r["batch_id"], r["id"], r["rel_path"]))
        if rows:
            log.info("requeued %d stale scan(s)", len(rows))

    async def stop(self) -> None:
        for _ in self._workers:
            self.queue.put_nowait(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self.executor.shutdown(wait=True)

    async def cleanup_orphans(self) -> None:
        """Remove leftover .part-* temp files and .tmp dirs from a crashed run."""
        spool_root = self.settings.spool_root
        if not spool_root.exists():
            return
        for part in spool_root.glob("*/**/*.part-*"):
            try:
                if time.time() - part.stat().st_mtime > 3600:
                    part.unlink(missing_ok=True)
            except OSError:
                pass
        for tmp in spool_root.glob("*/.tmp/*"):
            shutil.rmtree(tmp, ignore_errors=True)

    async def _worker(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    break
                batch_id, scan_id, rel_path = item
                await self._process(batch_id, scan_id, rel_path)
            except Exception:
                log.exception("worker failed on %s", item)
            finally:
                self.queue.task_done()

    async def _process(self, batch_id: uuid.UUID, scan_id: int, rel_path: str) -> None:
        spool = self.settings.spool_root / str(batch_id) / rel_path
        if not spool.exists():
            await self._record(batch_id, scan_id, None, f"spool file missing: {rel_path}")
            return
        await db.mark_processing(self.pool, scan_id)
        tmp_dir = None
        row, error = None, None
        try:
            tmp_parent = self.settings.spool_root / str(batch_id) / ".tmp"
            tmp_parent.mkdir(parents=True, exist_ok=True)
            tmp_dir = Path(tempfile.mkdtemp(prefix="scan_", dir=tmp_parent))
            row = await asyncio.get_running_loop().run_in_executor(
                self.executor, _analyze, str(spool), str(tmp_dir),
                self.settings.pixels_threshold)
        except Exception as exc:
            error = f"pipeline: {exc}"
        recorded = await self._record(batch_id, scan_id, row, error)
        if recorded:
            await self._cleanup(batch_id, spool, tmp_dir)
        else:
            # keep the spool file: the scan is still pending/processing in
            # Postgres and requeue_stale() will retry it after a restart
            log.error("keeping spool file %s for retry after restart", spool)

    async def _record(self, batch_id: uuid.UUID, scan_id: int,
                      row: dict | None, error: str | None) -> bool:
        values = map_row_to_scan(row, pipeline_error=error)
        try:
            completed = await db.finish_scan(self.pool, scan_id, batch_id,
                                             ok=row is not None, values=values)
        except Exception:
            log.exception("could not record scan %s", scan_id)
            return False
        if completed:
            try:
                path = await report.write_report(self.pool, self.settings, batch_id)
                log.info("report for batch %s written to %s", batch_id, path)
            except Exception:
                log.exception(
                    "report generation failed for batch %s "
                    "(retry with POST /api/v1/batches/%s/report)",
                    batch_id, batch_id)
        return True

    async def _cleanup(self, batch_id: uuid.UUID, spool: Path,
                       tmp_dir: Path | None) -> None:
        if tmp_dir is not None:
            if self.settings.keep_intermediates:
                dest_dir = (self.settings.data_root / "batches"
                            / str(batch_id) / "artifacts")
                dest_dir.mkdir(parents=True, exist_ok=True)
                prefix = (spool.relative_to(self.settings.spool_root)
                          .as_posix().replace("/", "__"))
                for artifact in sorted(tmp_dir.iterdir()):
                    try:
                        os.replace(artifact, dest_dir / f"{prefix}__{artifact.name}")
                    except OSError:
                        pass
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if not self.settings.keep_intermediates:
            spool.unlink(missing_ok=True)
            self._prune_empty_dirs(spool.parent)

    def _prune_empty_dirs(self, directory: Path) -> None:
        """Remove empty directories up to and including the per-batch spool
        dir (e.g. spool/<id>/sub after its last file was cleaned), so the
        spool check `ls spool/<id>` stays empty. Stops at spool_root."""
        stop = self.settings.spool_root
        current = directory
        while current != stop:
            try:
                if any(current.iterdir()):
                    return
                current.rmdir()
            except OSError:
                return
            current = current.parent


def _analyze(image_path: str, output_dir: str, pixels_threshold: int):
    return analysis.process_image(image_path, output_dir, pixels_threshold)
