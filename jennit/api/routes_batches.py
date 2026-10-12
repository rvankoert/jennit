"""REST endpoints for jennit batches."""
import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import (APIRouter, Depends, File, Form, HTTPException, Query,
                     Request, UploadFile)
from fastapi.responses import FileResponse, StreamingResponse

from .. import analysis, db, report
from ..config import Settings
from ..paths import sanitize_rel_path
from .auth import require_token

log = logging.getLogger("jennit.api")

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_token)])

CHUNK_SIZE = 1024 * 1024  # stream uploads in 1 MB chunks


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _pipeline(request: Request):
    return request.app.state.pipeline


def _batch_or_404(batch: dict | None) -> dict:
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found")
    return batch


def _batch_json(batch: dict) -> dict:
    return {
        "id": str(batch["id"]),
        "name": batch["name"],
        "status": batch["status"],
        "uploaded_count": batch["uploaded_count"],
        "processed_count": batch["processed_count"],
        "failed_count": batch["failed_count"],
        "created_at": batch["created_at"].isoformat()
        if batch["created_at"] else None,
        "completed_at": batch["completed_at"].isoformat()
        if batch["completed_at"] else None,
        "report_path": batch.get("report_path"),
    }


def _scan_json(row: dict) -> dict:
    out = dict(row)
    for key in ("uploaded_at", "finished_at"):
        if out.get(key) is not None:
            out[key] = out[key].isoformat()
    return out


@router.post("/batches")
async def create_batch(request: Request,
                       name: str = Form(...),
                       client_ref: str | None = Form(None)) -> dict:
    s = _settings(request)
    if not name.strip():
        raise HTTPException(status_code=400, detail="name must not be empty")
    batch = await db.create_batch(request.app.state.pool, name.strip()[:200],
                                  client_ref)
    (s.spool_root / str(batch["id"])).mkdir(parents=True, exist_ok=True)
    return _batch_json(batch)


@router.get("/batches")
async def list_batches(request: Request) -> list[dict]:
    return [_batch_json(b) for b in await db.list_batches(request.app.state.pool)]


@router.get("/batches/{batch_id}")
async def get_batch(request: Request, batch_id: uuid.UUID) -> dict:
    pool = request.app.state.pool
    batch = _batch_or_404(await db.get_batch(pool, batch_id))
    out = _batch_json(batch)
    out["folders"] = await db.folder_stats(pool, batch_id)
    return out


@router.get("/batches/{batch_id}/scans")
async def list_scans(request: Request, batch_id: uuid.UUID,
                     limit: int = Query(100, ge=1, le=500),
                     offset: int = Query(0, ge=0),
                     status: str | None = Query(None),
                     folder: str | None = Query(None)) -> list[dict]:
    pool = request.app.state.pool
    _batch_or_404(await db.get_batch(pool, batch_id))
    rows = await db.page_scans(pool, batch_id, limit=limit, offset=offset,
                               status=status, folder=folder)
    return [_scan_json(r) for r in rows]


@router.post("/batches/{batch_id}/files")
async def upload_file(request: Request, batch_id: uuid.UUID,
                      file: UploadFile = File(...),
                      path: str = Query("")) -> dict:
    """One file per request. The file streams to disk in 1 MB chunks and
    is enqueued for immediate analysis as soon as the full upload landed.

    Query param `path` is the FOLDER of the file relative to the batch
    root (may be empty for root). The file name comes from the multipart
    `file` field.
    """
    s = _settings(request)
    pool = request.app.state.pool
    batch = _batch_or_404(await db.get_batch(pool, batch_id))
    if batch["status"] == "failed":
        raise HTTPException(status_code=409, detail="batch is failed")
    try:
        rel_dir = sanitize_rel_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=400,
                            detail=f"invalid path: {exc}") from exc
    file_name = Path(file.filename or "upload").name
    if not file_name or file_name in (".", ".."):
        raise HTTPException(status_code=400, detail="missing file name")
    if not file_name.lower().endswith(analysis.IMAGE_EXTENSIONS):
        raise HTTPException(status_code=400,
                            detail="only image files can be uploaded")
    rel_path = f"{rel_dir}/{file_name}" if rel_dir else file_name

    batch_root = s.spool_root / str(batch_id)
    spool_dir = batch_root / rel_dir if rel_dir else batch_root
    spool_dir.mkdir(parents=True, exist_ok=True)
    final = spool_dir / file_name
    part = spool_dir / f".part-{uuid.uuid4().hex}"
    size = 0
    try:
        with open(part, "wb") as out:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > s.max_upload_bytes:
                    raise HTTPException(status_code=413,
                                        detail="file too large")
                await asyncio.to_thread(out.write, chunk)
        # Register the scan row BEFORE the file appears, so a worker can
        # never see a half-written spool file.
        scan_id, inserted = await db.enqueue_scan(pool, batch_id, rel_path,
                                                  file_name, size)
        await asyncio.to_thread(os.replace, part, final)
    except HTTPException:
        part.unlink(missing_ok=True)
        raise
    except Exception:
        part.unlink(missing_ok=True)
        log.exception("upload failed for %s", rel_path)
        raise HTTPException(status_code=500,
                            detail="upload failed") from None
    _pipeline(request).queue.put_nowait((batch_id, scan_id, rel_path))
    return {"scan_id": scan_id, "rel_path": rel_path, "size": size,
            "inserted": inserted}


@router.post("/batches/{batch_id}/complete")
async def complete_batch(request: Request, batch_id: uuid.UUID) -> dict:
    """Client calls this after all files are sent. The batch switches to
    'processing'; when the last scan is committed it switches to
    'completed' and the report is generated automatically. If every scan
    was already committed before this call, the batch completes here and
    the report is generated right away."""
    pool = request.app.state.pool
    batch = _batch_or_404(await db.get_batch(pool, batch_id))
    if batch["status"] == "failed":
        raise HTTPException(status_code=409, detail="batch is failed")
    completed_now = await db.signal_complete(pool, batch_id)
    batch = await db.get_batch(pool, batch_id)
    if completed_now and not batch.get("report_path"):
        # all scans finished before the signal: the pipeline's per-scan
        # report trigger never fired, so generate the report here.
        try:
            await report.write_report(pool, _settings(request), batch_id)
            batch = await db.get_batch(pool, batch_id)
        except Exception:
            log.exception("report for batch %s failed", batch_id)
    return _batch_json(batch)


@router.post("/batches/{batch_id}/report")
async def regenerate_report(request: Request, batch_id: uuid.UUID) -> dict:
    pool = request.app.state.pool
    _batch_or_404(await db.get_batch(pool, batch_id))
    path = await report.write_report(pool, _settings(request), batch_id)
    return {"report_path": str(path)}


@router.get("/batches/{batch_id}/report")
async def download_report(request: Request, batch_id: uuid.UUID):
    pool = request.app.state.pool
    batch = _batch_or_404(await db.get_batch(pool, batch_id))
    if not batch.get("report_path"):
        raise HTTPException(status_code=404,
                            detail="no report yet; use POST /report")
    p = Path(batch["report_path"])
    if not p.is_file():
        raise HTTPException(status_code=410,
                            detail="report file missing on disk; "
                                   "use POST /report to regenerate")
    return FileResponse(
        p,
        media_type=("application/vnd.openxmlformats-officedocument"
                    ".spreadsheetml.sheet"),
        filename="batch_report.xlsx",
    )


@router.get("/batches/{batch_id}/export")
async def export_batch(request: Request, batch_id: uuid.UUID,
                       format: str = Query("csv"),
                       folder: str | None = Query(None)):
    """Streaming export of the full per-scan data (streamed from
    Postgres, never materialized as one in-memory object). CSV always;
    Parquet requires pyarrow. `folder` limits the export to one
    subfolder (relative, same rules as the upload path)."""
    import csv as csv_mod
    import io

    pool = request.app.state.pool
    _batch_or_404(await db.get_batch(pool, batch_id))
    if format not in ("csv", "parquet"):
        raise HTTPException(status_code=400,
                            detail="format must be 'csv' or 'parquet'")
    if folder is not None:
        try:
            folder = sanitize_rel_path(folder)
        except ValueError as exc:
            raise HTTPException(status_code=400,
                                detail=f"invalid folder: {exc}") from exc
    fname = (f"scans_{folder.replace('/', '__') if folder else 'all'}"
             f".{format}")

    if format == "csv":
        async def gen_csv():
            buf = io.StringIO()
            writer = csv_mod.writer(buf)
            writer.writerow(report.SCAN_HEADERS)
            yield buf.getvalue().encode("utf-8")
            buf.seek(0)
            buf.truncate(0)
            async for row in db.iter_export_rows(pool, batch_id,
                                                 folder=folder):
                rec = report._row_to_record(row)
                writer.writerow(["" if rec[h] is None else rec[h]
                                 for h in report.SCAN_HEADERS])
                if buf.tell() > 256 * 1024:
                    yield buf.getvalue().encode("utf-8")
                    buf.seek(0)
                    buf.truncate(0)
            tail = buf.getvalue()
            if tail:
                yield tail.encode("utf-8")
        return StreamingResponse(
            gen_csv(), media_type="text/csv",
            headers={"Content-Disposition":
                     f'attachment; filename="{fname}"'})

    async def gen_parquet():
        export_dir = (_settings(request).data_root / "reports"
                      / str(batch_id) / "exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        tmp = export_dir / f".tmp-{uuid.uuid4().hex}.parquet"
        try:
            await report.export_batch_parquet(pool, batch_id, tmp,
                                              folder=folder)
            with open(tmp, "rb") as fh:
                while chunk := fh.read(CHUNK_SIZE):
                    yield chunk
        finally:
            tmp.unlink(missing_ok=True)
    return StreamingResponse(
        gen_parquet(), media_type="application/vnd.apache.parquet",
        headers={"Content-Disposition":
                 f'attachment; filename="{fname}"'})


@router.delete("/batches/{batch_id}")
async def delete_batch(request: Request, batch_id: uuid.UUID) -> dict:
    s = _settings(request)
    pool = request.app.state.pool
    batch = _batch_or_404(await db.get_batch(pool, batch_id))
    if batch["status"] in ("uploading", "processing"):
        raise HTTPException(status_code=409,
                            detail="batch is still uploading/processing")
    await db.delete_batch(pool, batch_id)
    # remove on-disk traces: spool (normally already empty), report,
    # and any retained intermediate artifacts
    shutil.rmtree(s.spool_root / str(batch_id), ignore_errors=True)
    shutil.rmtree(s.data_root / "batches" / str(batch_id), ignore_errors=True)
    shutil.rmtree(s.data_root / "reports" / str(batch_id), ignore_errors=True)
    return {"deleted": str(batch_id)}
