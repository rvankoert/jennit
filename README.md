# jennit

Jennit finds flaws in scanned documents: per-image measurements (validity,
overexposure, black/white pixels, EXIF, DPI, JPEG quality, compression
ratio, color profile/histograms, MD5/SHA-512), annotation of affected
images, and report generation.

It runs as a CLI, as a batch upload API (FastAPI + PostgreSQL), or in the
browser via the web UI (a separate front-end container).

## Installation

Requires Python 3.10+ and ImageMagick (`identify`), e.g. on Ubuntu:

```bash
sudo apt install imagemagick
```

```bash
git clone git@github.com:rvankoert/jennit.git
cd jennit
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"      # add [parquet] for the parquet export
```

## CLI

```bash
.venv/bin/jennit --input_dir ./input_images \
  --output_excel ./results/metadata.xlsx \
  --output_dir ./results/annotated \
  --threads 8 --pixels_threshold 10
```

`python flawfinder.py ...` still works (thin shim over the same code).

| Argument             | Description                                                                                      | Required | Default |
|----------------------|--------------------------------------------------------------------------------------------------|----------|---------|
| `--input_dir`        | Directory containing input images.                                                               | Yes      |         |
| `--output_excel`     | Excel file where extracted metadata is saved.                                                    | Yes      |         |
| `--output_dir`       | Directory for annotated images and analysis results.                                             | Yes      |         |
| `--threads`          | Parallel processing threads.                                                                     | No       | 16      |
| `--pixels_threshold` | Minimum black/white pixel count for annotation.                                                  | No       | 5       |
| `--depth`            | Maximum subdirectory depth to search.                                                            | No       | 3       |

## Batch server

A **separate PostgreSQL database** stores per-scan results (auto-created
on first start; its name comes from the DSN).

### Docker (recommended)

The whole application runs as three containers — front-end (static web
UI + `/api/*` proxy), back-end (FastAPI + analysis pipeline), Postgres:

```bash
docker compose up --build
```

* Web UI: http://localhost:8080/ — drag & drop a folder (or select
  files/folder), progress updates live.
* Interactive API docs: http://localhost:8000/docs — the API is also
  reachable through the front-end at http://localhost:8080/api.

| Service    | Image                             | Host port | Purpose |
|------------|-----------------------------------|-----------|---------|
| `frontend` | nginx (built from `frontend/`)    | 8080      | static web UI; proxies `/api/*` to the back-end |
| `backend`  | python (built from the repo root) | 8000      | FastAPI batch API + analysis pipeline (OpenCV, ImageMagick) |
| `postgres` | postgres:16                       | 5432      | dedicated `jennit` database |

Persistent data lives in named volumes: `jennit_pgdata` (database),
`jennit_spool` (upload spool; files are deleted after processing),
`jennit_data` (reports, exports, retained artifacts).

To require a bearer token, set `JENNIT_API_TOKEN` in your shell or in a
`.env` file next to `docker-compose.yml` before `docker compose up`; the
other `JENNIT_*` variables from `.env.example` are passed through the same
way (the database DSN already points at the compose network).

### Without Docker

```bash
docker compose up -d postgres       # database only
JENNIT_DB_DSN=postgresql://jennit:jennit@127.0.0.1:5432/jennit \
.venv/bin/jennit-server             # API on http://localhost:8000
```

The web UI is served by the `frontend` container; the API alone still
works for `jennit-upload` and direct API calls.

### Environment variables

See `.env.example`.

| Variable                  | Default | Meaning |
|---------------------------|---------|---------|
| `JENNIT_DB_DSN`           | `postgresql://jennit:jennit@127.0.0.1:5432/jennit` | DSN of the dedicated database (auto-created) |
| `JENNIT_SPOOL_ROOT`       | `./spool` | temp dir for uploads (files deleted after processing) |
| `JENNIT_DATA_ROOT`        | `./data` | reports, exports, retained artifacts |
| `JENNIT_THREADS`          | `16` | analysis worker threads |
| `JENNIT_PIXELS_THRESHOLD` | `5` | annotation threshold (same as CLI) |
| `JENNIT_MAX_UPLOAD_BYTES` | `2147483648` | per-file upload limit |
| `JENNIT_KEEP_INTERMEDIATES` | `false` | retain intermediate artifacts under `data/batches/<id>/artifacts/` |
| `JENNIT_API_TOKEN`        | *(empty)* | bearer token; empty = open (internal) access |
| `JENNIT_HOST` / `JENNIT_PORT` | `0.0.0.0` / `8000` | bind address |

### Uploading a batch from the command line

```bash
.venv/bin/jennit-upload --url http://localhost:8000 \
  --token <token> ./scans
```

The folder structure is preserved; each file is analyzed as soon as it
has fully arrived, so upload and processing overlap.

### API overview

| Method & path | Purpose |
|---------------|---------|
| `POST /api/v1/batches` | create a batch (form field `name`, optional `client_ref`) |
| `GET /api/v1/batches` | list batches |
| `GET /api/v1/batches/{id}` | status + per-folder progress |
| `POST /api/v1/batches/{id}/files` | upload one file (multipart `file`; query `path` = folder) |
| `POST /api/v1/batches/{id}/complete` | all files sent; batch completes when the last scan is committed |
| `GET /api/v1/batches/{id}/scans` | per-scan rows (paginated, filter by `status`/`folder`) |
| `POST /api/v1/batches/{id}/report` | (re)generate the Excel report |
| `GET /api/v1/batches/{id}/report` | download the report (.xlsx) |
| `GET /api/v1/batches/{id}/export?format=csv\|parquet&folder=…` | streaming full per-scan export |
| `DELETE /api/v1/batches/{id}` | delete batch rows, report, exports, artifacts |

### Reports and retention

* On completion an Excel workbook is generated: a Summary sheet (batch
  info + one aggregate row per folder) plus one detail sheet per folder.
  Full per-scan data at 1,000,000-scan scale is exported as CSV/Parquet
  from Postgres, because Excel's row limit is 1,048,576.
* After a scan's row is committed, its spool file and intermediate
  artifacts are deleted; set `JENNIT_KEEP_INTERMEDIATES=true` to retain
  them instead.
* The report is kept until the batch is deleted and can be regenerated
  with `POST /api/v1/batches/{id}/report`.

## Tests

```bash
docker compose up -d postgres   # Postgres for the integration test
.venv/bin/python -m pytest tests/ -v
```
