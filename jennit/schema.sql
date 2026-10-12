CREATE TABLE IF NOT EXISTS batches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    client_ref TEXT,
    status TEXT NOT NULL DEFAULT 'uploading'
        CHECK (status IN ('uploading', 'processing', 'completed', 'failed')),
    complete_signaled BOOLEAN NOT NULL DEFAULT FALSE,
    uploaded_count BIGINT NOT NULL DEFAULT 0,
    processed_count BIGINT NOT NULL DEFAULT 0,
    failed_count BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    report_path TEXT
);

CREATE TABLE IF NOT EXISTS scans (
    id BIGSERIAL PRIMARY KEY,
    batch_id UUID NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    folder_path TEXT NOT NULL DEFAULT '',
    file_name TEXT NOT NULL,
    file_size_bytes BIGINT,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    is_valid BOOLEAN,
    is_overexposed BOOLEAN,
    compression_ratio DOUBLE PRECISION,
    jpeg_quality INTEGER,
    black_pixels BIGINT,
    white_pixels BIGINT,
    angle DOUBLE PRECISION,
    md5 TEXT,
    sha512 TEXT,
    dpi JSONB,
    exif JSONB,
    color_profile JSONB,
    color_histograms JSONB,
    errors TEXT,
    UNIQUE (batch_id, rel_path)
);

CREATE INDEX IF NOT EXISTS scans_batch_status_idx ON scans (batch_id, status);

CREATE INDEX IF NOT EXISTS scans_batch_folder_idx ON scans (batch_id, folder_path);
