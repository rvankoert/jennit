"""Integration test: full batch flow through the API (needs compose Postgres)."""
import time

import pytest

pytestmark = pytest.mark.usefixtures("app_client")


def _make_image(path):
    import cv2
    import numpy as np

    img = np.full((100, 100, 3), 200, np.uint8)
    img[0:20, 0:20] = 30
    cv2.imwrite(str(path), img)


def _wait_completed(client, bid, timeout=60):
    deadline = time.time() + timeout
    batch = None
    while time.time() < deadline:
        batch = client.get(f"/api/v1/batches/{bid}").json()
        if batch["status"] == "completed":
            return batch
        time.sleep(0.5)
    raise AssertionError(f"batch did not complete: {batch}")


def test_full_batch_flow(app_client, tmp_path):
    f1, f2 = tmp_path / "p1.jpg", tmp_path / "p2.jpg"
    _make_image(f1)
    _make_image(f2)

    r = app_client.post("/api/v1/batches", data={"name": "flow-test"})
    assert r.status_code == 200, r.text
    bid = r.json()["id"]

    with open(f1, "rb") as fh:
        r = app_client.post(f"/api/v1/batches/{bid}/files",
                            files={"file": ("p1.jpg", fh)})
    assert r.status_code == 200, r.text
    with open(f2, "rb") as fh:
        r = app_client.post(f"/api/v1/batches/{bid}/files",
                            params={"path": "sub"},
                            files={"file": ("p2.jpg", fh)})
    assert r.status_code == 200, r.text

    r = app_client.post(f"/api/v1/batches/{bid}/complete")
    assert r.status_code == 200, r.text
    batch = _wait_completed(app_client, bid)
    assert batch["uploaded_count"] == 2 and batch["processed_count"] == 2

    rows = app_client.get(f"/api/v1/batches/{bid}/scans",
                          params={"limit": 50}).json()
    assert len(rows) == 2
    assert {row["folder_path"] for row in rows} == {"", "sub"}

    csv_resp = app_client.get(f"/api/v1/batches/{bid}/export",
                              params={"format": "csv"})
    assert csv_resp.status_code == 200
    lines = csv_resp.text.strip().splitlines()
    assert len(lines) == 3 and lines[0].startswith("Rel Path")

    rep = app_client.get(f"/api/v1/batches/{bid}/report")
    assert rep.status_code == 200
    assert rep.content[:2] == b"PK"  # xlsx is a zip file

    r = app_client.delete(f"/api/v1/batches/{bid}")
    assert r.status_code == 200
    assert app_client.get(f"/api/v1/batches/{bid}").status_code == 404
