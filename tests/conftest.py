import os
import shutil

import pytest

# must be set before the app/settings are imported
os.environ.setdefault("JENNIT_DB_DSN",
                      "postgresql://jennit:jennit@127.0.0.1:5432/jennit_test")
os.environ.setdefault("JENNIT_SPOOL_ROOT", "/tmp/jennit_test/spool")
os.environ.setdefault("JENNIT_DATA_ROOT", "/tmp/jennit_test/data")
os.environ.setdefault("JENNIT_THREADS", "4")


@pytest.fixture()
def app_client():
    """TestClient with the app lifespan (DB init + pipeline) active."""
    from fastapi.testclient import TestClient

    from jennit.api.app import create_app

    for d in ("/tmp/jennit_test/spool", "/tmp/jennit_test/data"):
        shutil.rmtree(d, ignore_errors=True)
    app = create_app()
    with TestClient(app) as client:
        yield client
