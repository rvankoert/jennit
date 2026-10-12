"""FastAPI application factory + lifespan (DB init, pool, pipeline).

The web UI is served by the separate front-end container (see frontend/);
this app only exposes the REST API, /health and the interactive docs.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .. import db
from ..config import Settings
from ..pipeline import Pipeline
from .routes_batches import router

log = logging.getLogger("jennit.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        log.info("initializing database %s",
                 db.database_name(settings.db_dsn))
        await db.ensure_database(settings.db_dsn)
        pool = await db.create_pool(settings.db_dsn)
        await db.init_schema(pool)
        app.state.pool = pool
        app.state.pipeline = Pipeline(pool, settings)
        await app.state.pipeline.start()
        await app.state.pipeline.requeue_stale()
        await app.state.pipeline.cleanup_orphans()
        log.info("jennit API ready (threads=%d)", settings.threads)
        try:
            yield
        finally:
            await app.state.pipeline.stop()
            await pool.close()

    app = FastAPI(title="jennit", version="0.2.0", lifespan=lifespan)
    # available before the lifespan runs (auth only reads this)
    app.state.settings = settings
    app.include_router(router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s %s",
                      request.method, request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": "internal server error"})

    return app
