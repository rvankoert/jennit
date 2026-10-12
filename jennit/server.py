"""jennit-server — run the jennit API with uvicorn (single worker)."""
import uvicorn

from .api.app import create_app
from .config import Settings


def run() -> None:
    settings = Settings.from_env()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
