import logging

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(levelname)-9s %(name)s - %(message)s",
    )
    # Every request to Ollama would otherwise be logged.
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
