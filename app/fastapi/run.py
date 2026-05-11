import warnings
import uvicorn
from api.app.config import LOG_LEVEL
from api.app.logging import logger


def main():
    warnings.simplefilter(action='ignore', category=Warning)
    logger.info("Starting 1C WebService API proxy")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level=LOG_LEVEL.lower(),
        reload=True
    )


if __name__ == "__main__":
    main()