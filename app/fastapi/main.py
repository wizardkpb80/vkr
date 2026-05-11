from fastapi import FastAPI
from api.api_route import router
from api.app.logging import logger
from api.app.utils import init_smb_session

app = FastAPI(title="1C WebService Proxy", description="Proxy для вызова функций 1С")
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    init_smb_session()  # now reads creds from config
    logger.info("Starting 1C API proxy")

@app.get("/health")
async def health():
    return {"status": "ok"}