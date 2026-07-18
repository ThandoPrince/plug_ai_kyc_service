"""
FastAPI application entrypoint. Route handlers only — no business logic
lives here; each endpoint delegates to a service module.
"""

import os

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session


from src.api.security import verify_api_key
from src.models.database import get_db
from src.services.affidavit_service import AffidavitService
from src.services.kyc_service import KYCService

MAX_IMAGE_SIZE = 10 * 1024 * 1024      # 10 MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024     # 100 MB

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
}

app = FastAPI(
    title="AI KYC System",
    docs_url=None if os.getenv("ENVIRONMENT") == "production" else "/docs",
    redoc_url=None if os.getenv("ENVIRONMENT") == "production" else "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")




@app.post("/api/kyc/initiate")
async def initiate_kyc(
    customer_id: str = Form(...),
    full_name: str = Form(...),
    id_document: UploadFile = File(...),
    liveness_video: UploadFile = File(...),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
):
    # Validate content types
    if id_document.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, "Unsupported ID document format.")

    if liveness_video.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(400, "Unsupported video format.")

    # Read files
    id_bytes = await id_document.read()
    video_bytes = await liveness_video.read()

    # Validate file sizes
    if len(id_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(400, "ID document exceeds 10 MB.")

    if len(video_bytes) > MAX_VIDEO_SIZE:
        raise HTTPException(400, "Liveness video exceeds 100 MB.")

    doc_path = os.path.join(UPLOAD_DIR, f"{customer_id}_doc.jpg")
    video_path = os.path.join(UPLOAD_DIR, f"{customer_id}_liveness.mp4")

    with open(doc_path, "wb") as f:
        f.write(id_bytes)

    with open(video_path, "wb") as f:
        f.write(video_bytes)

    return await run_in_threadpool(
        KYCService.run_kyc,
        customer_id,
        full_name,
        doc_path,
        video_path,
        db,
    )


@app.post("/api/affidavit/verify")
async def verify_affidavit(
    provider_id: str = Form(...),
    full_name: str = Form(...),
    id_number: str = Form(...),
    service_name: str = Form(...),
    _: None = Depends(verify_api_key),
    affidavit: UploadFile = File(...),
):
    # Keep original extension (affidavit can be a PDF or an image).
    extension = os.path.splitext(affidavit.filename)[1] or ".pdf"
    affidavit_path = os.path.join(UPLOAD_DIR, f"{provider_id}_affidavit{extension}")

    with open(affidavit_path, "wb") as f:
        f.write(await affidavit.read())

    return await run_in_threadpool(
        AffidavitService.verify_affidavit,
        affidavit_path,
        full_name,
        id_number,
        service_name,
    )