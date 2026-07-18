"""
FastAPI application entrypoint. Route handlers only — no business logic
lives here; each endpoint delegates to a service module.
"""

import os

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session


from src.api.security import verify_api_key
from src.models.database import get_db
from src.services.affidavit_service import AffidavitService
from src.services.kyc_service import KYCService

app = FastAPI(title="AI KYC System")

app.add_middleware(
    CORSMiddleware,
    # TODO: restrict this to the Plug backend's actual origin before
    # production — this service receives ID documents and biometric
    # video, wildcard CORS is fine for local dev only.
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    doc_path = os.path.join(UPLOAD_DIR, f"{customer_id}_doc.jpg")
    video_path = os.path.join(UPLOAD_DIR, f"{customer_id}_liveness.mp4")

    with open(doc_path, "wb") as f:
        f.write(await id_document.read())

    with open(video_path, "wb") as f:
        f.write(await liveness_video.read())

    return await run_in_threadpool(
        KYCService.run_kyc, customer_id, full_name, doc_path, video_path, db
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