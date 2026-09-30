"""
FastAPI application entrypoint. Route handlers only — no business logic
lives here; each endpoint delegates to a service module.
"""

import logging
import os
import traceback
from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session


from src.api.security import verify_api_key
from src.models.database import get_db
from src.services.address_verification_service import AddressVerificationService
from src.services.affidavit_service import AffidavitService
from src.services.kyc_service import KYCService

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 25 * 1024 * 1024      # 25 MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024     # 100 MB
MAX_ADDRESS_SIZE = 25 * 1024 * 1024    # 25 MB — same ceiling as the ID document

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
}

# Proof of address can be a scanned image or a PDF bank/utility statement.
ALLOWED_ADDRESS_TYPES = ALLOWED_IMAGE_TYPES | {"application/pdf"}

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


# ---------------------------------------------------------------------
# Global safety net: any exception that isn't already caught by a route
# (e.g. a raw TypeError/ValueError thrown while building a file path,
# before we ever reach a route's own try/except) previously surfaced as
# a bare 500 with no message and no logged traceback. This handler makes
# every unhandled error visible in the console and in the response body,
# so failures like this are diagnosable from the first occurrence instead
# of requiring a repro-and-guess cycle.
# ---------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {type(exc).__name__}: {exc}"},
    )


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
        raise HTTPException(400, "ID document exceeds 25 MB.")

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


@app.post("/api/address/verify")
async def verify_address(
    
    customer_id: str = Form(...),
    full_name: str = Form(...),
    id_number: str = Form(...),
    id_document: UploadFile = File(...),
    liveness_video: UploadFile = File(...),
    proof_of_address: UploadFile = File(...),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
):
    

    

    

    
    # Validate content types
    if id_document.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, "Unsupported ID document format.")

    if liveness_video.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(400, "Unsupported video format.")

    if proof_of_address.content_type not in ALLOWED_ADDRESS_TYPES:
        raise HTTPException(400, "Unsupported proof of address format.")

    # Read files
    id_bytes = await id_document.read()
    video_bytes = await liveness_video.read()
    address_bytes = await proof_of_address.read()

    # Validate file sizes
    if len(id_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(400, "ID document exceeds 25 MB.")

    if len(video_bytes) > MAX_VIDEO_SIZE:
        raise HTTPException(400, "Liveness video exceeds 100 MB.")

    if len(address_bytes) > MAX_ADDRESS_SIZE:
        raise HTTPException(400, "Proof of address exceeds 25 MB.")

    id_extension = os.path.splitext(id_document.filename or "")[1].lower()

    if id_extension not in {".jpg", ".jpeg", ".png"}:
        id_extension = ".jpg"

    id_doc_path = os.path.join(
        UPLOAD_DIR,
        f"{customer_id}_addr_id{id_extension}",
    )

    liveness_path = os.path.join(
        UPLOAD_DIR,
        f"{customer_id}_addr_liveness.mp4",
    )

    # Guarded against proof_of_address.filename being None/empty — this
    # was the source of the unhandled TypeError that surfaced as a bare
    # 500 (os.path.splitext(None) raises before we ever reach the
    # run_verification try/except below).
    address_extension = (
        os.path.splitext(proof_of_address.filename or "")[1]
        or ".pdf"
    )

    address_path = os.path.join(
        UPLOAD_DIR,
        f"{customer_id}_proof_of_address{address_extension}",
    )

    with open(id_doc_path, "wb") as f:
        f.write(id_bytes)

    with open(liveness_path, "wb") as f:
        f.write(video_bytes)

    with open(address_path, "wb") as f:
        f.write(address_bytes)

   
    try:
        result = await run_in_threadpool(
           
            AddressVerificationService.run_verification,
            customer_id,
            full_name,
            id_number,
            id_doc_path,
            liveness_path,
            address_path,
            db,
        )

       

        return result

    except Exception as exc:
        
        traceback.print_exc()
        

        raise HTTPException(
            status_code=500,
            detail=f"Address verification failed: {type(exc).__name__}: {exc}",
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
    extension = os.path.splitext(affidavit.filename or "")[1] or ".pdf"
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