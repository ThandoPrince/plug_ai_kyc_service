"""
Orchestrates document + liveness (biometric) verification for the AI
microservice. Distinct from the Django-side KYCService, which lives in
the Plug backend and calls this service over HTTP.
"""

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from src.api.kyc_processor import KYCProcessor
from src.models.database import KYCRecord

logger = logging.getLogger(__name__)


class KYCService:
    @staticmethod
    def run_kyc(
        customer_id: str,
        full_name: str,
        doc_path: str,
        video_path: str,
        db: Session,
    ) -> dict:
        processor = KYCProcessor()

        # 1. Document authenticity (resolution / color / size)
        doc_ok, doc_details = processor.verify_document_authenticity(doc_path)
        if not doc_ok:
            return KYCService._finalize(
                db, customer_id, full_name,
                success=False,
                verification_status="rejected",
                risk_level="high",
                face_match_score=0.0,
                message="Document failed quality checks.",
                reason="; ".join(doc_details.get("failed_checks", [])) or "Invalid document.",
                ai_response={"document_check": doc_details},
                document_path=doc_path,
                selfie_path=video_path,
            )

        # 2. Face match: ID document vs. liveness video
        match_ok, score, match_details = processor.verify_face_match(doc_path, video_path)

        ai_response = {
            "document_check": doc_details,
            "face_match": match_details,
        }

        if not match_ok:
            return KYCService._finalize(
                db, customer_id, full_name,
                success=True,  # the check itself ran fine — it just didn't pass
                verification_status="rejected",
                risk_level="high",
                face_match_score=score,
                message=match_details.get("message", "Face verification failed."),
                reason=match_details.get("details", ""),
                ai_response=ai_response,
                document_path=doc_path,
                selfie_path=video_path,
            )

        risk_level = "low" if score >= 80 else "medium"

        return KYCService._finalize(
            db, customer_id, full_name,
            success=True,
            verification_status="verified",
            risk_level=risk_level,
            face_match_score=score,
            message="Identity verification successful.",
            reason="",
            ai_response=ai_response,
            document_path=doc_path,
            selfie_path=video_path,
        )

    @staticmethod
    def _finalize(
        db: Session,
        customer_id: str,
        full_name: str,
        *,
        success: bool,
        verification_status: str,
        risk_level: str,
        face_match_score: float,
        message: str,
        reason: str,
        ai_response: dict,
        document_path: str = None,
        selfie_path: str = None,
    ) -> dict:
        try:
            record = KYCRecord(
                customer_id=customer_id,
                full_name=full_name,
                verification_status=verification_status,
                face_match_score=face_match_score,
                risk_level=risk_level,
                ai_response=ai_response,
                document_path=document_path,
                selfie_path=selfie_path,
                created_at=datetime.utcnow(),
            )
            db.add(record)
            db.commit()
        except Exception:
            logger.exception("Failed to persist KYC record for customer %s", customer_id)
            db.rollback()

        return {
            "success": success,
            "verification_status": verification_status,
            "face_match_score": face_match_score,
            "risk_level": risk_level,
            "message": message,
            "reason": reason,
        }