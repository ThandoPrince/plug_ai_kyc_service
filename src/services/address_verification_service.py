"""
Orchestrates address verification for the AI microservice.

Reuses KYCProcessor's existing biometric and ID-document checks, then
adds one more gate: the submitted proof of address must reference the
same customer (by name + ID number) that the ID document and liveness
video already confirmed.

Pipeline, each step a hard gate (short-circuits on failure):
  1. ID document quality (resolution/color/size)
  2. Liveness match — ID document face vs liveness video
  3. ID document identity check — does the ID actually show this
     customer's name + ID number
  4. Proof of address check — does the address document reference
     that same name + ID number
"""

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from src.api.kyc_processor import KYCProcessor
from src.models.database import AddressVerificationRecord

logger = logging.getLogger(__name__)


class AddressVerificationService:
    @staticmethod
    def run_verification(
        customer_id: str,
        full_name: str,
        id_number: str,
        id_doc_path: str,
        liveness_path: str,
        address_doc_path: str,
        db: Session,
    ) -> dict:
        processor = KYCProcessor()

        # 1. ID document quality gate
        doc_ok, doc_details = processor.verify_document_authenticity(id_doc_path)
        if not doc_ok:
            return AddressVerificationService._finalize(
                db, customer_id, full_name,
                success=False,
                verification_status="rejected",
                risk_level="high",
                face_match_score=0.0,
                message="ID document failed quality checks.",
                reason="; ".join(doc_details.get("failed_checks", [])) or "Invalid ID document.",
                ai_response={"document_check": doc_details},
                id_document_path=id_doc_path,
                selfie_path=liveness_path,
                address_document_path=address_doc_path,
            )

        # 2. Liveness gate — ID document face vs liveness video
        match_ok, score, match_details = processor.verify_face_match(id_doc_path, liveness_path)

        ai_response = {
            "document_check": doc_details,
            "face_match": match_details,
        }

        if not match_ok:
            return AddressVerificationService._finalize(
                db, customer_id, full_name,
                success=True,  # the check itself ran fine — it just didn't pass
                verification_status="rejected",
                risk_level="high",
                face_match_score=score,
                message=match_details.get("message", "Face verification failed."),
                reason=match_details.get("details", ""),
                ai_response=ai_response,
                id_document_path=id_doc_path,
                selfie_path=liveness_path,
                address_document_path=address_doc_path,
            )

        # 3. ID document identity gate — confirms the ID itself belongs
        # to this customer before we trust it as the anchor for step 4.
        id_result = processor.verify_id_document(id_doc_path, full_name, id_number)
        ai_response["id_document_check"] = id_result

        if id_result["verification_status"] != "verified":
            return AddressVerificationService._finalize(
                db, customer_id, full_name,
                success=True,
                verification_status="rejected",
                risk_level="high",
                face_match_score=score,
                message=id_result.get("message", "ID document details did not match."),
                reason="ID document identity check failed.",
                ai_response=ai_response,
                id_document_path=id_doc_path,
                selfie_path=liveness_path,
                address_document_path=address_doc_path,
            )

        # 4. Proof of address must reference the same customer
        address_result = processor.verify_proof_of_address(address_doc_path, full_name, id_number)
        ai_response["address_check"] = address_result

        if address_result["verification_status"] != "verified":
            return AddressVerificationService._finalize(
                db, customer_id, full_name,
                success=True,
                verification_status="rejected",
                risk_level="medium",
                face_match_score=score,
                message=address_result.get(
                    "message", "Proof of address did not match customer details."
                ),
                reason="Proof of address identity check failed.",
                ai_response=ai_response,
                id_document_path=id_doc_path,
                selfie_path=liveness_path,
                address_document_path=address_doc_path,
            )

        risk_level = "low" if score >= 80 else "medium"

        return AddressVerificationService._finalize(
            db, customer_id, full_name,
            success=True,
            verification_status="verified",
            risk_level=risk_level,
            face_match_score=score,
            message="Address verification successful.",
            reason="",
            ai_response=ai_response,
            id_document_path=id_doc_path,
            selfie_path=liveness_path,
            address_document_path=address_doc_path,
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
        id_document_path: str = None,
        selfie_path: str = None,
        address_document_path: str = None,
    ) -> dict:
        try:
            record = AddressVerificationRecord(
                customer_id=customer_id,
                full_name=full_name,
                verification_status=verification_status,
                face_match_score=face_match_score,
                risk_level=risk_level,
                ai_response=ai_response,
                id_document_path=id_document_path,
                selfie_path=selfie_path,
                address_document_path=address_document_path,
                created_at=datetime.utcnow(),
            )
            db.add(record)
            db.commit()
        except Exception:
            logger.exception(
                "Failed to persist address verification record for customer %s", customer_id
            )
            db.rollback()

        return {
            "success": success,
            "verification_status": verification_status,
            "face_match_score": face_match_score,
            "risk_level": risk_level,
            "message": message,
            "reason": reason,
        }