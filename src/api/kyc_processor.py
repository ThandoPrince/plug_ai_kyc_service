"""
KYCProcessor: core identity-verification logic.

Two distinct responsibilities live here because they both center on
"does this person match/attest to what they claim":

1. Biometric face matching — ID document photo vs. liveness video.
2. Affidavit verification — OCR + computer-vision checks + template
   matching against Plug's required declarations.

Document loading/OCR mechanics live in DocumentProcessor; affidavit-
specific image-quality/marking checks live in AffidavitVision. This
class orchestrates them and applies the business rules (thresholds,
matching logic, confidence scoring).
"""

from datetime import date
import os
from typing import Dict

import cv2
import numpy as np
from numpy.linalg import norm
import mediapipe as mp
from insightface.app import FaceAnalysis

from src.api.config.affidavit_template import REQUIRED_DECLARATIONS

from src.api.utils.date_utils import MAX_AFFIDAVIT_AGE_DAYS, is_date_recent, most_likely_document_date
from src.api.utils.sa_id_utils import extract_id_number, parse_sa_id_number
from src.api.vision.affidavit_vision import AffidavitVision
from src.api.utils.ocr_utils import detect_saps_stamp_text, find_identifier, fuzzy_name_match, match_declarations
from src.services.document_processer import DocumentProcessor

mp_face_detection = mp.solutions.face_detection

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)
face_app.prepare(ctx_id=-1, det_size=(640, 640))

FACE_MATCH_THRESHOLD = 0.65
AFFIDAVIT_CONFIDENCE_APPROVE_THRESHOLD = 0.85



class KYCProcessor:
    def __init__(self):
        self.document_processor = DocumentProcessor()

    # =========================================================
    # Biometric face matching (ID document vs liveness video)
    # =========================================================

    def verify_face_match(self, id_doc_path: str, liveness_video: str) -> tuple:
        """Verify if faces are detected in both images and match."""
        try:
            id_image = cv2.imread(id_doc_path)

            if id_image is None:
                return False, 0.0, {
                    "error_type": "upload_error",
                    "message": "Failed to load ID image.",
                }

            selfie_image = self.extract_best_face_frame(liveness_video)

            if selfie_image is None:
                return False, 0.0, {
                    "error_type": "upload_error",
                    "message": "Unable to extract a usable frame.",
                }

            id_faces = face_app.get(id_image)
            video_faces = face_app.get(selfie_image)

            # Validate face counts BEFORE indexing [0] anywhere — the
            # original code logged faces[0] before this check, which
            # raised IndexError outright on any zero-face frame instead
            # of returning the intended "no face detected" response.
            if len(id_faces) != 1:
                return False, 0.0, {
                    "error_type": "face_verification_error",
                    "message": (
                        "No face detected in ID." if len(id_faces) == 0
                        else "Multiple faces detected in ID."
                    ),
                }

            if len(video_faces) != 1:
                return False, 0.0, {
                    "error_type": "face_verification_error",
                    "message": (
                        "No face detected in liveness video." if len(video_faces) == 0
                        else "Multiple faces detected in liveness video."
                    ),
                }

            id_embedding = id_faces[0].embedding
            video_embedding = video_faces[0].embedding

            similarity = np.dot(id_embedding, video_embedding) / (
                norm(id_embedding) * norm(video_embedding)
            )
            similarity_score = float(similarity * 100)

            if similarity >= FACE_MATCH_THRESHOLD:
                return True, similarity_score, {
                    "error_type": None,
                    "message": "Face verification successful",
                    "details": "Your face has been successfully verified against your ID document.",
                }

            return False, similarity_score, {
                "error_type": "face_verification_error",
                "message": "Face verification failed",
                "details": "The face in your liveness video does not match the ID document.",
            }

        except ValueError as ve:
            return False, 0.0, {
                "error_type": "face_verification_error",
                "message": "Face verification failed",
                "details": str(ve),
            }
        except Exception:
            return False, 0.0, {
                "error_type": "system_error",
                "message": "System error during face verification",
                "details": "An unexpected error occurred during face verification. Please try again later.",
            }

    def extract_best_face_frame(self, video_path: str):
        """Returns the sharpest frame that contains exactly one face."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        detector = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.6,
        )

        best_frame = None
        best_score = -1
        frame_number = 0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_number += 1
                if frame_number % 5 != 0:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)

                if not results.detections or len(results.detections) != 1:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

                if sharpness > best_score:
                    best_score = sharpness
                    best_frame = frame.copy()
        finally:
            cap.release()
            detector.close()

        return best_frame

    # =========================================================
    # Generic document authenticity (resolution/color/size)
    # =========================================================

    @staticmethod
    def verify_document_authenticity(document_path: str):
        try:
            if not os.path.exists(document_path):
                raise FileNotFoundError("Document file not found")

            image = cv2.imread(document_path)
            if image is None:
                raise ValueError("Could not load document image")

            checks = {
                "resolution_check": {
                    "passed": image.shape[0] >= 1000 and image.shape[1] >= 1000,
                    "message": "Image resolution must be at least 1000x1000 pixels",
                },
                "color_check": {
                    "passed": len(image.shape) == 3,
                    "message": "Image must be in color format",
                },
                "size_check": {
                    "passed": os.path.getsize(document_path) < 10 * 1024 * 1024,
                    "message": "File size must be less than 10MB",
                },
            }

            failed_checks = [v["message"] for v in checks.values() if not v["passed"]]
            is_valid = len(failed_checks) == 0

            return is_valid, {
                "success": is_valid,
                "checks": checks,
                "failed_checks": failed_checks,
            }

        except Exception as e:
            return False, {
                "success": False,
                "error_type": "document_rejected",
                "reason": str(e),
                "checks": [],
            }

    # =========================================================
    # Affidavit verification
    # =========================================================



    def verify_affidavit(
            self,
            document_path: str,
            full_name: str,
            id_number: str,
            service_name: str,
            upload_date: date = None,
        ) -> Dict:
            """
            Full affidavit verification pipeline: OCR -> template matching
            -> image-quality/marking checks -> stamp/date validity gates ->
            name/ID/service matching, rolled up into a single confidence
            score and verification status.

            upload_date defaults to today if not supplied. The commissioner's
            stamp/signing date on the affidavit must fall within
            MAX_AFFIDAVIT_AGE_DAYS of upload_date, and the stamp must contain
            SAPS wording (English or Afrikaans) — both are hard gates,
            independent of the weighted confidence score, since neither is
            something a strong score elsewhere should be able to compensate for.
            """
            upload_date = upload_date or date.today()

            response = {
                "success": False,
                "verification_status": "rejected",
                "confidence_score": 0.0,
                "ocr": {},
                "vision": {},
                "template": {},
                "matches": {},
                "stamp_text": {},
                "document_date": None,
                "message": "",
                "error_type": None,
            }

            # 1. OCR the document
            ocr_result = self.document_processor.process(document_path)
            response["ocr"] = {
                "success": ocr_result["success"],
                "document_type": ocr_result["document_type"],
                "page_count": ocr_result["page_count"],
                "error": ocr_result["error"],
            }

            if not ocr_result["success"]:
                response["message"] = ocr_result["error"] or "Could not read the affidavit document."
                response["error_type"] = "ocr_error"
                return response

            raw_text = ocr_result["raw_text"]

            # 2. Vision checks on the primary page
            try:
                primary_page = self.document_processor.get_primary_page_image(ocr_result)
                vision = AffidavitVision(primary_page)
                vision_result = vision.analyse()
            except ValueError as e:
                response["message"] = str(e)
                response["error_type"] = "vision_error"
                return response

            response["vision"] = vision_result

            # 3. Stamp text gate — geometric stamp detection (vision_result)
            # only tells us "a circle-ish shape exists"; this confirms it's
            # actually worded as a SAPS stamp.
            stamp_text_result = detect_saps_stamp_text(raw_text)
            response["stamp_text"] = stamp_text_result

            if not stamp_text_result["detected"]:
                response["message"] = (
                    "Could not find South African Police Service / SAPS "
                    "wording on the commissioner's stamp."
                )
                response["error_type"] = "stamp_text_error"
                return response

            # 4. Date recency gate
            document_date = most_likely_document_date(raw_text, upload_date=upload_date)
            response["document_date"] = document_date.isoformat() if document_date else None

            if document_date is None:
                response["message"] = "Could not locate a signing/stamp date on the affidavit."
                response["error_type"] = "date_extraction_error"
                return response

            if not is_date_recent(document_date, upload_date=upload_date):
                response["message"] = (
                    f"Affidavit is dated {document_date.isoformat()}, which is older than "
                    f"{MAX_AFFIDAVIT_AGE_DAYS} days from the upload date "
                    f"({upload_date.isoformat()})."
                )
                response["error_type"] = "affidavit_expired"
                return response

            # 5. Template / declaration matching
            declaration_results = match_declarations(raw_text, REQUIRED_DECLARATIONS)
            response["template"] = declaration_results

            # 6. Name / ID / service matching
            matches = {
                "full_name": find_identifier(raw_text, full_name),
                "id_number": find_identifier(raw_text, id_number),
                "service_name": find_identifier(raw_text, service_name),
            }
            response["matches"] = matches

            # 7. Confidence scoring
            confidence = self._calculate_affidavit_confidence(vision_result, declaration_results, matches)
            response["confidence_score"] = confidence

            if confidence >= AFFIDAVIT_CONFIDENCE_APPROVE_THRESHOLD:
                response["success"] = True
                response["verification_status"] = "verified"
                response["message"] = "Affidavit verified successfully."
            else:
                response["success"] = True  # the check itself ran fine; the affidavit just didn't pass
                response["verification_status"] = "rejected"
                response["message"] = "Affidavit did not meet verification requirements."
                response["error_type"] = "affidavit_verification_error"

            return response

    @staticmethod
    def _calculate_affidavit_confidence(
        vision_result: Dict,
        declaration_results: Dict,
        matches: Dict,
    ) -> float:
        weights = {
            "declarations": 0.40,
            "identity_matches": 0.35,
            "vision_quality": 0.15,
            "markings": 0.10,
        }

        categories = list(declaration_results.values())
        declaration_score = (
            sum(1 for c in categories if c["matched"]) / len(categories)
            if categories else 0.0
        )

        identity_score = (
            sum(1 for v in matches.values() if v) / len(matches)
            if matches else 0.0
        )

        quality_checks = [
            vision_result.get("blur", {}).get("passed", False),
            vision_result.get("brightness", {}).get("passed", False),
            vision_result.get("rotation", {}).get("passed", False),
            vision_result.get("document_edges", {}).get("passed", False),
        ]
        vision_score = sum(1 for c in quality_checks if c) / len(quality_checks)

        has_stamp = vision_result.get("stamp", {}).get("detected", False)
        has_signature = len(vision_result.get("signatures", [])) > 0
        markings_score = sum([has_stamp, has_signature]) / 2

        confidence = (
            declaration_score * weights["declarations"]
            + identity_score * weights["identity_matches"]
            + vision_score * weights["vision_quality"]
            + markings_score * weights["markings"]
        )

        return round(float(confidence), 4)
    
# =========================================================
    # ID document verification (Smart ID Card or green ID book)
    # =========================================================

    def verify_id_document(
            self,
            document_path: str,
            full_name: str,
            id_number: str,
        ) -> Dict:
            """
            OCR the ID document, confirm it's a card or green book, extract
            the ID number, validate its checksum, derive DOB/gender/
            citizenship, and cross-check name + ID number against what the
            customer submitted at signup.

            Green ID books get lenient handling for OCR digit-confusion
            (when extracting the ID number) and fuzzy name matching, since
            older paper stock and typewriter-style print produce more OCR
            noise than the Smart ID Card. The Luhn checksum on the ID
            number itself is never relaxed — leniency only affects how we
            arrive at a candidate number/name, not what counts as a match.
            """
            response = {
                "success": False,
                "verification_status": "rejected",
                "confidence_score": 0.0,
                "ocr": {},
                "extracted": {},
                "matches": {},
                "message": "",
                "error_type": None,
                "verified_via_lenient_path": False,
            }

            ocr_result = self.document_processor.process(document_path)
            response["ocr"] = {
                "success": ocr_result["success"],
                "document_type": ocr_result["document_type"],
                "page_count": ocr_result["page_count"],
                "error": ocr_result["error"],
            }

            if not ocr_result["success"]:
                response["message"] = ocr_result["error"] or "Could not read the ID document."
                response["error_type"] = "ocr_error"
                return response

            if ocr_result["document_type"] not in ("id_card", "green_book"):
                response["message"] = (
                    f"Expected an ID card or green ID book, got "
                    f"'{ocr_result['document_type']}'."
                )
                response["error_type"] = "document_type_error"
                return response

            raw_text = ocr_result["raw_text"]
            is_green_book = ocr_result["document_type"] == "green_book"

            # ID number extraction — lenient mode only kicks in for green
            # books, and only retries with OCR digit-confusion correction
            # after a clean pass fails. Luhn validation is identical either way.
            extracted_id_number = extract_id_number(raw_text, lenient=is_green_book)
            response["extracted"]["id_number"] = extracted_id_number

            if extracted_id_number is None:
                response["message"] = "Could not locate a valid ID number on the document."
                response["error_type"] = "id_extraction_error"
                return response

            id_number_matches = extracted_id_number == id_number.strip()

            # Name matching — exact for ID cards, fuzzy (similarity-based)
            # for green books to tolerate spacing/case/OCR noise.
            if is_green_book:
                name_match = fuzzy_name_match(raw_text, full_name)
            else:
                name_match = find_identifier(raw_text, full_name)

            response["matches"] = {
                "id_number": id_number_matches,
                "full_name": name_match,
            }

            try:
                response["extracted"].update(parse_sa_id_number(extracted_id_number))
            except ValueError as e:
                response["message"] = str(e)
                response["error_type"] = "id_parsing_error"
                return response

            if id_number_matches and name_match:
                response["success"] = True
                response["verification_status"] = "verified"
                response["confidence_score"] = 1.0
                response["message"] = "ID document verified successfully."
                response["verified_via_lenient_path"] = is_green_book
            else:
                response["success"] = True
                response["verification_status"] = "rejected"
                response["confidence_score"] = round(sum([id_number_matches, name_match]) / 2, 4)
                response["message"] = "ID document details did not match the submitted information."
                response["error_type"] = "id_verification_error"

            return response