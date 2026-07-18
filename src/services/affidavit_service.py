from datetime import date

from src.api.kyc_processor import KYCProcessor


class AffidavitService:
    @staticmethod
    def verify_affidavit(
        affidavit_path: str,
        full_name: str,
        id_number: str,
        service_name: str,
        upload_date: date = None,
    ):
        processor = KYCProcessor()
        return processor.verify_affidavit(
            document_path=affidavit_path,
            full_name=full_name,
            id_number=id_number,
            service_name=service_name,
            upload_date=upload_date,
        )