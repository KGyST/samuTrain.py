from typing import List, Dict, Any, Optional
from src.db import Database

class CaseRepository:
    def __init__(self, db: Database):
        self.db = db

    def get_all(self, limit: int = 100, failset_only: bool = False) -> List[Dict[str, Any]]:
        return self.db.get_cases(limit=limit, failset_only=failset_only)

    def get_by_id(self, case_id: int) -> Optional[Dict[str, Any]]:
        return self.db.get_case_by_id(case_id)

    def get_by_img_path(self, img_path: str) -> Optional[Dict[str, Any]]:
        return self.db.get_case_by_img_path(img_path)

    def create(self, img_path: str, ocr_text: str, confidence: float,
               gt_text: Optional[str] = None, is_failset: bool = False,
               model_prediction: Optional[str] = None) -> int:
        return self.db.insert_case(
            img_path=img_path,
            ocr_text=ocr_text,
            model_prediction=model_prediction,
            gt_text=gt_text,
            confidence=confidence,
            is_failset=is_failset
        )

    def update_correction(self, case_id: int, corrected_text: str) -> bool:
        return self.db.update_case_correction(case_id, corrected_text)

    def update_ocr_result(self, case_id: int, ocr_text: str, confidence: float) -> bool:
        return self.db.update_case_ocr_result(case_id, ocr_text, confidence)

    def delete_all(self) -> bool:
        return self.db.delete_all_cases()

    def get_statistics(self) -> Dict[str, Any]:
        return self.db.get_statistics()
