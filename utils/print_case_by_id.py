#!/usr/bin/env python3
"""
samuTrain Print Case by ID Tool
Prints detailed information about a single case identified by ID
"""

import sys
import os
import argparse
from typing import Optional

# Add src directory to path to import db module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from db import get_case_by_id, _resolve_case_image_file


def print_case_details(case_id: int, db_path: str = "samu.db") -> bool:
    """Print detailed information about a single case"""
    try:
        # Import Database class to use the specified db_path
        from db import Database
        
        # Create database instance with specified path
        db = Database(db_path)
        
        # Get case by ID
        case = db.get_case_by_id(case_id)
        
        if not case:
            print(f"❌ Case with ID {case_id} not found in database: {db_path}")
            return False
        
        print(f"📋 Case Details (ID: {case_id})")
        print("=" * 50)
        print(f"ID:              {case['id']}")
        print(f"Image Path:      {case['img_path']}")
        
        # Resolve and show actual file path
        try:
            full_img_path = _resolve_case_image_file(case['img_path'])
            file_exists = os.path.exists(full_img_path)
            print(f"Full Image Path: {full_img_path}")
            print(f"File Exists:     {file_exists}")
        except Exception as e:
            print(f"Full Image Path: Error resolving path: {e}")
        
        print(f"OCR Text:        '{case['ocr_text']}'")
        print(f"Model Prediction: '{case.get('model_prediction', 'N/A')}'")
        print(f"GT Text:         '{case.get('gt_text', 'N/A')}'")
        print(f"Confidence:      {case['confidence']}")
        print(f"Is Failset:      {bool(case['is_failset'])}")
        print(f"Is Corrected:    {bool(case.get('is_corrected', False))}")
        print(f"Updated Flag:    {bool(case.get('updated_flag', False))}")
        print(f"Timestamp:       {case['timestamp']}")
        print(f"Last Updated:    {case.get('last_updated', 'N/A')}")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error retrieving case {case_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Print case details by ID')
    parser.add_argument('case_id', type=int, help='Case ID to retrieve')
    parser.add_argument('--db', default='samu.db', help='Database file path (default: samu.db)')
    
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists(args.db):
        print(f"❌ Database file not found: {args.db}")
        sys.exit(1)
    
    success = print_case_details(args.case_id, args.db)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
