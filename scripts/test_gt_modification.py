#!/usr/bin/env python3
"""
Test script for ground truth modification functionality.
Tests the complete flow: Database update -> Disk write -> Verification
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from db import Database

def test_gt_modification():
    """Test ground truth modification with various scenarios"""
    print("🧪 Testing Ground Truth Modification Functionality")
    print("=" * 60)
    
    # Create temporary test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        test_db_path = os.path.join(temp_dir, "test_samu.db")
        test_data_dir = os.path.join(temp_dir, "data")
        test_subdir = os.path.join(test_data_dir, "test_subdir")
        
        # Create test directories
        os.makedirs(test_subdir, exist_ok=True)
        
        # Set environment for test
        os.environ['SAMUTRAIN_DATA_FOLDER'] = test_data_dir
        
        # Initialize test database
        db = Database(test_db_path)
        print(f"✅ Test database initialized: {test_db_path}")
        print(f"✅ Test data directory: {test_data_dir}")
        
        # Test Case 1: Simple file in main data directory
        print("\n📝 Test Case 1: Simple file in main directory")
        case1_id = db.insert_case(
            img_path="test001.bin.png",
            ocr_text="initial_ocr",
            confidence=0.8,
            gt_text="initial_gt"
        )
        print(f"✅ Created test case {case1_id}")
        
        # Test correction
        success = db.update_case_correction(case1_id, "corrected_gt_1")
        gt_file_path = os.path.join(test_data_dir, "test001.gt.txt")
        
        if success and os.path.exists(gt_file_path):
            with open(gt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if content == "corrected_gt_1":
                print("✅ Test Case 1 PASSED: File written correctly")
            else:
                print(f"❌ Test Case 1 FAILED: Expected 'corrected_gt_1', got '{content}'")
        else:
            print(f"❌ Test Case 1 FAILED: Correction failed or file not created")
        
        # Test Case 2: File in subdirectory
        print("\n📝 Test Case 2: File in subdirectory")
        case2_id = db.insert_case(
            img_path="test_subdir/test002.bin.png",
            ocr_text="initial_ocr",
            confidence=0.8,
            gt_text="initial_gt"
        )
        print(f"✅ Created test case {case2_id}")
        
        # Test correction
        success = db.update_case_correction(case2_id, "corrected_gt_2")
        gt_file_path = os.path.join(test_subdir, "test002.gt.txt")
        
        if success and os.path.exists(gt_file_path):
            with open(gt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if content == "corrected_gt_2":
                print("✅ Test Case 2 PASSED: Subdirectory file written correctly")
            else:
                print(f"❌ Test Case 2 FAILED: Expected 'corrected_gt_2', got '{content}'")
        else:
            print(f"❌ Test Case 2 FAILED: Correction failed or file not created")
        
        # Test Case 3: Overwrite existing file
        print("\n📝 Test Case 3: Overwrite existing file")
        # Create existing GT file
        existing_gt_path = os.path.join(test_data_dir, "test003.gt.txt")
        with open(existing_gt_path, 'w', encoding='utf-8') as f:
            f.write("existing_content")
        
        case3_id = db.insert_case(
            img_path="test003.bin.png",
            ocr_text="initial_ocr",
            confidence=0.8,
            gt_text="existing_content"
        )
        
        # Test correction (should overwrite)
        success = db.update_case_correction(case3_id, "overwritten_content")
        
        if success and os.path.exists(existing_gt_path):
            with open(existing_gt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if content == "overwritten_content":
                print("✅ Test Case 3 PASSED: Existing file overwritten correctly")
            else:
                print(f"❌ Test Case 3 FAILED: Expected 'overwritten_content', got '{content}'")
        else:
            print(f"❌ Test Case 3 FAILED: Correction failed")
        
        # Test Case 4: Database verification
        print("\n📝 Test Case 4: Database verification")
        updated_case = db.get_case_by_id(case1_id)
        if (updated_case and 
            updated_case['gt_text'] == "corrected_gt_1" and 
            updated_case['ocr_text'] == "corrected_gt_1" and
            updated_case['is_corrected'] == True):
            print("✅ Test Case 4 PASSED: Database updated correctly")
        else:
            print(f"❌ Test Case 4 FAILED: Database not updated correctly")
            print(f"   Case data: {updated_case}")
        
        # Test Case 5: Error handling - invalid case ID
        print("\n📝 Test Case 5: Error handling - invalid case ID")
        success = db.update_case_correction(99999, "should_fail")
        if not success:
            print("✅ Test Case 5 PASSED: Invalid case ID handled correctly")
        else:
            print("❌ Test Case 5 FAILED: Invalid case ID should have failed")
        
        print("\n" + "=" * 60)
        print("🎯 Ground Truth Modification Testing Complete!")
        print(f"📁 Test files created in: {test_data_dir}")
        print("🗑️  Test directory will be cleaned up automatically")

if __name__ == "__main__":
    test_gt_modification()
