#!/usr/bin/env python3

import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, './src')

from bridge import OCRBridge

def test_training():
    """Test the training functionality with sample data"""
    print("🧪 Testing training functionality...")
    
    # Create a temporary training directory with sample data
    temp_dir = tempfile.mkdtemp(prefix="test_train_")
    print(f"📁 Using temp dir: {temp_dir}")
    
    try:
        # Copy some sample data
        sample_images = [
            "data/64_case/010001.bin.png",
            "data/64_case/010002.bin.png"
        ]
        
        count = 0
        for img_path in sample_images:
            if os.path.exists(img_path):
                dst = os.path.join(temp_dir, os.path.basename(img_path))
                shutil.copy2(img_path, dst)
                
                # Create corresponding GT file
                gt_src = img_path.replace('.bin.png', '.gt.txt')
                gt_dst = dst.replace('.bin.png', '.gt.txt')
                if os.path.exists(gt_src):
                    shutil.copy2(gt_src, gt_dst)
                else:
                    # Create a dummy GT file
                    with open(gt_dst, 'w', encoding='utf-8') as f:
                        f.write("sample text")
                count += 1
        
        print(f"📋 Prepared {count} training samples")
        
        if count > 0:
            # Test the bridge training
            bridge = OCRBridge()
            success = bridge.train_on_failset(temp_dir)
            
            if success:
                print("✅ Training test successful!")
            else:
                print("❌ Training test failed!")
        else:
            print("⚠️ No sample data found for testing")
            
    except Exception as e:
        print(f"❌ Test error: {e}")
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"🧹 Cleaned up temp dir")

if __name__ == "__main__":
    test_training()
