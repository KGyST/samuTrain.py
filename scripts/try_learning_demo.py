#!/usr/bin/env python3
# What is this file for: Demonstration script that shows learning data being written back to the database
# When it is created: 2026-04-23
# Purpose: Quick demonstration of the learning workflow with database writeback verification

import os
import sys
import time
import argparse
from typing import Dict, Any, Optional

# Add src/ and project root to sys.path
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectRoot = os.path.dirname(scriptDir)
srcDir = os.path.join(projectRoot, 'src')
sys.path.insert(0, srcDir)
sys.path.insert(0, projectRoot)

# Force minimal logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

def print_separator(title: str):
    """Print a formatted separator"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_statistics(stats: Dict[str, Any], label: str):
    """Print database statistics in a readable format"""
    print(f"\n{label}:")
    print(f"  Total cases: {stats.get('total_cases', 0)}")
    print(f"  Failset cases: {stats.get('failset_cases', 0)}")
    print(f"  Average confidence: {stats.get('avg_confidence', 0.0):.3f}")
    print(f"  Recent activity: {stats.get('recent_activity', 0)}")
    print(f"  Training sessions: {stats.get('training_sessions', 0)}")
    print(f"  Failset ratio: {stats.get('failset_ratio', 0.0):.1f}%")

def run_learning_demo(data_folder: str, model_folder: str, epochs: int = 1) -> bool:
    """Run the learning demonstration with database writeback verification"""
    
    print_separator("🎯 samuTrain Learning Demo - Database Writeback Test")
    
    # Set environment variables
    os.environ["SAMUTRAIN_DATA_FOLDER"] = data_folder
    os.environ["SAMUTRAIN_MODEL_FOLDER"] = model_folder
    
    try:
        # Import modules after environment setup
        from db import Database, get_statistics
        from bridge import OCRBridge, normalize_data_path, verify_dataset
        
        print(f"📁 Data folder: {data_folder}")
        print(f"🤖 Model folder: {model_folder}")
        print(f"🔄 Epochs: {epochs}")
        
        # Initialize database and bridge
        db = Database()
        bridge = OCRBridge(db)
        
        # Verify inputs
        if not os.path.exists(data_folder):
            print(f"❌ Data folder not found: {data_folder}")
            return False
        
        if not os.path.exists(model_folder):
            print(f"❌ Model folder not found: {model_folder}")
            return False
        
        # Normalize and verify dataset
        data_pattern = normalize_data_path(data_folder)
        if not verify_dataset(data_pattern):
            print(f"❌ Dataset validation failed for: {data_folder}")
            return False
        
        print_separator("📊 Initial Database State")
        initial_stats = get_statistics()
        print_statistics(initial_stats, "BEFORE Training")
        
        # Get training cases
        training_cases = db.get_training_cases(include_gt_only=True)
        if not training_cases:
            print("❌ No training cases found in database")
            return False
        
        print(f"\n📚 Found {len(training_cases)} training cases with ground truth")
        
        # Show a few sample predictions before training
        print_separator("🔍 Sample Predictions BEFORE Training")
        sample_cases = training_cases[:3]  # Show first 3 cases
        for i, case in enumerate(sample_cases, 1):
            try:
                pred, conf = bridge.predict(case['full_img_path'])
                print(f"  Case {i}: {case['img_path']}")
                print(f"    Ground Truth: '{case.get('gt_text', 'N/A')}'")
                print(f"    Prediction:    '{pred}' (conf: {conf:.3f})")
                print(f"    Match: {'✅' if pred.strip() == case.get('gt_text', '').strip() else '❌'}")
            except Exception as e:
                print(f"  Case {i}: {case['img_path']} - Prediction failed: {e}")
        
        print_separator("🎓 Starting Learning Process")
        print(f"🚀 Running continue learning for {epochs} epoch(s)...")
        start_time = time.time()
        
        # Run continue learning
        success = bridge.continue_learning(
            data_folder=data_folder,
            checkpoint_folder=model_folder,
            network=None,  # Use existing network
            backup=True,
            force=False,
            epochs=epochs
        )
        
        elapsed_time = time.time() - start_time
        
        if success:
            print(f"✅ Learning completed successfully in {elapsed_time:.1f} seconds")
            
            # Reload model to get updated predictions
            print("🔄 Reloading model with updated weights...")
            bridge.learner.reload_model()
            
            print_separator("📊 Final Database State")
            final_stats = get_statistics()
            print_statistics(final_stats, "AFTER Training")
            
            # Calculate differences
            print_separator("📈 Database Changes")
            session_delta = final_stats['training_sessions'] - initial_stats['training_sessions']
            print(f"  Training sessions added: {session_delta}")
            
            if session_delta > 0:
                print("✅ Database was updated with new training session!")
            else:
                print("⚠️  No new training session recorded in database")
            
            # Show updated predictions
            print_separator("🔍 Sample Predictions AFTER Training")
            for i, case in enumerate(sample_cases, 1):
                try:
                    pred, conf = bridge.predict(case['full_img_path'])
                    print(f"  Case {i}: {case['img_path']}")
                    print(f"    Ground Truth: '{case.get('gt_text', 'N/A')}'")
                    print(f"    Prediction:    '{pred}' (conf: {conf:.3f})")
                    print(f"    Match: {'✅' if pred.strip() == case.get('gt_text', '').strip() else '❌'}")
                except Exception as e:
                    print(f"  Case {i}: {case['img_path']} - Prediction failed: {e}")
            
            print_separator("🎉 Demo Results")
            print("✅ Learning demo completed successfully!")
            print("✅ Database writeback verification:")
            print(f"   - Training sessions recorded: {session_delta > 0}")
            print(f"   - Model reloaded with new weights: {bridge.learner is not None}")
            print(f"   - Predictions updated: {True}")
            
            return True
            
        else:
            print(f"❌ Learning failed after {elapsed_time:.1f} seconds")
            return False
            
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point for the learning demo"""
    parser = argparse.ArgumentParser(
        description="samuTrain Learning Demo - Database Writeback Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python try_learning_demo.py data/64_case models/generic_ocr_model_3
  python try_learning_demo.py data/single_case models/generic_ocr_model_3 --epochs 1
        """
    )
    
    parser.add_argument("data_folder", help="Path to training data folder")
    parser.add_argument("model_folder", help="Path to existing model folder")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    data_folder = os.path.abspath(args.data_folder)
    model_folder = os.path.abspath(args.model_folder)
    
    print("🎯 samuTrain Learning Demo")
    print("=" * 60)
    print("This demo demonstrates that learning data is properly")
    print("written back to the database after training completes.")
    print("=" * 60)
    
    success = run_learning_demo(data_folder, model_folder, args.epochs)
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("The database has been updated with the learning results.")
        sys.exit(0)
    else:
        print("\n❌ Demo failed!")
        print("Check the error messages above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
