# What is this file for: Backend emulation script that validates learning workflow and database integration
# When it is created: 2026-03-27
# Rewritten to maximize reuse of existing src/ functions

import os
import sys
import argparse
from datetime import datetime

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

# Set environment for minimal logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Set UTF-8 encoding for stdout to handle emoji characters
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

def validate_data_folder(data_folder):
    """Validate that data folder exists and contains training data using existing functions"""
    try:
        from src.bridge import verify_dataset, normalize_data_path
        
        if not os.path.exists(data_folder):
            print(f"❌ Data folder not found: {data_folder}")
            return False
        
        if not os.path.isdir(data_folder):
            print(f"❌ Data path is not a directory: {data_folder}")
            return False
        
        # Use existing verification function
        normalized_path = normalize_data_path(data_folder)
        if verify_dataset(normalized_path):
            # Count files for reporting
            import glob
            bin_files = glob.glob(os.path.join(data_folder, "*.bin.png"))
            gt_files = glob.glob(os.path.join(data_folder, "*.gt.txt"))
            print(f"✅ Data folder validated: {len(bin_files)} images, {len(gt_files)} GT files")
            return True
        else:
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def validate_model_folder(model_folder):
    """Validate that model folder exists and contains required files"""
    if not os.path.exists(model_folder):
        print(f"❌ Model folder not found: {model_folder}")
        return False
    
    if not os.path.isdir(model_folder):
        print(f"❌ Model path is not a directory: {model_folder}")
        return False
    
    # Check for essential model files
    required_files = ["best.ckpt.json", "trainer_params.json"]
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_folder, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        return False
    
    print(f"✅ Model folder validated: {model_folder}")
    return True

def get_database_stats():
    """Get current database statistics using existing db function"""
    try:
        from src.db import db
        return db.get_statistics()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return {
            'total_cases': 0,
            'failset_cases': 0,
            'avg_confidence': 0.0,
            'recent_activity': 0,
            'training_sessions': 0
        }

def count_predictions():
    """Count cases with predictions in database using existing db function"""
    try:
        from src.db import db
        with db.get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as count,
                       COUNT(CASE WHEN model_prediction IS NOT NULL THEN 1 END) as with_pred,
                       COUNT(CASE WHEN confidence > 0 THEN 1 END) as with_conf
                FROM cases
            """)
            result = cursor.fetchone()
            return {
                'total': result['count'],
                'with_predictions': result['with_pred'],
                'with_confidence': result['with_conf']
            }
    except Exception as e:
        print(f"❌ Error counting predictions: {e}")
        return {'total': 0, 'with_predictions': 0, 'with_confidence': 0}

def validate_learning_workflow(data_folder, model_folder):
    """Main validation function for learning workflow using existing src functions"""
    print("🔧 Backend Emulation - Learning Workflow Validation")
    print("=" * 60)
    
    # Import required modules from src
    try:
        from src.bridge import OCRBridge
        from src.db import db
        print("✅ Successfully imported required modules")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Validate inputs
    if not validate_data_folder(data_folder):
        return False
    
    if not validate_model_folder(model_folder):
        return False
    
    # Set environment variables for database operations
    os.environ['SAMUTRAIN_DATA_FOLDER'] = data_folder
    os.environ['SAMUTRAIN_MODEL_FOLDER'] = model_folder
    
    print("\n📊 Initial Database State:")
    initial_stats = get_database_stats()
    initial_predictions = count_predictions()
    
    print(f"   Total cases: {initial_stats['total_cases']}")
    print(f"   Failset cases: {initial_stats['failset_cases']}")
    print(f"   Cases with predictions: {initial_predictions['with_predictions']}")
    print(f"   Cases with confidence: {initial_predictions['with_confidence']}")
    print(f"   Average confidence: {initial_stats['avg_confidence']}")
    print(f"   Training sessions: {initial_stats['training_sessions']}")
    
    # Initialize OCRBridge using existing class
    print("\n🔧 Initializing OCRBridge...")
    try:
        bridge = OCRBridge()
        print(f"✅ OCRBridge initialized with model: {bridge.model_dir}")
        
        # Test prediction capability using existing method
        test_img = os.path.join(data_folder, os.listdir(data_folder)[0])
        if test_img.endswith('.bin.png'):
            pred, conf = bridge.predict(test_img)
            print(f"✅ Test prediction successful: '{pred}' (confidence: {conf:.3f})")
        
    except Exception as e:
        print(f"❌ OCRBridge initialization failed: {e}")
        return False
    
    # Get training cases from database using existing function
    print("\n📚 Checking training cases in database...")
    training_cases = db.get_training_cases(include_gt_only=True)
    
    if not training_cases:
        print("❌ No training cases found in database")
        print("   Note: This script expects data to be loaded into samu.db first")
        return False
    
    print(f"✅ Found {len(training_cases)} training cases in database")
    
    # Run continue learning using existing OCRBridge method
    print("\n🎓 Starting Continue Learning...")
    print(f"   Data folder: {data_folder}")
    print(f"   Model folder: {model_folder}")
    
    try:
        # Record session start time
        start_time = datetime.now()
        
        # Use existing OCRBridge.continue_learning method
        success = bridge.continue_learning(
            data_folder=data_folder,
            checkpoint_folder=model_folder,
            network=None,  # Use existing network
            backup=True,
            force=False
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if success:
            print(f"✅ Continue learning completed successfully in {duration:.1f} seconds")
        else:
            print("❌ Continue learning failed")
            return False
            
    except Exception as e:
        print(f"❌ Continue learning error: {e}")
        return False
    
    # Validate database updates using existing functions
    print("\n🔍 Validating Database Updates...")
    
    final_stats = get_database_stats()
    final_predictions = count_predictions()
    
    print("\n📊 Final Database State:")
    print(f"   Total cases: {final_stats['total_cases']}")
    print(f"   Failset cases: {final_stats['failset_cases']}")
    print(f"   Cases with predictions: {final_predictions['with_predictions']}")
    print(f"   Cases with confidence: {final_predictions['with_confidence']}")
    print(f"   Average confidence: {final_stats['avg_confidence']}")
    print(f"   Training sessions: {final_stats['training_sessions']}")
    
    # Check for changes
    print("\n📈 Database Changes:")
    pred_change = final_predictions['with_predictions'] - initial_predictions['with_predictions']
    conf_change = final_predictions['with_confidence'] - initial_predictions['with_confidence']
    session_change = final_stats['training_sessions'] - initial_stats['training_sessions']
    
    print(f"   New predictions: +{pred_change}")
    print(f"   New confidence scores: +{conf_change}")
    print(f"   New training sessions: +{session_change}")
    
    # Validate learning effectiveness
    validation_passed = True
    
    if session_change <= 0:
        print("❌ No new training session recorded")
        validation_passed = False
    
    if pred_change <= 0 and conf_change <= 0:
        print("⚠️  No new predictions or confidence scores detected")
        print("   This might indicate the model was already up-to-date")
    else:
        print("✅ Database was updated with new learning data")
    
    # Test model reloading using existing OCRBridge method
    print("\n🔄 Testing Model Reload...")
    try:
        bridge.learner.reload_model()
        print("✅ Model reloaded successfully")
        
        # Test prediction after reload using existing method
        if test_img.endswith('.bin.png'):
            pred_after, conf_after = bridge.predict(test_img)
            print(f"✅ Post-reload prediction: '{pred_after}' (confidence: {conf_after:.3f})")
        
    except Exception as e:
        print(f"❌ Model reload failed: {e}")
        validation_passed = False
    
    # Final validation result
    print("\n" + "=" * 60)
    if validation_passed:
        print("🎉 BACKEND EMULATION VALIDATION PASSED")
        print("   ✓ Learning workflow completed successfully")
        print("   ✓ Database integration working correctly")
        print("   ✓ Model reloading functional")
        print("   ✓ Training sessions tracked properly")
    else:
        print("❌ BACKEND EMULATION VALIDATION FAILED")
        print("   Some validation checks failed")
    
    print("=" * 60)
    return validation_passed

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Backend emulation script for learning workflow validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python try_backend_emulation.py data/64_case models/generic_ocr_model_3
  python try_backend_emulation.py --force data/new_case models/existing_model
        """
    )
    
    parser.add_argument("data_folder", help="Path to training data folder")
    parser.add_argument("model_folder", help="Path to existing model folder")
    parser.add_argument("--force", action="store_true", help="Force validation even if checks fail")
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    data_folder = os.path.abspath(args.data_folder)
    model_folder = os.path.abspath(args.model_folder)
    
    print(f"🔧 Backend Emulation Script")
    print(f"   Data folder: {data_folder}")
    print(f"   Model folder: {model_folder}")
    print(f"   Force mode: {args.force}")
    print()
    
    # Run validation
    success = validate_learning_workflow(data_folder, model_folder)
    
    if success:
        print("\n✅ Backend emulation completed successfully")
        sys.exit(0)
    else:
        print("\n❌ Backend emulation failed")
        if args.force:
            print("   (Force mode enabled - continuing despite failures)")
            sys.exit(0)
        else:
            print("   Use --force to continue despite validation failures")
            sys.exit(1)

if __name__ == "__main__":
    main()
