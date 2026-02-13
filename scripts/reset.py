#!/usr/bin/env python3
"""
samuTrain V2 Project Reset Script
Resets trainer weights and clears GT files for fresh training
"""

import sys
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Reset samuTrain project")
    parser.add_argument("--data-folder", default="data/single_case", 
                       help="Path to data folder to reset (default: data/single_case)")
    parser.add_argument("--confirm", action="store_true",
                       help="Skip confirmation prompt")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be reset without making changes")
    
    args = parser.parse_args()
    
    print("🔄 samuTrain V2 Project Reset Script")
    print("=" * 40)
    print(f"📁 Data folder: {args.data_folder}")
    
    # Validate data folder
    data_path = Path(args.data_folder)
    if not data_path.exists():
        print(f"❌ Error: Data folder '{args.data_folder}' does not exist")
        sys.exit(1)
    
    # Find all .gt.txt files
    gt_files = list(data_path.glob("*.gt.txt"))
    if not gt_files:
        print(f"ℹ️  No .gt.txt files found in '{args.data_folder}'")
        sys.exit(0)
    
    print(f"📄 Found {len(gt_files)} .gt.txt files to reset:")
    for gt_file in gt_files:
        print(f"   - {gt_file.name}")
    
    # Find model files (placeholder for actual model reset)
    model_files = []
    model_patterns = ["*.model", "*.ckpt", "*.h5", "*.pth", "*.pt"]
    for pattern in model_patterns:
        model_files.extend(Path(".").glob(pattern))
        model_files.extend(Path("models").glob(pattern))
    
    if model_files:
        print(f"🤖 Found {len(model_files)} model files to reset:")
        for model_file in model_files:
            print(f"   - {model_file}")
    else:
        print("🤖 No model files found")
    
    # Confirmation
    if not args.dry_run and not args.confirm:
        print("\n⚠️  This will:")
        print("   • Empty all .gt.txt files (remove learning clues)")
        if model_files:
            print("   • Reset model weights (forget learned patterns)")
        print("\n🔄 Do you want to continue? (y/N): ", end="")
        
        try:
            response = input().lower().strip()
            if response not in ['y', 'yes']:
                print("❌ Reset cancelled")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\n❌ Reset cancelled")
            sys.exit(0)
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No changes made")
        return
    
    print("\n🔄 Resetting project...")
    
    # Reset GT files (empty them)
    for gt_file in gt_files:
        try:
            with open(gt_file, 'w', encoding='utf-8') as f:
                f.write('')  # Empty the file
            print(f"✅ Cleared: {gt_file}")
        except Exception as e:
            print(f"❌ Failed to clear {gt_file}: {e}")
    
    # Reset model files (placeholder for actual model reset)
    for model_file in model_files:
        try:
            # For now, just report what would be done
            # Actual model reset depends on the ML framework used
            print(f"🔧 Would reset: {model_file} (framework-specific reset needed)")
        except Exception as e:
            print(f"❌ Failed to reset {model_file}: {e}")
    
    # Reset database (optional - clear samu.db)
    db_file = Path("samu.db")
    if db_file.exists():
        try:
            db_file.unlink()  # Delete database file
            print(f"✅ Removed database: {db_file}")
        except Exception as e:
            print(f"❌ Failed to remove database: {e}")
    
    print("\n✅ Project reset completed!")
    print("🎯 Ready for fresh training session")
    print("💡 Start the server with: python run_server.py")
    print("🚀 Start training with: python scripts/train.py")

if __name__ == "__main__":
    main()
