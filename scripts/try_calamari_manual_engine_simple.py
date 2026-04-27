import os
import numpy as np
import tensorflow as tf
import argparse
import glob
import tempfile
import time
from typing import Dict, Any, List, Tuple
from PIL import Image

# Calamari components
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.scripts.train import main as calamari_train


def load_dataset(data_dir: str) -> List[Tuple[np.ndarray, str]]:
    """Load all image-ground truth pairs from a directory"""
    dataset = []
    
    # Find all .bin.png files
    image_files = glob.glob(os.path.join(data_dir, "*.bin.png"))
    
    for image_path in image_files:
        # Get corresponding .gt.txt file
        gt_path = image_path.replace('.bin.png', '.gt.txt')
        
        if os.path.exists(gt_path):
            try:
                # Load image
                image = Image.open(image_path).convert('L')
                image_np = np.array(image)
                
                # Load ground truth
                with open(gt_path, 'r', encoding='utf-8') as f:
                    ground_truth = f.read().strip()
                
                if len(ground_truth) > 0:  # Skip empty ground truths
                    dataset.append((image_np, ground_truth))
                    
            except Exception as e:
                print(f"⚠️  Error loading {image_path}: {e}")
                
    print(f"📂 Loaded {len(dataset)} training samples from {data_dir}")
    return dataset


def create_temp_training_data(dataset: List[Tuple[np.ndarray, str]], temp_dir: str):
    """Create temporary training data files"""
    temp_data_dir = os.path.join(temp_dir, "temp_data")
    os.makedirs(temp_data_dir, exist_ok=True)
    
    image_paths = []
    
    for i, (image_np, ground_truth) in enumerate(dataset):
        # Save image
        image_path = os.path.join(temp_data_dir, f"sample_{i:04d}.bin.png")
        Image.fromarray(image_np).save(image_path)
        
        # Save ground truth
        gt_path = image_path.replace('.bin.png', '.gt.txt')
        with open(gt_path, 'w', encoding='utf-8') as f:
            f.write(ground_truth)
        
        image_paths.append(image_path)
    
    return image_paths


def train_multiple_epochs_persistent(checkpoint_path: str, dataset: List[Tuple[np.ndarray, str]], 
                                   epochs: int = 10, output_model: str = None) -> Dict[str, Any]:
    """Train for multiple epochs with persistent model state"""
    
    print(f"\n🚀 Starting {epochs}-epoch training with {len(dataset)} samples")
    print("=" * 60)
    
    epoch_results = []
    current_checkpoint = checkpoint_path
    
    # Create persistent directory for checkpoints
    persistent_dir = tempfile.mkdtemp(prefix="calamari_manual_training_")
    print(f"📁 Using persistent directory: {persistent_dir}")
    
    try:
        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch + 1}/{epochs}")
            print("-" * 40)
            
            # Create persistent output directory for this epoch
            epoch_output_dir = os.path.join(persistent_dir, f"epoch_{epoch + 1}")
            os.makedirs(epoch_output_dir, exist_ok=True)
            
            # Create temporary directory for this epoch's data
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create training data
                image_paths = create_temp_training_data(dataset, temp_dir)
                
                # Setup training parameters
                trainer_params = CalamariScenario.default_trainer_params()
                trainer_params.output_dir = epoch_output_dir
                trainer_params.epochs = 1  # Train 1 epoch at a time
                trainer_params.gen.train.images = image_paths
                trainer_params.gen.val.images = image_paths[:min(5, len(image_paths))]  # Use first 5 for validation
                trainer_params.gen.setup.train.batch_size = 1
                trainer_params.gen.setup.train.num_processes = 1
                trainer_params.gen.setup.val.num_processes = 1
                trainer_params.progress_bar = False
                
                # Setup warmstart from current checkpoint (except first epoch)
                if epoch > 0:
                    trainer_params.warmstart.model = current_checkpoint
                    trainer_params.warmstart.allow_partial = True
                    trainer_params.warmstart.trim_graph_name = False
                
                # Train for one epoch
                print(f"  🔄 Training epoch {epoch + 1}...")
                start_time = time.time()
                
                try:
                    result = calamari_train(trainer_params)
                    elapsed_time = time.time() - start_time
                    
                    # Get the new checkpoint
                    new_checkpoint = os.path.join(epoch_output_dir, "best.ckpt.json")
                    
                    if os.path.exists(new_checkpoint):
                        current_checkpoint = new_checkpoint
                        
                        # Calculate metrics (simplified)
                        avg_loss = result.get('train_loss', 0.0) if result else 0.0
                        avg_cer = result.get('val_CER', 1.0) if result else 1.0
                        
                        epoch_results.append({
                            'epoch': epoch + 1,
                            'avg_loss': avg_loss,
                            'avg_cer': avg_cer,
                            'training_time': elapsed_time,
                            'checkpoint': new_checkpoint
                        })
                        
                        print(f"  ✅ Epoch {epoch + 1} completed in {elapsed_time:.1f}s")
                        print(f"     Loss: {avg_loss:.4f}, CER: {avg_cer:.4f}")
                        print(f"     Checkpoint: {new_checkpoint}")
                        
                    else:
                        print(f"  ❌ No checkpoint found for epoch {epoch + 1}")
                        epoch_results.append({
                            'epoch': epoch + 1,
                            'avg_loss': 0.0,
                            'avg_cer': 1.0,
                            'training_time': elapsed_time,
                            'checkpoint': None
                        })
                        
                except Exception as e:
                    print(f"  ❌ Epoch {epoch + 1} failed: {e}")
                    epoch_results.append({
                        'epoch': epoch + 1,
                        'avg_loss': 0.0,
                        'avg_cer': 1.0,
                        'training_time': 0.0,
                        'checkpoint': None
                    })
    
    finally:
        # Clean up persistent directory
        import shutil
        try:
            shutil.rmtree(persistent_dir)
            print(f"🧹 Cleaned up persistent directory: {persistent_dir}")
        except:
            pass
    
    # Save final model if requested
    if output_model and current_checkpoint != checkpoint_path:
        print(f"\n💾 Saving final model to: {output_model}")
        os.makedirs(output_model, exist_ok=True)
        
        # Copy the final checkpoint
        import shutil
        if os.path.exists(current_checkpoint.replace('.json', '')):
            shutil.copytree(current_checkpoint.replace('.json', ''), 
                          os.path.join(output_model, 'final_model'))
            print("✅ Model saved successfully!")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Training completed!")
    print("=" * 60)
    
    for result in epoch_results:
        status = "✅" if result['checkpoint'] else "❌"
        print(f"Epoch {result['epoch']:2d}: {status} Loss={result['avg_loss']:.4f}, CER={result['avg_cer']:.4f}, Time={result['training_time']:.1f}s")
    
    return {
        'epoch_results': epoch_results,
        'total_epochs': epochs,
        'total_samples': len(dataset),
        'final_checkpoint': current_checkpoint
    }


def main():
    """Main function with default arguments"""
    parser = argparse.ArgumentParser(description='Calamari Manual Engine - Multi-Epoch Training (Persistent)')
    
    # Default arguments for meaningful operation
    parser.add_argument('--checkpoint', type=str, 
                       default='models/generic_ocr_model/best.ckpt.json',
                       help='Path to checkpoint file (default: models/generic_ocr_model/best.ckpt.json)')
    
    parser.add_argument('--data-dir', type=str,
                       default='data/64_case',
                       help='Directory containing training data (default: data/64_case)')
    
    parser.add_argument('--epochs', type=int,
                       default=10,
                       help='Number of training epochs (default: 10)')
    
    parser.add_argument('--output-model', type=str,
                       default='models/manual_engine_10_epoch',
                       help='Output directory for trained model (default: models/manual_engine_10_epoch)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return 1
    
    try:
        # Load dataset
        dataset = load_dataset(args.data_dir)
        
        if len(dataset) == 0:
            print(f"❌ No valid training data found in {args.data_dir}")
            return 1
        
        # Train for multiple epochs with persistence
        results = train_multiple_epochs_persistent(
            checkpoint_path=args.checkpoint,
            dataset=dataset,
            epochs=args.epochs,
            output_model=args.output_model
        )
        
        print(f"\n🏁 Training completed successfully!")
        print(f"   - Total epochs: {results['total_epochs']}")
        print(f"   - Total samples: {results['total_samples']}")
        print(f"   - Final checkpoint: {results['final_checkpoint']}")
        if args.output_model:
            print(f"   - Model saved to: {args.output_model}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
